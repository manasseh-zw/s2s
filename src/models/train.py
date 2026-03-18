"""
Focused Shona CSM training loop with AMP, W&B logging, and early stopping.

Adapted from `knottwill/sesame-finetune`, but stripped down to the training
path used by this project.
"""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import torch
import wandb
import yaml
from torch.amp import GradScaler, autocast
from torch.nn.utils import clip_grad_norm_
from tqdm import tqdm

from audit import AuditLogger, collect_environment_snapshot
from dataloaders import create_dataloaders
from utils import (
    MIMI_SAMPLE_RATE,
    WarmupDecayLR,
    generate_audio,
    load_model,
    load_tokenizers,
    load_watermarker,
    validate,
)

DEFAULT_CONFIG = Path(__file__).resolve().parent / "configs" / "shona_finetune.yaml"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--model_name_or_checkpoint_path",
        type=str,
        default="sesame/csm-1b",
    )
    parser.add_argument("--train_from_scratch", action="store_true")
    parser.add_argument("--partial_data_loading", action="store_true")
    parser.add_argument("--wandb_project", type=str, default="shona-csm-finetune")
    parser.add_argument("--wandb_name", type=str, default=None)
    parser.add_argument("--gen_sentences", type=str, default="mhoro, unonzi ani?")
    parser.add_argument("--gen_speaker", type=int, default=999)
    parser.add_argument("--use_amp", action="store_true")
    parser.add_argument("--n_epochs", type=int, default=3)

    args = parser.parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.train_from_scratch:
        args.model_name_or_checkpoint_path = None
    return args


def load_config(config_path: Path) -> dict:
    with config_path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def build_run_manifest(args: argparse.Namespace, config: dict, device: torch.device) -> dict:
    return {
        "args": vars(args),
        "config": config,
        "device": str(device),
        "environment": collect_environment_snapshot(),
    }


def resolve_generation_sentences(raw_value: str) -> list[str]:
    if raw_value.endswith(".txt"):
        with open(raw_value, "r", encoding="utf-8") as handle:
            return [line.strip() for line in handle if line.strip()]
    return [raw_value]


def checkpoint_state(
    model,
    optimizer,
    scheduler,
    scaler,
    config: dict,
    args: argparse.Namespace,
    best_val_loss: float,
    epoch: int,
    step: int,
) -> dict:
    return {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "scaler": scaler.state_dict(),
        "config": config,
        "args": vars(args),
        "best_val_loss": best_val_loss,
        "epoch": epoch,
        "step": step,
        "effective_batch_size": config["batch_size"] * config["grad_acc_steps"],
    }


def save_checkpoint(state: dict, output_dir: Path, filename: str) -> None:
    torch.save(state, output_dir / filename)


def run_generation(
    model,
    audio_tokenizer,
    text_tokenizer,
    watermarker,
    args: argparse.Namespace,
    device: torch.device,
    step: int,
    audit_logger: AuditLogger,
) -> None:
    for index, sentence in enumerate(resolve_generation_sentences(args.gen_sentences)):
        audio = generate_audio(
            model,
            audio_tokenizer,
            text_tokenizer,
            watermarker,
            sentence,
            args.gen_speaker,
            device,
            use_amp=args.use_amp,
        )
        wandb.log(
            {f"audio_{index}": wandb.Audio(audio, sample_rate=MIMI_SAMPLE_RATE)},
            step=step,
        )
        audit_logger.log_event(
            "generation_sample",
            step=step,
            sample_index=index,
            sentence=sentence,
            sample_rate=MIMI_SAMPLE_RATE,
            audio_num_samples=int(len(audio)),
            audio_duration_sec=float(len(audio) / MIMI_SAMPLE_RATE),
        )


def train_loop(
    args: argparse.Namespace,
    config: dict,
    device: torch.device,
    audit_logger: AuditLogger,
) -> float:
    if wandb.run is None:
        raise RuntimeError("W&B must be initialized before calling train_loop")

    model = load_model(
        model_name_or_checkpoint_path=args.model_name_or_checkpoint_path,
        device=device,
        decoder_loss_weight=config["decoder_loss_weight"],
    )
    text_tokenizer, audio_tokenizer = load_tokenizers(device)
    watermarker = load_watermarker(device=device)
    train_loader, val_loader = create_dataloaders(
        args.data,
        batch_size=config["batch_size"],
        load_in_memory=not args.partial_data_loading,
        num_workers=config.get("num_workers", 0),
    )

    total_steps = args.n_epochs * len(train_loader)
    warmup_steps = min(config["warmup_steps"], max(total_steps - 1, 0))

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["learning_rate"],
        weight_decay=config["weight_decay"],
    )
    scheduler = WarmupDecayLR(
        optimizer,
        warmup_steps=warmup_steps,
        total_steps=total_steps,
        decay_type=config["lr_decay"],
    )
    scaler = GradScaler(enabled=args.use_amp)
    optimizer.zero_grad(set_to_none=True)

    best_val_loss = float("inf")
    stale_validations = 0
    early_stopped = False
    train_losses: list[float] = []
    progress = tqdm(total=total_steps, desc="Training")
    audit_logger.log_event(
        "training_started",
        total_steps=total_steps,
        epochs=args.n_epochs,
        train_batches=len(train_loader),
        val_batches=len(val_loader),
        warmup_steps=warmup_steps,
    )

    step = 0
    for epoch in range(args.n_epochs):
        model.train()
        for tokens, tokens_mask in train_loader:
            step_start_time = time.perf_counter()
            tokens = tokens.to(device)
            tokens_mask = tokens_mask.to(device)

            with autocast(device_type=str(device), enabled=args.use_amp):
                loss = model(tokens, tokens_mask)
                scaled_loss = loss / config["grad_acc_steps"]

            scaler.scale(scaled_loss).backward()
            train_losses.append(float(loss.item()))
            grad_norm = None

            if (step + 1) % config["grad_acc_steps"] == 0:
                scaler.unscale_(optimizer)
                grad_norm = float(
                    clip_grad_norm_(model.parameters(), config["max_grad_norm"]).item()
                )
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()

            if step % config["log_every"] == 0:
                train_loss_avg = sum(train_losses) / len(train_losses)
                step_duration = time.perf_counter() - step_start_time
                tokens_per_batch = int(tokens.shape[0])
                sequence_length = int(tokens.shape[1])
                metrics = {
                    "train_loss_avg": train_loss_avg,
                    "train_loss": float(loss.item()),
                    "epoch": epoch,
                    "learning_rate": optimizer.param_groups[0]["lr"],
                    "step_time_sec": step_duration,
                    "samples_per_sec": float(tokens_per_batch / max(step_duration, 1e-6)),
                    "batch_size_actual": tokens_per_batch,
                    "sequence_length": sequence_length,
                }
                if grad_norm is not None:
                    metrics["grad_norm"] = grad_norm
                if torch.cuda.is_available():
                    metrics["cuda_memory_allocated_mb"] = float(
                        torch.cuda.memory_allocated(device) / (1024**2)
                    )
                    metrics["cuda_memory_reserved_mb"] = float(
                        torch.cuda.memory_reserved(device) / (1024**2)
                    )
                wandb.log(
                    metrics,
                    step=step,
                )
                audit_logger.log_event(
                    "train_metrics",
                    step=step,
                    **metrics,
                )
                train_losses.clear()

            if step % config["save_every"] == 0 or step == total_steps - 1:
                state = checkpoint_state(
                    model,
                    optimizer,
                    scheduler,
                    scaler,
                    config,
                    args,
                    best_val_loss,
                    epoch,
                    step,
                )
                checkpoint_name = f"model_step_{step}.pt"
                save_checkpoint(state, args.output_dir, checkpoint_name)
                audit_logger.log_event(
                    "checkpoint_saved",
                    step=step,
                    epoch=epoch,
                    checkpoint=checkpoint_name,
                )

            if step % config["val_every"] == 0 or step == total_steps - 1:
                validation_start = time.perf_counter()
                val_loss = validate(model, val_loader, device, use_amp=args.use_amp)
                validation_metrics = {
                    "val_loss": val_loss,
                    "validation_time_sec": time.perf_counter() - validation_start,
                }
                wandb.log(validation_metrics, step=step)

                if val_loss < best_val_loss - config.get("early_stopping_min_delta", 0.0):
                    best_val_loss = val_loss
                    stale_validations = 0
                    state = checkpoint_state(
                        model,
                        optimizer,
                        scheduler,
                        scaler,
                        config,
                        args,
                        best_val_loss,
                        epoch,
                        step,
                    )
                    save_checkpoint(state, args.output_dir, "model_bestval.pt")
                    validation_metrics["best_val_loss"] = best_val_loss
                    validation_metrics["new_best"] = True
                else:
                    stale_validations += 1
                    validation_metrics["best_val_loss"] = best_val_loss
                    validation_metrics["new_best"] = False

                validation_metrics["stale_validations"] = stale_validations
                audit_logger.log_event(
                    "validation",
                    step=step,
                    epoch=epoch,
                    **validation_metrics,
                )
                model.train()
                progress.set_postfix(
                    train_loss=f"{loss.item():.4f}",
                    val_loss=f"{val_loss:.4f}",
                )

                patience = config.get("early_stopping_patience", 0)
                if patience and stale_validations >= patience:
                    early_stopped = True
                    wandb.log({"early_stopped": 1}, step=step)
                    audit_logger.log_event(
                        "early_stopping_triggered",
                        step=step,
                        epoch=epoch,
                        patience=patience,
                        best_val_loss=best_val_loss,
                    )
                    break
            else:
                progress.set_postfix(
                    train_loss=f"{loss.item():.4f}",
                    learning_rate=f"{optimizer.param_groups[0]['lr']:.2e}",
                )

            if config["gen_every"] and step % config["gen_every"] == 0:
                run_generation(
                    model,
                    audio_tokenizer,
                    text_tokenizer,
                    watermarker,
                    args,
                    device,
                    step,
                    audit_logger,
                )
                model.train()

            progress.update(1)
            step += 1

        if early_stopped:
            break

    progress.close()

    final_state = checkpoint_state(
        model,
        optimizer,
        scheduler,
        scaler,
        config,
        args,
        best_val_loss,
        epoch,
        step,
    )
    save_checkpoint(final_state, args.output_dir, "model_final.pt")
    audit_logger.log_event(
        "training_finished",
        final_step=step,
        final_epoch=epoch,
        best_val_loss=best_val_loss,
        early_stopped=early_stopped,
    )
    audit_logger.update_summary(
        status="completed",
        best_val_loss=best_val_loss,
        final_step=step,
        final_epoch=epoch,
        early_stopped=early_stopped,
        ended_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    )
    return best_val_loss


def main(argv: list[str] | None = None) -> float:
    args = parse_args(argv)
    if os.getenv("WANDB_API_KEY") is None:
        raise ValueError("WANDB_API_KEY is not set")

    config = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    audit_logger = AuditLogger(
        run_dir=args.output_dir,
        manifest=build_run_manifest(args, config, device),
    )

    wandb.init(
        project=args.wandb_project,
        name=args.wandb_name or args.output_dir.name,
        config={**config, **vars(args)},
        dir=args.output_dir / "wandb",
    )
    audit_logger.log_event(
        "wandb_initialized",
        project=args.wandb_project,
        run_name=wandb.run.name if wandb.run else None,
        run_id=wandb.run.id if wandb.run else None,
    )

    try:
        best_val_loss = train_loop(args, config, device, audit_logger)
        wandb.log({"best_val_loss": best_val_loss})
        return best_val_loss
    except Exception as exc:
        audit_logger.log_event("training_error", error=str(exc))
        audit_logger.update_summary(
            status="failed",
            ended_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        )
        raise
    finally:
        wandb.finish()


if __name__ == "__main__":
    main()
