"""Fine-tunes a PaliGemma model on a given dataset with BF16 mixed precision using custom training loop."""

from datasets import load_dataset
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.optim import AdamW

# from torch.profiler import profile, ProfilerActivity  # Unused in this version
from transformers import (
    PaliGemmaProcessor,
    PaliGemmaForConditionalGeneration,
    get_linear_schedule_with_warmup,
)
from accelerate import Accelerator
from opentelemetry import context as context_api
import os
import logging

# import math  # Unused in this version
import random
import time
from checkpoint import setup_checkpoint_manager
import nvidia_resiliency_ext.fault_tolerance as fault_tolerance

# Import tracing
try:
    from xtorch_wandb import XtorchWandb

    TRACING_ENABLED = True
except ImportError:
    TRACING_ENABLED = False

logging.basicConfig(
    level=logging.INFO,
    force=True,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logging.getLogger("transformers").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)

# Enable TF32 for better performance on modern GPUs (H100/H200/B200)
torch.set_float32_matmul_precision("high")


# Connect to distributed trace
tracer = None
if TRACING_ENABLED:
    try:
        tracer = XtorchWandb()
        logger.info("Connected to distributed trace from XTorch CLI")
    except Exception as e:
        logger.warning(f"Failed to connect to trace: {e}")
        tracer = None
else:
    logger.info("xtorch-wandb not available, proceeding without tracing")


# --- Environment Variables and Configuration ---
num_train_epochs = int(os.getenv("NUM_TRAIN_EPOCHS", "3"))
per_device_train_batch_size = int(os.getenv("PER_DEVICE_TRAIN_BATCH_SIZE", "4"))
gradient_accumulation_steps = int(os.getenv("GRADIENT_ACCUMULATION_STEPS", "4"))
learning_rate = float(os.getenv("LEARNING_RATE", "2e-5"))
warmup_steps = int(os.getenv("WARMUP_STEPS", "2"))
weight_decay = float(os.getenv("WEIGHT_DECAY", "1e-6"))
adam_beta2 = float(os.getenv("ADAM_BETA2", "0.999"))
optim = os.getenv("OPTIM", "adamw_torch")
logging_steps = int(os.getenv("LOGGING_STEPS", "100"))
save_steps = int(os.getenv("SAVE_STEPS", "1000"))
preset_max_steps = int(os.getenv("MAX_STEPS", "1000"))
save_total_limit = int(os.getenv("SAVE_TOTAL_LIMIT", "3"))
dataloader_num_workers = int(os.getenv("DATALOADER_NUM_WORKERS", "1"))
sleep_timeout = (
    int(os.getenv("SLEEP_TIMEOUT", "0")) if os.getenv("SLEEP_TIMEOUT") else None
)
restart_count = int(os.getenv("TORCHELASTIC_RESTART_COUNT", "0"))
dataset_name = os.getenv("DATASET_NAME", "merve/vqav2-small")
data_source = os.getenv("DATA_SOURCE", "lssd")
local_dataset_path = os.getenv("LOCAL_DATASET_PATH", "/ssd/waymo-open-dataset")
gcs_dataset_path = os.getenv("GCS_DATASET_PATH", "/data")
model_id = os.getenv("MODEL_ID", "google/paligemma2-3b-pt-448")
tensorboard_output_dir = os.getenv("EXPLICIT_LOG_DIR", "out_paligemma")
checkpoints_dir = os.getenv("CHECKPOINTS_ROOT_DIR", "/ssd/checkpoints")
profiling_mode = os.getenv("PROFILING_MODE", "false")
profiling_ranks = os.getenv("PROFILING_RANKS", "none")
local_rank = int(os.getenv("LOCAL_RANK", "0"))
job_id = (
    os.getenv("JOB_ID")
    or os.getenv("COMPUTESUBSTRATE_JOBID")
    or "paligemma_training_job"
)


# Track overall training start time
start_time = time.time()


def initialize_rank_monitor_with_retry(
    *,
    max_attempts: int = 10,
    initial_delay_sec: float = 0.5,
    max_delay_sec: float = 5.0,
    backoff_multiplier: float = 1.5,
) -> fault_tolerance.RankMonitorClient:
    """Initialize rank monitor with retry and exponential backoff.

    Creates a fresh RankMonitorClient for each retry attempt to avoid
    assertion failures from partially-initialized connections. Uses
    exponential backoff to handle the rank monitor server startup delay.

    Args:
        max_attempts: Maximum number of connection attempts
        initial_delay_sec: Initial retry delay in seconds
        max_delay_sec: Maximum retry delay (caps exponential growth)
        backoff_multiplier: Factor to multiply delay by after each failure

    Returns:
        Successfully connected RankMonitorClient instance.
    """
    delay_sec = initial_delay_sec

    for attempt_idx in range(1, max_attempts + 1):
        # Create fresh client for each attempt to avoid assertion failures
        client = fault_tolerance.RankMonitorClient()
        try:
            client.init_workload_monitoring()
            if attempt_idx > 1:
                logger.info(
                    "Rank monitor connection succeeded after %s attempts",
                    attempt_idx,
                )
            return client  # Success - return the connected client
        except (
            fault_tolerance.RankMonitorClientError,
            ConnectionRefusedError,
            OSError,
            AssertionError,
        ) as error:
            socket_path = os.getenv("FT_RANK_MONITOR_IPC_SOCKET", "<not set>")
            logger.warning(
                "Rank monitor connection attempt %s/%s failed (socket=%s): %s. Retrying in %.2fs...",
                attempt_idx,
                max_attempts,
                socket_path,
                error,
                delay_sec,
            )
            if attempt_idx == max_attempts:
                logger.error(
                    "Failed to connect to rank monitor after %s attempts. Socket path: %s",
                    max_attempts,
                    socket_path,
                )
                raise
            time.sleep(delay_sec)
            # Exponential backoff with cap
            delay_sec = min(delay_sec * backoff_multiplier, max_delay_sec)

    # Should never reach here
    raise RuntimeError("Retry loop exited unexpectedly")


# NOTE: ft_client initialization moved AFTER DataLoader creation to avoid fork conflicts
# The DataLoader uses multiprocessing (fork), which conflicts with gRPC threads in ft_client
ft_client = None


# --- SETUP PHASE ---
# This is the FIRST span created, so it uses parent_context from job_submission
with tracer.span(
    "training.execution.setup", {"rank": local_rank, "model_id": model_id}
) as training_setup:
    # ========== SPAN RESTRUCTURE START (2025-11-03) ==========
    # Changed: model.initialization now closes early and other spans are siblings
    # Previous: model.initialization wrapped all other setup spans (dataset_loading, dataloader_creation, etc.)
    # Revert instructions: Move all code below this back inside model.initialization span
    with tracer.span("model.initialization", {"rank": local_rank}) as startup_span:
        # --- Initialize Accelerator for BF16 ---
        accelerator = Accelerator(mixed_precision="bf16")

        logger.info(f"Starting with {accelerator.num_processes} processes.")
        logger.info(f"Using {model_id} model.")
        logger.info(
            "Using BF16 mixed precision training with custom loop (MULTI-GPU FIXED). 🎯"
        )

        logger.info(f"Device: {accelerator.device}")
        logger.info(f"Checkpoints will be saved to: {checkpoints_dir}")
        logger.info(f"TensorBoard logs will be saved to: {tensorboard_output_dir}")

        # --- Initialize Checkpoint Manager ---
        checkpoint_manager = setup_checkpoint_manager(
            job_id=job_id, checkpoints_root_dir=checkpoints_dir
        )
        logger.info(f"Checkpoint manager initialized for job: {job_id}")

        processor = PaliGemmaProcessor.from_pretrained(model_id)

    # model.initialization span closes here (early closure)

    # Dataset loading with tracing (now a sibling to model.initialization)
    with tracer.span(
        "training.dataset_loading",
        {"rank": local_rank, "dataset_name": dataset_name, "data_source": data_source},
    ) as dataset_span:
        logger.info(f"Loading {dataset_name} dataset from {data_source}")
        ds = None
        is_waymo_dataset = "waymo" in dataset_name
        if is_waymo_dataset:
            if data_source == "lssd":
                logger.info(f"Loading Waymo dataset from local {local_dataset_path}")
                ds = load_dataset(
                    "parquet",
                    data_dir=local_dataset_path,
                    split="train",
                    streaming=True,
                )
            else:  # GCS
                ds = load_dataset(
                    "parquet", data_dir=gcs_dataset_path, split="train", streaming=True
                )
        else:
            if data_source == "lssd":
                logger.info("Loading dataset from huggingface datasets")
                ds = load_dataset(dataset_name, split="validation")
            else:  # GCS
                ds = load_dataset(
                    "parquet", data_dir=gcs_dataset_path, split="validation"
                )
        logger.info(f"Dataset loaded successfully from {data_source}")

    # Define collate function (needs processor from outer scope)
    def collate_fn(examples):
        texts = ["<image>answer en " + example["question"] for example in examples]
        labels = [example["multiple_choice_answer"] for example in examples]
        images = [example["image"].convert("RGB") for example in examples]

        # MULTI-GPU FIX: Use fixed padding to ensure consistent batch shapes across all processes
        tokens = processor(
            text=texts,
            images=images,
            suffix=labels,
            return_tensors="pt",
            padding="max_length",  # Fixed padding instead of "longest"
            max_length=32,  # Consistent sequence length across all 8 GPU processes
            truncation=True,  # Handle long sequences gracefully
        )
        return tokens

    # Load model
    model = PaliGemmaForConditionalGeneration.from_pretrained(
        model_id, torch_dtype=torch.bfloat16
    )
    logger.info(f"Model loaded with dtype: {model.dtype}")

    # MEMORY FIX: Enable gradient checkpointing for 8 GPU training
    model.gradient_checkpointing_enable()
    logger.info("✅ Gradient checkpointing enabled for memory efficiency")

    # --- Set up Optimizer and Scheduler ---
    optimizer = AdamW(
        model.parameters(),
        lr=learning_rate,
        betas=(0.9, adam_beta2),
        weight_decay=weight_decay,
    )

    num_samples_in_dataset = 999 * 798 if is_waymo_dataset else 21500
    global_batch_size = (
        per_device_train_batch_size
        * gradient_accumulation_steps
        * accelerator.num_processes
    )

    if preset_max_steps < 0:
        target_sample_ratio = float(os.environ.get("TARGET_SAMPLE_RATIO", "0.9"))
        total_samples_available = num_samples_in_dataset * num_train_epochs
        target_total_samples = int(total_samples_available * target_sample_ratio)
        target_total_samples = (
            target_total_samples // global_batch_size
        ) * global_batch_size
        max_steps = target_total_samples // global_batch_size
        logger.info("Equal work calculation:")
        logger.info(f"  - Global batch size: {global_batch_size}")
        logger.info(f"  - Target sample ratio: {target_sample_ratio}")
        logger.info(
            f"  - Target samples: {target_total_samples} ({target_sample_ratio * 100:.1f}% of {total_samples_available})"
        )
        logger.info(f"  - Calculated max_steps: {max_steps}")
    else:
        max_steps = preset_max_steps

    lr_scheduler = get_linear_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=max_steps,
    )

    # DataLoader creation with tracing (now a sibling to model.initialization)
    with tracer.span(
        "training.dataloader_creation",
        {"rank": local_rank, "num_workers": dataloader_num_workers},
    ) as dataloader_span:
        # MULTI-GPU FIX: Add drop_last=True to ensure consistent batch sizes
        use_persistent_workers = (
            os.getenv("DATALOADER_PERSISTENT_WORKERS", "false").lower() == "true"
        )

        train_dataloader = DataLoader(
            ds,
            batch_size=per_device_train_batch_size,
            collate_fn=collate_fn,
            num_workers=dataloader_num_workers,
            persistent_workers=use_persistent_workers,
            drop_last=True,  # Drop incomplete batches to ensure consistent sizes across GPUs
        )

        model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
            model, optimizer, train_dataloader, lr_scheduler
        )

        # CRITICAL FIX: Synchronize CUDA/NCCL before DataLoader workers fork
        # Without this, workers inherit stale CUDA state causing "Invalid peer GPU memory access" errors
        if accelerator.num_processes > 1:
            logger.info(
                "Synchronizing CUDA/NCCL across all ranks before DataLoader fork..."
            )
            torch.cuda.synchronize()  # Wait for all CUDA operations to complete
            dist.barrier()  # NCCL barrier to ensure all ranks are ready
            logger.info("CUDA/NCCL synchronization complete")

            # Warmup NCCL with a dummy all-reduce to establish peer-to-peer connections
            logger.info("Warming up NCCL peer-to-peer connections...")
            dummy_tensor = torch.zeros(1, device=accelerator.device)
            dist.all_reduce(dummy_tensor, op=dist.ReduceOp.SUM)
            torch.cuda.synchronize()
            logger.info("NCCL warmup complete - peer GPU connections established")

    # Load Checkpoint with auto strategy selection
    with tracer.span(
        "training.checkpoint_restore", {"rank": accelerator.process_index}
    ) as restore_span:
        # Initialize torch.distributed for checkpoint loading if needed
        if accelerator.num_processes > 1 and not dist.is_initialized():
            dist.init_process_group(backend="nccl")

        # Use auto checkpoint loading based on SAVE_RANKS environment variable
        resume_step, current_epoch, lr_scheduler_state = (
            checkpoint_manager.load_checkpoint_auto(
                rank=accelerator.process_index,
                world_size=accelerator.num_processes,
                model=model,
                optimizer=optimizer,
            )
        )

        # Restore scheduler state if available
        if lr_scheduler_state:
            lr_scheduler.load_state_dict(lr_scheduler_state)
            logger.info(f"[Rank {accelerator.process_index}] Restored scheduler state")

        logger.info(f"Starting training from step {resume_step}, epoch {current_epoch}")
    # checkpoint_restore span closes here
    # ========== SPAN RESTRUCTURE END (2025-11-03) ==========

    # --- Synchronize GPU State After Checkpoint Restore ---
    # CRITICAL: After loading checkpoint, ensure all ranks have consistent GPU/NCCL state
    if accelerator.num_processes > 1:
        logger.info("Synchronizing CUDA/NCCL after checkpoint restore...")
        torch.cuda.synchronize()  # Wait for checkpoint load operations
        torch.cuda.empty_cache()  # Clear fragmented GPU memory from checkpoint load
        dist.barrier()  # Ensure all ranks completed restore before proceeding
        logger.info("Post-checkpoint GPU synchronization complete")

    # --- Force DataLoader Workers to Fork NOW ---
    # CRITICAL FIX: DataLoader workers fork lazily on first iteration, not at construction
    # We must force them to fork BEFORE initializing FT client (which creates gRPC threads)
    if dataloader_num_workers > 0:
        # Final CUDA sync to ensure NCCL state is stable before worker fork
        if accelerator.num_processes > 1:
            torch.cuda.synchronize()
        logger.info(
            f"Pre-starting DataLoader workers (num_workers={dataloader_num_workers}) after NCCL warmup..."
        )
        try:
            # Force worker processes to fork by creating iterator and fetching first batch
            # This triggers the actual fork() calls that create worker subprocesses
            temp_iter = iter(train_dataloader)
            _ = next(temp_iter)  # Fetch one batch to ensure workers are fully started
            del temp_iter  # Clean up the temporary iterator
            logger.info("✅ DataLoader workers successfully pre-started and forked")
        except StopIteration:
            logger.warning("DataLoader is empty, workers not started")
        except Exception as e:
            logger.error(f"Failed to pre-start DataLoader workers: {e}")
            raise
    else:
        logger.info("DataLoader workers disabled (num_workers=0), no pre-start needed")

    # --- Initialize Fault Tolerance Client AFTER Workers Fork ---
    # NOW it's safe: worker processes already forked, gRPC threads will only be in parent
    logger.info("Initializing fault tolerance client (after DataLoader workers forked)")
    ft_client = initialize_rank_monitor_with_retry()
    logger.info("Fault tolerance client initialized successfully")

    # CRITICAL: Capture context BEFORE training_setup closes
    # This preserves the trace context so step spans remain in the same trace
    stored_context = context_api.get_current()

    # Setup span closes here
# training_setup span closes here, but we've captured its context

# --- TRAINING LOOP PHASE ---
completed_steps = resume_step
last_log_time = time.time()
total_loss = 0
cumulative_training_time = 0.0  # Track training time excluding checkpoints

# Attach the stored context so training.step_* spans remain in the same trace
# They will be siblings to training_setup (both children of launcher.process_started)
token = context_api.attach(stored_context)
try:
    model.train()

    # Start first training interval section
    ft_client.start_section("step")

    for step, batch in enumerate(train_dataloader):
        if completed_steps >= max_steps:
            break

        # Create training step span to wrap the actual training work
        with tracer.span(
            f"training.step_{completed_steps:08d}",
            {"rank": accelerator.process_index, "step": completed_steps},
        ):
            # FP8 mixed precision is managed within this context manager
            with accelerator.accumulate(model):
                with accelerator.autocast():
                    outputs = model(**batch)
                    loss = outputs.loss

                # This will scale the loss and handle gradient accumulation
                accelerator.backward(loss)

                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), 1.0)

                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

        # End training interval section before logging
        if completed_steps % logging_steps == 0 and completed_steps > 0:
            # Optional sleep for testing - only inject on FIRST run (restart_count == 0)
            # This tests FT restart mechanism once, then training proceeds normally after restart
            if sleep_timeout is not None and sleep_timeout > 5 and restart_count == 0:
                sleep_time = random.uniform(5, sleep_timeout)
                if accelerator.is_main_process:
                    logger.info(
                        f"[FIRST RUN] Sleeping for {sleep_time:.2f} seconds to trigger FT restart (SLEEP_TIMEOUT={sleep_timeout}, restart_count={restart_count})"
                    )
                time.sleep(sleep_time)

            ft_client.end_section("step")

        # Log and save
        if completed_steps % logging_steps == 0 and completed_steps > 0:
            current_time = time.time()
            total_elapsed_time = current_time - start_time
            interval_elapsed_time = current_time - last_log_time

            # Accumulate only training time (excludes checkpoint time)
            cumulative_training_time += interval_elapsed_time

            # --- FIXED FLOPs Calculation ---
            unwrapped_model = accelerator.unwrap_model(model)
            num_parameters = sum(p.numel() for p in unwrapped_model.parameters())

            # Calculate current interval performance metrics
            steps_in_interval = logging_steps
            samples_in_interval = global_batch_size * steps_in_interval

            # Current throughput based on recent interval
            current_samples_per_second = (
                samples_in_interval / interval_elapsed_time
                if interval_elapsed_time > 0
                else 0
            )

            # Current TFLOPS based on recent interval
            current_tflops_per_device_per_sec = (
                (2 * num_parameters * samples_in_interval)
                / (interval_elapsed_time)
                / 1e12
                if interval_elapsed_time > 0
                else 0
            )

            # Calculate cumulative averages using training time only (excludes checkpoints)
            avg_samples_per_second = (
                global_batch_size * completed_steps / cumulative_training_time
                if cumulative_training_time > 0
                else 0
            )

            logger.info(
                f"Step {completed_steps}: "
                f"Loss = {loss.detach().item():.4f}, "
                f"Total Runtime = {total_elapsed_time:.2f}s, "
                f"Training Time = {cumulative_training_time:.2f}s, "
                f"Interval = {interval_elapsed_time:.2f}s, "
                f"Current Samples/s = {current_samples_per_second:.2f}, "
                f"Avg Samples/s = {avg_samples_per_second:.2f}, "
                f"Current TFLOPS/device/s = {current_tflops_per_device_per_sec:.2f}"
            )

        # Start next training interval section after logging
        # BUT skip if we're about to checkpoint (checkpoint happens outside step section)
        if completed_steps % logging_steps == 0 and completed_steps > 0:
            if not (completed_steps % save_steps == 0):
                ft_client.start_section("step")

        if completed_steps % save_steps == 0 and completed_steps > 0:
            accelerator.wait_for_everyone()

            # Start checkpoint section (300s timeout) - only for save operation
            ft_client.start_section("checkpoint")

            # Checkpoint save with tracing
            with tracer.span(
                "training.checkpoint_save",
                {"rank": accelerator.process_index, "step": completed_steps},
            ) as save_span:
                # Save checkpoint using CheckpointManager with auto strategy selection
                unwrapped_model = accelerator.unwrap_model(model)
                checkpoint_path = checkpoint_manager.save_checkpoint_auto(
                    rank=accelerator.process_index,
                    model=unwrapped_model,
                    optimizer=optimizer,
                    epoch=current_epoch,
                    global_step=completed_steps,
                    world_size=accelerator.num_processes,
                    lr_scheduler_state=lr_scheduler.state_dict(),
                    loss=loss.detach().item(),
                )
            # save_span auto-closes here

            # End checkpoint section - before cleanup
            ft_client.end_section("checkpoint")

            # Cleanup old checkpoints (uses out-of-section timeout: 120s)
            if accelerator.is_main_process:
                # Track checkpoint cleanup time
                with tracer.span(
                    "training.checkpoint_cleanup",
                    {"rank": accelerator.process_index, "step": completed_steps},
                ) as cleanup_span:
                    deleted = checkpoint_manager.cleanup_old_checkpoints(
                        rank=accelerator.process_index,
                        keep_last_n=save_total_limit,
                        keep_every_n_steps=None,
                    )
                # cleanup_span auto-closes here

                if deleted:
                    logger.info(f"Cleaned up {len(deleted)} old checkpoints")

            # Wait for all ranks to complete checkpoint operations before starting new section
            # This prevents fast ranks from timing out while waiting for slow ranks
            accelerator.wait_for_everyone()

            # Start new 'step' section after checkpoint completes
            ft_client.start_section("step")

        # Update last_log_time AFTER checkpoint operations complete (if logging happened)
        # This ensures checkpoint save time is excluded from interval calculations
        if completed_steps % logging_steps == 0 and completed_steps > 0:
            last_log_time = time.time()  # Fresh timestamp excludes checkpoint time

        # Send heartbeat AFTER all operations (training + logging + checkpoint)
        # This ensures we send heartbeat even after long checkpoint saves
        ft_client.send_heartbeat()

        completed_steps += 1
finally:
    # Close any open sections before exiting (avoids warnings at training end)
    ft_client.end_all_sections()

    # Detach the context to restore the original context
    context_api.detach(token)

accelerator.wait_for_everyone()
accelerator.end_training()
ft_client.shutdown_workload_monitoring()
