"""Fine-tunes vision-language models using HuggingFace transformers.

Supports any vision-to-sequence model compatible with AutoModelForVision2Seq:
- LLaVA models (default: llava-hf/llava-1.5-7b-hf)
- PaliGemma (google/paligemma2-3b-pt-448)
- Other vision-language models

Usage:
    # Single GPU training
    python paligemma.py
    
    # Multi-GPU training with DistributedDataParallel (DDP)
    torchrun --nproc_per_node=2 paligemma.py
    
    # Multi-node training
    torchrun --nproc_per_node=8 --nnodes=2 --master_addr=192.168.1.1 paligemma.py
    
    # Custom model
    MODEL_ID=google/paligemma2-3b-pt-448 python paligemma.py
"""

from datasets import load_dataset
import torch
from torch.profiler import profile, ProfilerActivity
from transformers import (
    AutoProcessor,
    AutoModelForVision2Seq,
    Trainer,
    TrainingArguments,
)
import os

import logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

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
save_total_limit = int(os.getenv("SAVE_TOTAL_LIMIT", "1"))
dataloader_num_workers = int(os.getenv("DATALOADER_NUM_WORKERS", "1"))
dataset_name = os.getenv("DATASET_NAME", "merve/vqav2-small")
model_id = os.getenv("MODEL_ID", "llava-hf/llava-1.5-7b-hf")
tensorboard_output_dir = os.getenv("EXPLICIT_LOG_DIR", "out_vision_language")
checkpoints_dir = os.getenv("CHECKPOINTS_ROOT_DIR", "./checkpoints")
profiling_mode = os.getenv("PROFILING_MODE", "false")
# Device detection and distributed training setup
cuda_available = torch.cuda.is_available()
world_size = int(os.environ.get("WORLD_SIZE", "1"))
local_rank = int(os.environ.get("LOCAL_RANK", "0"))

if cuda_available:
    gpu_count = torch.cuda.device_count()
    logger.info(f"CUDA available: {gpu_count} GPU(s) detected")
    for i in range(gpu_count):
        gpu_name = torch.cuda.get_device_name(i)
        logger.info(f"  GPU {i}: {gpu_name}")
else:
    logger.info("CUDA not available - training will use CPU (very slow)")

# Distributed training detection
if world_size > 1:
    logger.info(f"DistributedDataParallel (DDP) detected: {world_size} processes, local_rank={local_rank}")
    logger.info("HuggingFace Trainer will automatically use DDP for multi-GPU/multi-node training")
else:
    logger.info("Single process training (no DDP)")

logger.info(f"Loading model: {model_id}")
processor = AutoProcessor.from_pretrained(model_id)
logger.info(f"Loading dataset: {dataset_name}")
ds = load_dataset(dataset_name, split="validation")
logger.info("Dataset loaded successfully")


def collate_fn(examples):
    """Generic collate function that works with different vision-language models."""
    # Format prompts - this works for PaliGemma and similar models
    texts = ["<image>answer en " + example["question"] for example in examples]
    labels = [example["multiple_choice_answer"] for example in examples]
    images = [example["image"].convert("RGB") for example in examples]
    
    # Use the processor to handle model-specific tokenization and image processing
    # AutoProcessor automatically adapts to the specific model's requirements
    try:
        # Try with suffix parameter (PaliGemma style)
        tokens = processor(text=texts, images=images, suffix=labels, return_tensors="pt", padding="longest")
    except TypeError:
        # Fallback for models that don't support suffix parameter
        # Concatenate text and labels manually
        full_texts = [text + " " + label for text, label in zip(texts, labels)]
        tokens = processor(text=full_texts, images=images, return_tensors="pt", padding="longest")
    
    return tokens


model = AutoModelForVision2Seq.from_pretrained(model_id, torch_dtype=torch.bfloat16)
logger.info(f"Model loaded: {model.__class__.__name__}")
logger.info(f"Model type: {model.config.model_type if hasattr(model.config, 'model_type') else 'Unknown'}")

# Log where the model is actually located
model_device = next(model.parameters()).device
logger.info(f"Model device: {model_device}")
if cuda_available and str(model_device) == "cpu":
    logger.warning("Model is on CPU despite CUDA being available - Trainer will move it to GPU during training")

num_samples_in_dataset = 21500
global_batch_size = per_device_train_batch_size * gradient_accumulation_steps * int(os.environ.get("WORLD_SIZE", "1"))

# Calculate max_steps ensuring equal work across different GPU types
if preset_max_steps < 0:
    # Get target sample ratio from environment (default 90% of dataset)
    target_sample_ratio = float(os.environ.get("TARGET_SAMPLE_RATIO", "0.9"))
    total_samples_available = num_samples_in_dataset * num_train_epochs
    target_total_samples = int(total_samples_available * target_sample_ratio)

    # Round down to nearest multiple of global_batch_size for exact division
    target_total_samples = (target_total_samples // global_batch_size) * global_batch_size
    max_steps = target_total_samples // global_batch_size

    logger.info(f"Calculated max_steps: {max_steps} (global batch size: {global_batch_size})")
else:
    max_steps = preset_max_steps

# Training Arguments from parsed arguments
training_args = TrainingArguments(
    num_train_epochs=num_train_epochs,
    remove_unused_columns=False,
    per_device_train_batch_size=per_device_train_batch_size,
    gradient_accumulation_steps=gradient_accumulation_steps,
    warmup_steps=warmup_steps,
    learning_rate=learning_rate,
    weight_decay=weight_decay,
    adam_beta2=adam_beta2,
    logging_steps=logging_steps,
    optim=optim,
    save_strategy="steps",
    save_steps=save_steps,
    save_total_limit=save_total_limit,
    output_dir=checkpoints_dir,  # Checkpoints go to CHECKPOINTS_ROOT_DIR
    logging_dir=tensorboard_output_dir,  # TensorBoard logs go to EXPLICIT_LOG_DIR
    bf16=True,
    gradient_checkpointing=True,  # enable reduce memory consumption
    report_to=["tensorboard"] if int(os.environ.get("LOCAL_RANK", "0")) == 0 else [],
    dataloader_pin_memory=True,
    max_steps=max_steps,
    dispatch_batches=True,
    dataloader_num_workers=dataloader_num_workers,
    dataloader_persistent_workers=True,
)

trainer = Trainer(
    model=model,
    train_dataset=ds,
    data_collator=collate_fn,
    args=training_args,
)
logger.info("Starting training...")

# --- Start Training ---
if profiling_mode == "true":
    prof_log_path = os.path.join(tensorboard_output_dir, "trace.json")
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        train_result = trainer.train()
    try:
        prof.export_chrome_trace(prof_log_path)
        logger.info(f"Profiling trace saved to: {prof_log_path}")
    except Exception as e:
        logger.warning(f"Failed to export profiling trace: {e}")
else:
    train_result = trainer.train()

logger.info("Training completed")

# After training is complete, access the state
total_flops_calculated = trainer.state.total_flos
train_runtime = train_result.metrics["train_runtime"]
num_devices = int(os.environ.get("WORLD_SIZE", "8"))

if local_rank == 0 and total_flops_calculated and total_flops_calculated > 0:
    tflops_per_device_per_sec = (total_flops_calculated / 1e12 / train_runtime) / num_devices
    logger.info(f"Performance: {tflops_per_device_per_sec:.4f} TFLOPS/device/second")
