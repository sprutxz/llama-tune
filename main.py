import os
import random
import torch
import torch.distributed as dist
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
import wandb

from config import TrainConfig, FSDPConfig
from utils import log_gpu_memory
from data import get_dataloaders
from model import get_model_and_processor, prepare_model_for_training
from train import train

def main():
    # Initialize configurations
    train_config = TrainConfig()
    fsdp_config = FSDPConfig()
    
    # Set seeds
    torch.manual_seed(train_config.seed)
    random.seed(train_config.seed)
    
    local_rank = 0
    rank = 0
    world_size = 1
    
    # Initialize distributed training if enabled
    if train_config.enable_fsdp:
        dist.init_process_group("nccl")
        local_rank = int(os.environ["LOCAL_RANK"])
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])

    # Initialize wandb on rank 0 only
    if rank == 0:
        wandb.init(
            project="llama-tune",
            config = {
                "model": train_config.model_name,
                "epochs": train_config.num_epochs,
                "lr": train_config.lr,
                "seed": train_config.seed,
                "world_size": world_size if train_config.enable_fsdp else 1,
            }
        )

    # Set CUDA device for distributed training
    if dist.is_initialized():
        torch.cuda.set_device(local_rank)
        torch.cuda.empty_cache()
        os.environ["TORCH_SHOW_CPP_STACKTRACES"] = str(1)
        os.environ["NCCL_ASYNC_ERROR_HANDLING"] = str(1)
    
    # Track GPU memory
    log_gpu_memory("Before model initialization", reset_peak=True, rank=rank)
    
    # Initialize model and processor
    model, processor = get_model_and_processor(train_config=train_config)
    
    log_gpu_memory("After model initialization", rank=rank)
    
    # Set up FSDP for model
    device_id = torch.cuda.current_device()
    log_gpu_memory("Before FSDP wrapping", rank=rank)
    
    model = prepare_model_for_training(model, train_config, fsdp_config, device_id)
    
    log_gpu_memory("After FSDP wrapping", rank=rank)
    
    # Create dataloaders
    train_dataloader, eval_dataloader = get_dataloaders(
        train_config=train_config,
        processor=processor
    )
    
    log_gpu_memory("After dataloader creation", rank=rank)
    
    # Set up optimizer and scheduler
    optimizer = optim.AdamW(
        model.parameters(),
        lr=train_config.lr,
        weight_decay=train_config.weight_decay,
    )
    scheduler = StepLR(optimizer, step_size=1, gamma=train_config.gamma)
    
    log_gpu_memory("After optimizer creation", rank=rank)
    
    # Train the model
    try:
        results = train(
            model,
            train_dataloader,
            eval_dataloader,
            processor.tokenizer,
            optimizer,
            scheduler,
            train_config.gradient_accumulation_steps,
            train_config,
            fsdp_config if train_config.enable_fsdp else None,
            local_rank if train_config.enable_fsdp else None,
            rank if train_config.enable_fsdp else None,
        )
    finally:
        log_gpu_memory("End of training", rank=rank)
        if train_config.enable_fsdp:
            dist.destroy_process_group()
    
    # Finish wandb on rank 0
    if rank == 0:
        wandb.finish()

if __name__ == "__main__":
    main()