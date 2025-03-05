import os
import time
from pathlib import Path
from tqdm import tqdm

import torch
import torch.distributed as dist
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torch.distributed.fsdp.fully_sharded_data_parallel import StateDictType

import wandb

from utils import (
    log_gpu_memory,
    save_fsdp_model_checkpoint_full,
    save_optimizer_checkpoint,
    save_model_and_optimizer_sharded
)

def train(model, train_dataloader, eval_dataloader, tokenizer, optimizer, lr_scheduler, 
          gradient_accumulation_steps, train_config, fsdp_config=None, local_rank=None, rank=None, wandb_run=None):
    world_size = int(os.environ["WORLD_SIZE"])
    
    train_prep = []
    train_loss = []
    
    if not os.path.exists(train_config.output_dir):
        os.makedirs(train_config.output_dir, exist_ok=True)
    
    train_step_perplexity = []
    train_step_loss = []
    
    epoch_times = []
    checkpoint_times = []
    results = {}
    
    # Log initial memory state
    log_gpu_memory("Training start", reset_peak=True, rank=rank)
    
    for epoch in range(train_config.num_epochs):
        print(f"Starting epoch {epoch}/{train_config.num_epochs}")
        epoch_start_time = time.perf_counter()
        
        model.train()
        total_loss = 0.0
        total_length = len(train_dataloader)//gradient_accumulation_steps
        pbar = tqdm(colour="blue", desc=f"Training Epoch: {epoch+1}", total=total_length, dynamic_ncols=True)
        
        log_gpu_memory(f"Epoch {epoch+1} start", rank=rank)
        
        for step, batch in enumerate(train_dataloader):
            # Memory tracking before batch processing
            if step % 10 == 0:  # Log every 10 steps to avoid excessive output
                log_gpu_memory(f"Epoch {epoch+1}, Step {step} before batch", rank=rank)
            
            for key in batch.keys():
                batch[key] = batch[key].to(local_rank)
            
            # Memory after moving batch to GPU
            if step % 10 == 0:
                log_gpu_memory(f"Epoch {epoch+1}, Step {step} after batch to GPU", rank=rank)
            
            loss = model(**batch).loss
            loss = loss/gradient_accumulation_steps
            
            train_step_loss.append(loss.detach().float().item())
            train_step_perplexity.append(float(torch.exp(loss.detach().float())))
            
            total_loss += loss.detach().float()
            
            # Memory before backward pass
            if step % 10 == 0:
                log_gpu_memory(f"Epoch {epoch+1}, Step {step} before backward", rank=rank)
            
            loss.backward()
            
            # Memory after backward pass
            if step % 10 == 0:
                log_gpu_memory(f"Epoch {epoch+1}, Step {step} after backward", rank=rank)
            
            if (step+1) % gradient_accumulation_steps == 0 or step == len(train_dataloader)+1:
                # Memory before optimizer step
                if step % 10 == 0:
                    log_gpu_memory(f"Epoch {epoch+1}, Step {step} before optimizer step", rank=rank)
                
                optimizer.step()
                optimizer.zero_grad()
                
                # Memory after optimizer step
                if step % 10 == 0:
                    log_gpu_memory(f"Epoch {epoch+1}, Step {step} after optimizer step", rank=rank)
                
                pbar.update(1)
                
            pbar.set_description(f"Training Epoch: {epoch+1}/{train_config.num_epochs}, step {step}/{len(train_dataloader)} completed (loss: {loss.detach().float()})")
        
        pbar.close()
        log_gpu_memory(f"End of epoch {epoch+1}", rank=rank)
        
        epoch_end_time = time.perf_counter()-epoch_start_time
        epoch_times.append(epoch_end_time)
        
        if torch.cuda.device_count() > 1:
            dist.all_reduce(total_loss, op=dist.ReduceOp.SUM)
        
        train_epoch_loss = total_loss / len(train_dataloader)
        train_epoch_loss = train_epoch_loss/world_size
        train_perplexity = torch.exp(train_epoch_loss)
        train_prep.append(float(train_perplexity))
        train_loss.append(float(train_epoch_loss))
        
        # Only log to wandb from rank 0
        if rank == 0:
            wandb.log({
                "train_perplexity": train_perplexity, 
                "train_epoch_loss": train_epoch_loss,
                "epoch": epoch
            })
        
        lr_scheduler.step()
            
    dist.barrier()
    
    log_gpu_memory("Before saving checkpoint", rank=rank)
    
    if fsdp_config.checkpoint_type == StateDictType.FULL_STATE_DICT:
        save_fsdp_model_checkpoint_full(
                    model, optimizer, rank, train_config, epoch=epoch
                )
        if train_config.save_optimizer:
            save_optimizer_checkpoint(
                        model, optimizer, rank, train_config, epoch=epoch
                    )
    elif fsdp_config.checkpoint_type == StateDictType.SHARDED_STATE_DICT:
        if train_config.save_optimizer:
            print(" Saving the FSDP model checkpoints using SHARDED_STATE_DICT")
            print("=====================================================")
            save_model_and_optimizer_sharded(model, rank, train_config, optim=optimizer)
        else:
            print(" Saving the FSDP model checkpoints and optimizer using SHARDED_STATE_DICT")
            print("=====================================================")
            save_model_and_optimizer_sharded(model, rank, train_config)
    
    log_gpu_memory("After saving checkpoint", rank=rank)
    
    dist.barrier()
    
    if train_config.enable_fsdp:
        if rank==0:
            print(f"Epoch {epoch+1}: train_perplexity={train_perplexity:.4f}, train_epoch_loss={train_epoch_loss:.4f}, epoch time {epoch_end_time}s")
            
    avg_epoch_time = sum(epoch_times)/ len(epoch_times)
    avg_train_prep = sum(train_prep)/len(train_prep)
    avg_train_loss = sum(train_loss)/len(train_loss)
    
    results['avg_train_prep'] = avg_train_prep
    results['avg_train_loss'] = avg_train_loss
    results["avg_epoch_time"] = avg_epoch_time
    print("*"*200)
    print(results)
    
    return results