import copy
import functools
from functools import partial
import os
from pathlib import Path
import time

import torch
import torch.distributed as dist
import torch.distributed._shard.checkpoint as dist_cp
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    CheckpointImpl,
    apply_activation_checkpointing,
    checkpoint_wrapper,
)
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from transformers.models.llama.modeling_llama import LlamaDecoderLayer
from transformers.models.mllama.modeling_mllama import (
    MllamaCrossAttentionDecoderLayer,
    MllamaSelfAttentionDecoderLayer,
    MllamaVisionEncoderLayer,
)

# GPU memory tracking functions
def get_gpu_memory_info():
    """Return detailed GPU memory usage information in a formatted string"""
    if not torch.cuda.is_available():
        return "CUDA not available"
    
    device = torch.cuda.current_device()
    allocated = torch.cuda.memory_allocated(device) / (1024 ** 3)  # GB
    max_allocated = torch.cuda.max_memory_allocated(device) / (1024 ** 3)  # GB
    reserved = torch.cuda.memory_reserved(device) / (1024 ** 3)  # GB
    max_reserved = torch.cuda.max_memory_reserved(device) / (1024 ** 3)  # GB
    
    return (f"GPU {device}: "
            f"Allocated: {allocated:.2f}GB (Max: {max_allocated:.2f}GB), "
            f"Reserved: {reserved:.2f}GB (Max: {max_reserved:.2f}GB)")

def log_gpu_memory(message="", reset_peak=False, rank=0):
    """Log GPU memory usage with an optional message"""
    if not torch.cuda.is_available():
        return
    
    if rank == 0:  # Only log on main process
        memory_info = get_gpu_memory_info()
        print(f"[MEMORY] {message} - {memory_info}")
    
    if reset_peak:
        torch.cuda.reset_peak_memory_stats()

def resize_image(img, max_dimension = 1120):
    original_width, original_height = img.size

    if original_width > original_height:
        scaling_factor = max_dimension / original_width
    else:
        scaling_factor = max_dimension / original_height

    new_width = int(original_width * scaling_factor)
    new_height = int(original_height * scaling_factor)

    # Resize the image while maintaining the aspect ratio
    resized_img = img.resize((new_width, new_height))
    return resized_img

# Functions for token manipulation
def check_header(targets, seq):
    """Check system prompt token seq or user prompt token seq is in the current token list"""
    for i in range(len(seq) - 3):
        if seq[i : i + 3] in targets:
            return True
    return False

def replace_target(target, seq):
    for i in range(len(seq) - 3):
        if seq[i : i + 3] == target:
            seq[i], seq[i + 1], seq[i + 2] = -100, -100, -100
    return seq

# FSDP utility functions
def get_polices():
    from torch.distributed.fsdp import MixedPrecision
    
    mixed_precision_policy = MixedPrecision(
        param_dtype=torch.bfloat16,
        # Gradient communication precision.
        reduce_dtype=torch.bfloat16,
        # Buffer precision.
        buffer_dtype=torch.bfloat16,
        cast_forward_inputs=True,
    )

    wrapping_policy = functools.partial(
        transformer_auto_wrap_policy,
        transformer_layer_cls=set([LlamaDecoderLayer, MllamaSelfAttentionDecoderLayer,MllamaVisionEncoderLayer,MllamaCrossAttentionDecoderLayer])
    )
    
    return mixed_precision_policy, wrapping_policy

def apply_fsdp_checkpointing(model):
    print("--> applying fsdp activation checkpointing...")
    
    non_reentrant_wrapper = partial(
        checkpoint_wrapper,
        checkpoint_impl=CheckpointImpl.NO_REENTRANT,
    )

    check_fn = lambda submodule: isinstance(submodule, LlamaDecoderLayer)
    apply_activation_checkpointing(
        model, checkpoint_wrapper_fn=non_reentrant_wrapper, check_fn=check_fn
    )

# Checkpoint saving functions
def save_fsdp_model_checkpoint_full(model, optimizer, rank, cfg, epoch=1):
    """saving model via rank0 cpu streaming and full_state_dict
    This means there is an upper limit of model sizes that can be saved based on CPU memory. For larger model use local state dict
    """
    from torch.distributed.fsdp import FullStateDictConfig
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
    from torch.distributed.fsdp.fully_sharded_data_parallel import StateDictType

    with FSDP.state_dict_type(
        model, StateDictType.FULL_STATE_DICT, FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
    ):
        cpu_state = model.state_dict()
        print(f"saving process: rank {rank}  done w model state_dict\n")

    if rank == 0:
        print("--> saving model ...")
        # create save path
        folder_name = (
            cfg.dist_checkpoint_root_folder
            + "/"
            + cfg.dist_checkpoint_folder
            + "-"
            + cfg.model_name
        )
        save_dir = Path.cwd() / folder_name
        save_dir.mkdir(parents=True, exist_ok=True)
        save_name = cfg.model_name.replace("/","--") + "-" + str(epoch) + ".pt"
        save_full_path = str(save_dir) + "/" + save_name
        # save model
        torch.save(cpu_state, save_full_path)
        print(f"model checkpoint saved for epoch {epoch} at {save_full_path}\n")

def save_optimizer_checkpoint(model, optimizer, rank, cfg, epoch=1):
    """save optimizer state via full state dict"""
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
    
    print(f"--> optim state call on rank {rank}\n")
    # pull all sharded optimizer states to rank0 cpu...
    optim_state = FSDP.full_optim_state_dict(model, optimizer)
    print(f"optim state dict ready on {rank} and len of {len(optim_state)}\n")

    if rank == 0:
        folder_name = (
            cfg.dist_checkpoint_root_folder
            + "/"
            + cfg.dist_checkpoint_folder
            + "-"
            + cfg.model_name
            )
        
        save_dir = Path.cwd() / folder_name
        save_dir.mkdir(parents=True, exist_ok=True)

        opt_save_name = (
            "optimizer" + "-" + cfg.model_name + "-" + str(epoch) + ".pt"
        )
        opt_save_full_path = save_dir / opt_save_name
        print("--> saving optimizer state...")
        torch.save(optim_state, opt_save_full_path)
        print(f"--> saved {opt_save_full_path} to disk")

def save_model_and_optimizer_sharded(model, rank, cfg, optim=None):
    """save model and optimizer via sharded_state_dict to save_dir"""
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
    from torch.distributed.fsdp.fully_sharded_data_parallel import StateDictType
    
    folder_name = (
        cfg.dist_checkpoint_root_folder
        + "/"
        + cfg.dist_checkpoint_folder
        + "-"
        + cfg.model_name
    )

    save_dir = Path.cwd() / folder_name
    if rank == 0:
        print(f"Saving model to {save_dir}")

    distributed_writer = dist_cp.FileSystemWriter(
        save_dir
    )
    t0 = time.perf_counter()

    with FSDP.state_dict_type(model, StateDictType.SHARDED_STATE_DICT):
        state_dict = model.state_dict()
        
        if optim is not None:
            state_dict1 = {
                "model": state_dict
            }
            state_dict1["optim"] = FSDP.optim_state_dict(model, optim)
            
            state_dict = state_dict1

    dist_cp.save_state_dict(
        state_dict=state_dict,
        storage_writer=distributed_writer,
        planner=dist_cp.DefaultSavePlanner(),
    )
    
    dist.barrier()
    t1 = time.perf_counter()
    if rank == 0:
        print(f"Sharded state checkpoint saved to {save_dir}")
        print(
            f"Checkpoint Time = {t1-t0:.4f}\n"
        )