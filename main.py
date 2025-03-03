import copy
from dataclasses import dataclass
import functools
from functools import partial
import itertools
from itertools import islice
import os
from pathlib import Path
import random
import time
from typing import Iterator, List

import numpy as np
from tqdm import tqdm

from datasets import DatasetDict, load_dataset
import torch
import torch.distributed as dist
import torch.distributed._shard.checkpoint as dist_cp
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    CheckpointImpl,
    apply_activation_checkpointing,
    checkpoint_wrapper,
)
from torch.distributed.fsdp import FullStateDictConfig, MixedPrecision, ShardingStrategy
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.fully_sharded_data_parallel import StateDictType
from torch.distributed.fsdp.wrap import (
    size_based_auto_wrap_policy,
    transformer_auto_wrap_policy,
)
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import Sampler
from transformers import AutoProcessor, MllamaForConditionalGeneration
from transformers.models.llama.modeling_llama import LlamaDecoderLayer
from transformers.models.mllama.modeling_mllama import (
    MllamaCrossAttentionDecoderLayer,
    MllamaSelfAttentionDecoderLayer,
    MllamaVisionEncoderLayer,
)
import wandb


# Add GPU memory tracking functions
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

@dataclass
class train_config:
    model_name: str="meta-llama/Llama-3.2-11B-Vision-Instruct"
    batch_size_training: int=2
    batching_strategy: str="padding" #alternative is packing but vision model doesn't work with packing.
    context_length: int =4096
    gradient_accumulation_steps: int=4
    num_epochs: int=3
    lr: float=1e-5
    weight_decay: float=0.0
    gamma: float= 0.85 # multiplicatively decay the learning rate by gamma after each epoch
    seed: int=42
    use_fp16: bool=False
    mixed_precision: bool=True
    val_batch_size:int = 1
    use_peft: bool = True 
    output_dir: str = "/common/users/lms548/models"
    enable_fsdp: bool = True
    dist_checkpoint_root_folder: str="/common/users/lms548/FSDP/model" # will be used if using FSDP
    dist_checkpoint_folder: str="fine-tuned" # will be used if using FSDP
    save_optimizer: bool=False # will be used if using FSDP
    
@dataclass
class fsdp_config:
    mixed_precision: bool = True
    use_fp16: bool=True
    sharding_strategy: ShardingStrategy = ShardingStrategy.FULL_SHARD # HYBRID_SHARD "Full Shard within a node DDP cross Nodes", SHARD_GRAD_OP "Shard only Gradients and Optimizer States", NO_SHARD "Similar to DDP".
    hsdp : bool =False # Require HYBRID_SHARD to be set. This flag can extend the HYBRID_SHARD by allowing sharding a model on customized number of GPUs (Sharding_group) and Replicas over Sharding_group.
    sharding_group_size: int=0 # requires hsdp to be set. This specifies the sharding group size, number of GPUs that you model can fit into to form a replica of a model.
    replica_group_size: int=0 #requires hsdp to be set. This specifies the replica group size, which is world_size/sharding_group_size.
    checkpoint_type: StateDictType = StateDictType.FULL_STATE_DICT  # alternatively FULL_STATE_DICT can be used. SHARDED_STATE_DICT saves one file with sharded weights per rank while FULL_STATE_DICT will collect all weights on rank 0 and save them in a single file.
    fsdp_activation_checkpointing: bool=True
    fsdp_cpu_offload: bool=False
    pure_bf16: bool = False
    optimizer: str= "AdamW"
    
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

# check system prompt token seq or user prompt token seq is in the current token list
def check_header(targets, seq):
    for i in range(len(seq) - 3):
        if seq[i : i + 3] in targets:
            return True
    return False


def replace_target(target, seq):
    for i in range(len(seq) - 3):
        if seq[i : i + 3] == target:
            seq[i], seq[i + 1], seq[i + 2] = -100, -100, -100
    return seq

def get_custom_dataset():
    dataset_dict = load_dataset("RekaAI/VibeEval")
    dataset = dataset_dict["test"]
    
    return DatasetDict({
        "train": dataset.select(range(10,268)),
        "test": dataset.select(range(10))
    })

def tokenize_dialogs(dialogs, images, processor):
    text_prompt = processor.apply_chat_template(dialogs)
    text_prompt = [prompt.replace('<|begin_of_text|>','') for prompt in text_prompt]
    batch = processor(
        images=images,
        text=text_prompt,
        padding=True,
        return_tensors="pt",
    )
    label_list = []
    for i in range(len(batch["input_ids"])):
        dialog_tokens = batch["input_ids"][i].tolist()
        labels = copy.copy(dialog_tokens)
        eot_indices = [i for i, n in enumerate(labels) if n == 128009]
        last_idx = 0
        # system prompt header "<|start_header_id|>system<|end_header_id|>" has been tokenized to [128006, 9125, 128007]
        # user prompt header "<|start_header_id|>user<|end_header_id|>" has been tokenized to [128006, 882, 128007]
        prompt_header_seqs = [[128006, 9125, 128007], [128006, 882, 128007]]
        for n, idx in enumerate(eot_indices):
            current_seq = labels[last_idx : idx + 1]
            if check_header(prompt_header_seqs, current_seq):
                # found prompt header, indicating that this seq should be masked
                labels[last_idx : idx + 1] = [-100] * (idx - last_idx + 1)
            else:
                last_idx = idx + 1
            #  Mask all the assistant header prompt <|start_header_id|>assistant<|end_header_id|>, which has been tokenized to [128006, 78191, 128007]
        assistant_header_seq = [128006, 78191, 128007]
        labels = replace_target(assistant_header_seq, labels)
        # Mask the padding token and image token 128256
        for i in range(len(labels)):
            if (
                labels[i] == processor.tokenizer.pad_token_id or labels[i] == 128256
            ):  #  128256 is image token index
                labels[i] = -100
        label_list.append(labels)
    batch["labels"] = torch.tensor(label_list)
    return batch
    
class VibeEvalDataCollator:
    
    def __init__(self, processor):
        self.processor = processor
        self.processor.tokenizer.padding_side = "right"
    
    def __call__(self, samples):
        dialogs, images = [], []
        for sample in samples:  
            image, prompt, reference = sample["image"], sample["prompt"], sample["reference"]
            image = resize_image(image.convert("RGB"))
            dialog = [
                {
                    "role":"user",
                    "content": [
                        {
                            "type":"image"
                        },
                        {
                            "type":"text",
                            "text":prompt.strip()
                        }
                    ]
                },
                {
                    "role":"assistant",
                    "content":[
                        {
                            "type":"text",
                            "text":reference.strip()
                        }
                    ]
                }
            ]
            dialogs.append(dialog)
            images.append([image])      
        return tokenize_dialogs(dialogs,images, self.processor)
    
def get_data_collator(processor):
    return VibeEvalDataCollator(processor)



class LengthBasedBatchSampler(torch.utils.data.BatchSampler):
    
    def __init__(self, data_source, batch_size: int, drop_last: bool, shuffle: bool=True):
        if isinstance(next(iter(data_source)), dict):
            self.lengths = [len(d["prompt"]) for d in data_source]
        
        else:
            self.lengths = [len(d) for d in data_source]

        self.batch_size = batch_size
        self.drop_last = drop_last
        self.shuffle = shuffle
    
    def __iter__(self):

        ids = np.argsort(self.lengths, kind='mergesort')
        #we now have index sorted in ascending order
        if self.drop_last:
            ids = ids[:len(ids)//self.batch_size*self.batch_size]
            
        batches = [ids[i:i+self.batch_size] for i in range(0, len(ids), self.batch_size)]
        
        if self.shuffle:
            random.shuffle(batches)
        
        for b in batches:
            yield b
    
    def __len__(self):
        if self.drop_last:
            return len(self.lengths)//self.batch_size
        
        return len(self.lengths) // self.batch_size + (len(self.lengths) % self.batch_size > 0)
    
class DistributedLengthBasedBatchSampler(torch.utils.data.BatchSampler):
    
    def __init__(self, data_source, batch_size: int, num_replicas: int, rank: int, shuffle: bool = True, seed: int = 0):
        random.seed(seed)
        self.batch_sampler = LengthBasedBatchSampler(
            data_source, batch_size=batch_size, drop_last=True, shuffle=shuffle
        )
        self.num_replicas = num_replicas
        self.rank = rank
        print(self.num_replicas)

    def __iter__(self):
        max_length = len(self.batch_sampler) // self.num_replicas * self.num_replicas
        return islice(self.batch_sampler, self.rank, max_length, self.num_replicas)
        #Here, since we are creating distributed sampler. what islice is doing here is just we are filtering batches with params (iterable, start, end, steps)
        #for example we have [batch_0, batch_1, batch_2, batch_3, batch_4, batch_5, batch_6, batch_7, batch_8, batch_9, batch_10, ...] , max_length=10, replica=2
        #for rank 1 it will only use [batch_1, batch_3, batch_5, batch_7, batch_9]
    
    def __len__(self):
        return len(self.batch_sampler) // self.num_replicas

def get_dataloaders(train_config, processor):
    data = get_custom_dataset()
    dataset_train = data["train"]
    dataset_val = data["test"]
    
    data_collator = get_data_collator(processor)
    train_kwargs = {
        "batch_sampler": DistributedLengthBasedBatchSampler(
                dataset_train,
                batch_size=train_config.batch_size_training,
                rank=dist.get_rank(),
                num_replicas=dist.get_world_size(),
                shuffle=True,
            ),
        "collate_fn" : data_collator
    }
    train_dataloader = torch.utils.data.DataLoader(
        dataset_train,
        num_workers=2,
        pin_memory=True,
        **train_kwargs,
    )
    
    valid_kwargs = {
        "batch_sampler": DistributedLengthBasedBatchSampler(
                dataset_val,
                batch_size=1,
                rank=dist.get_rank(),
                num_replicas=dist.get_world_size(),
                shuffle=False,
            ),
        "collate_fn" : data_collator
    }
    valid_dataloader = torch.utils.data.DataLoader(
        dataset_val,
        num_workers=2,
        pin_memory=True,
        **valid_kwargs,
    )
    
    return train_dataloader, valid_dataloader

# class DisHallDataCollator:
#     def __init__(self, processor):
#         self.processor = processor
#         self.processor.tokenizer.padding_side = (
#             "right"  # during training, one always uses padding on the right
#         )
#
#     def __call__(self, samples):
#         dialogs, images = [], []
#         for sample in samples:
#             image, qa_pairs = sample["image"], sample["QA_pairs"]
#
#             image = image.convert("RGB")  # only use the first image
#
#             for qnas in qa_pairs:
#                 dialog = [
#                     {
#                         "role": "user",
#                         "content": [
#                             {"type": "image"},
#                             {"type": "text", "text": qnas["question"].strip()},
#                         ],
#                     },
#                     {
#                         "role": "assistant",
#                         "content": [
#                             {"type": "text", "text": qnas["answer"].strip()},
#                         ],
#                     },
#                 ]
#                 dialogs.append(dialog)
#                 images.append(image)
#         return tokenize_dialogs(dialogs, images, self.processor) 
#
# def get_data_collator(processor):
#     return DisHallDataCollator(processor)

# class SimpleDistributedSampler(Sampler):
#     """
#     A simplified distributed sampler that evenly divides the dataset across processes.
#     """
#
#     def __init__(self, dataset, num_replicas: int, rank: int, shuffle: bool = True, seed: int = 54):
#         """
#         Args:
#             dataset: Dataset to sample from
#             num_replicas: Number of processes/GPUs
#             rank: Process rank
#             shuffle: Whether to shuffle the indices
#             seed: Random seed for reproducibility
#         """
#         self.dataset = dataset
#         self.num_replicas = num_replicas
#         self.rank = rank
#         self.epoch = 0
#         self.shuffle = shuffle
#         self.seed = seed
#
#         # Make sure each process gets same number of batches
#         self.num_samples = len(self.dataset) // self.num_replicas
#         self.total_size = self.num_samples * self.num_replicas
#
#     def __iter__(self) -> Iterator[int]:
#         # Deterministically shuffle based on epoch and seed
#         if self.shuffle:
#             g = torch.Generator()
#             g.manual_seed(self.seed + self.epoch)
#             indices = torch.randperm(len(self.dataset), generator=g).tolist()
#         else:
#             indices = list(range(len(self.dataset)))
#
#         # Truncate to make it evenly divisible across processes
#         indices = indices[:self.total_size]
#         assert len(indices) == self.total_size
#
#         # Get subset for this rank
#         indices = indices[self.rank:self.total_size:self.num_replicas]
#         assert len(indices) == self.num_samples
#
#         return iter(indices)
#
#     def __len__(self) -> int:
#         return self.num_samples
#
#     def set_epoch(self, epoch: int) -> None:
#         """
#         Sets the epoch for this sampler. This ensures different shuffling
#         order at each epoch when shuffle=True
#         """
#         self.epoch = epoch
#
# def get_dataloaders(train_config, processor):
#     # Get dataset using the dataset config and processor
#     dataset_train = get_custom_dataset(None, processor, "train")
#     dataset_val = get_custom_dataset(None, processor, "test")
#
#     data_collator = get_data_collator(processor)
#
#     # Create train sampler using SimpleDistributedSampler
#     train_sampler = SimpleDistributedSampler(
#         dataset=dataset_train,
#         num_replicas=dist.get_world_size(),
#         rank=dist.get_rank(),
#         shuffle=True
#     )
#
#     # Create train dataloader with batch_size parameter (instead of batch_sampler)
#     train_dataloader = torch.utils.data.DataLoader(
#         dataset_train,
#         batch_size=train_config.batch_size_training,
#         sampler=train_sampler,
#         collate_fn=data_collator,
#         num_workers=2,
#         pin_memory=True,
#     )
#
#     # Create validation sampler
#     val_sampler = SimpleDistributedSampler(
#         dataset=dataset_val,
#         num_replicas=dist.get_world_size(),
#         rank=dist.get_rank(),
#         shuffle=False
#     )
#
#     # Create validation dataloader
#     valid_dataloader = torch.utils.data.DataLoader(
#         dataset_val,
#         batch_size=train_config.val_batch_size,
#         sampler=val_sampler,
#         collate_fn=data_collator,
#         num_workers=2,
#         pin_memory=True,
#     )
#
#     return train_dataloader, valid_dataloader

def get_model_and_processor(train_config):
    model = MllamaForConditionalGeneration.from_pretrained(
        train_config.model_name,
        torch_dtype=torch.bfloat16,
        device_map=None,
    )
    processor = AutoProcessor.from_pretrained(train_config.model_name)
    processor.tokenizer.padding_side='right'
    model.supports_gradient_checkpointing = True
    model.language_model.supports_gradient_checkpointing = True
    
    if not processor.tokenizer.pad_token_id:
        processor.tokenizer.pad_token_id = processor.tokenizer.eos_token_id
        
    # If there is a mismatch between tokenizer vocab size and embedding matrix,
    # throw a warning and then expand the embedding matrix
    if len(processor.tokenizer) > model.get_input_embeddings().weight.shape[0]:
        print("WARNING: Resizing the embedding matrix to match the tokenizer vocab size.")
        model.resize_token_embeddings(len(processor.tokenizer))
    
    return model, processor

def get_polices():
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

    check_fn = lambda submodule: isinstance(submodule, LlamaDecoderLayer) #checking the layer we want to add checkpoint
    #check_fn checks if submodule is LlamaDecoderLayer
    apply_activation_checkpointing(
        model, checkpoint_wrapper_fn=non_reentrant_wrapper, check_fn=check_fn
    )
    
def save_fsdp_model_checkpoint_full(
    model,
    optimizer,
    rank,
    cfg,
    epoch=1,
):
    """saving model via rank0 cpu streaming and full_state_dict
    THis means there is an upper limit of model sizes that can be saved based on CPU memory. For larger model use local state dict
    """

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

def save_model_and_optimizer_sharded(model, rank, cfg,optim=None):
    """save model and optimizer via sharded_state_dict to save_dir"""
    
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

def train(model, train_dataloader, eval_dataloader, tokenizer, optimizer, lr_scheduler, gradient_accumulation_steps, train_config, fsdp_config=None, local_rank=None, rank=None, wandb_run=None):
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
        
        wandb.log({"train_perplexity": train_perplexity, "train_epoch_loss": train_epoch_loss})
        
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

def main():
    torch.manual_seed(train_config.seed)
    random.seed(train_config.seed)
    
    local_rank = 0
    rank = 0
    
    wandb.init(
        project="llama-tune",
        config = {
            "model" : train_config.model_name,
            "epochs" : train_config.num_epochs,
            "lr" : train_config.lr,
            "seed" : train_config.seed,
        }
    )
    
    if train_config.enable_fsdp:
        dist.init_process_group("nccl")
        # torchrun specific, it is set automatically by torchrun
        local_rank = int(os.environ["LOCAL_RANK"])
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])

    if dist.is_initialized():
        torch.cuda.set_device(local_rank)
        torch.cuda.empty_cache()
        os.environ["TORCH_SHOW_CPP_STACKTRACES"] = str(1)
        os.environ["NCCL_ASYNC_ERROR_HANDLING"] = str(1)
    
    log_gpu_memory("Before model initialization", reset_peak=True, rank=rank)
    
    model, processor = get_model_and_processor(train_config= train_config)
    
    log_gpu_memory("After model initialization", rank=rank)
    
    mixed_precision_policy, wrap_policy = get_polices()
    device_id = torch.cuda.current_device()
    
    log_gpu_memory("Before FSDP wrapping", rank=rank)
    
    model = FSDP(
            model,
            auto_wrap_policy= wrap_policy,
            cpu_offload=None,
            mixed_precision=mixed_precision_policy,
            sharding_strategy=fsdp_config.sharding_strategy,
            device_id=device_id,
            limit_all_gathers=True
        )
    
    log_gpu_memory("After FSDP wrapping", rank=rank)
    
    if fsdp_config.fsdp_activation_checkpointing:
        model.enable_input_require_grads()
        model.gradient_checkpointing_enable()
        apply_fsdp_checkpointing(model)
        
        log_gpu_memory("After activation checkpointing setup", rank=rank)
    
    train_dataloader , eval_dataloader = get_dataloaders(
        train_config= train_config,
        processor= processor
    )
    
    log_gpu_memory("After dataloader creation", rank=rank)
    
    optimizer = optim.AdamW(
            model.parameters(),
            lr=train_config.lr,
            weight_decay=train_config.weight_decay,
        )
    scheduler = StepLR(optimizer, step_size=1, gamma=train_config.gamma)
    
    log_gpu_memory("After optimizer creation", rank=rank)
    
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
    
    wandb.finish()

if __name__ == "__main__":
    main()
