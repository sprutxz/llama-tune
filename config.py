from dataclasses import dataclass
from torch.distributed.fsdp import ShardingStrategy
from torch.distributed.fsdp.fully_sharded_data_parallel import StateDictType

@dataclass
class TrainConfig:
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
class FSDPConfig:
    mixed_precision: bool = True
    use_fp16: bool=False
    sharding_strategy: ShardingStrategy = ShardingStrategy.FULL_SHARD # HYBRID_SHARD "Full Shard within a node DDP cross Nodes", SHARD_GRAD_OP "Shard only Gradients and Optimizer States", NO_SHARD "Similar to DDP".
    hsdp : bool =False # Require HYBRID_SHARD to be set. This flag can extend the HYBRID_SHARD by allowing sharding a model on customized number of GPUs (Sharding_group) and Replicas over Sharding_group.
    sharding_group_size: int=0 # requires hsdp to be set. This specifies the sharding group size, number of GPUs that you model can fit into to form a replica of a model.
    replica_group_size: int=0 #requires hsdp to be set. This specifies the replica group size, which is world_size/sharding_group_size.
    checkpoint_type: StateDictType = StateDictType.FULL_STATE_DICT  # alternatively FULL_STATE_DICT can be used. SHARDED_STATE_DICT saves one file with sharded weights per rank while FULL_STATE_DICT will collect all weights on rank 0 and save them in a single file.
    fsdp_activation_checkpointing: bool=True
    fsdp_cpu_offload: bool=False
    pure_bf16: bool = True
    optimizer: str= "AdamW"