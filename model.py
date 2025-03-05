import torch
from transformers import AutoProcessor, MllamaForConditionalGeneration
from peft import get_peft_model, LoraConfig
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

from utils import get_polices, apply_fsdp_checkpointing

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
        
    if len(processor.tokenizer) > model.get_input_embeddings().weight.shape[0]:
        print("WARNING: Resizing the embedding matrix to match the tokenizer vocab size.")
        model.resize_token_embeddings(len(processor.tokenizer))
    
    if train_config.use_peft:
        peft_config = LoraConfig(
            r=16,
            lora_alpha=32,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            lora_dropout=0.05,
            bias="none",
        )
        model = get_peft_model(model, peft_config)
        print("Applied PEFT (LoRA) to model")
        
        # Convert all PEFT adapter parameters to bfloat16
        for name, param in model.named_parameters():
            if param.dtype == torch.float32:
                param.data = param.data.to(torch.bfloat16)
                print(f"Converted parameter {name} from float32 to bfloat16")
    
    return model, processor

def prepare_model_for_training(model, train_config, fsdp_config, device_id):
    mixed_precision_policy, wrap_policy = get_polices()
    
    model = FSDP(
        model,
        auto_wrap_policy=wrap_policy,
        cpu_offload=None,
        mixed_precision=mixed_precision_policy,
        sharding_strategy=fsdp_config.sharding_strategy,
        device_id=device_id,
        limit_all_gathers=True,
        use_orig_params=True
    )
    
    if fsdp_config.fsdp_activation_checkpointing:
        model.enable_input_require_grads()
        model.gradient_checkpointing_enable()
        apply_fsdp_checkpointing(model)
    
    return model