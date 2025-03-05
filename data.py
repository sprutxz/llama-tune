import copy
import torch
from torch.utils.data import Sampler
from typing import Iterator
from datasets import load_dataset

# Dataset functions
def get_custom_dataset(dataset_config, processor, split, split_ratio=0.9):
    # load_dataset will return DatasetDict that contains all the data in the train set
    dataset_dict = load_dataset("Sprutz/DisHall")
    dataset = dataset_dict["train"]
    dataset = dataset.train_test_split(
        test_size=1 - split_ratio, shuffle=True, seed=54
    )[split]
    return dataset

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

class DisHallDataCollator:
    def __init__(self, processor):
        self.processor = processor
        self.processor.tokenizer.padding_side = (
            "right"  # during training, one always uses padding on the right
        )

    def __call__(self, samples):
        dialogs, images = [], []
        for sample in samples:
            image, qa_pairs = sample["image"], sample["QA_pairs"]

            image = image.convert("RGB")  # only use the first image

            for qnas in qa_pairs:
                dialog = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image"},
                            {"type": "text", "text": qnas["question"].strip()},
                        ],
                    },
                    {
                        "role": "assistant",
                        "content": [
                            {"type": "text", "text": qnas["answer"].strip()},
                        ],
                    },
                ]
                dialogs.append(dialog)
                images.append(image)
        return tokenize_dialogs(dialogs, images, self.processor) 

def get_data_collator(processor):
    return DisHallDataCollator(processor)

class SimpleDistributedSampler(Sampler):
    """
    A simplified distributed sampler that evenly divides the dataset across processes.
    """

    def __init__(self, dataset, num_replicas: int, rank: int, shuffle: bool = True, seed: int = 54):
        """
        Args:
            dataset: Dataset to sample from
            num_replicas: Number of processes/GPUs
            rank: Process rank
            shuffle: Whether to shuffle the indices
            seed: Random seed for reproducibility
        """
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        self.shuffle = shuffle
        self.seed = seed

        # Make sure each process gets same number of batches
        self.num_samples = len(self.dataset) // self.num_replicas
        self.total_size = self.num_samples * self.num_replicas

    def __iter__(self) -> Iterator[int]:
        # Deterministically shuffle based on epoch and seed
        if self.shuffle:
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            indices = torch.randperm(len(self.dataset), generator=g).tolist()
        else:
            indices = list(range(len(self.dataset)))

        # Truncate to make it evenly divisible across processes
        indices = indices[:self.total_size]
        assert len(indices) == self.total_size

        # Get subset for this rank
        indices = indices[self.rank:self.total_size:self.num_replicas]
        assert len(indices) == self.num_samples

        return iter(indices)

    def __len__(self) -> int:
        return self.num_samples

    def set_epoch(self, epoch: int) -> None:
        """
        Sets the epoch for this sampler. This ensures different shuffling
        order at each epoch when shuffle=True
        """
        self.epoch = epoch

def get_dataloaders(train_config, processor):
    import torch.distributed as dist
    
    # Get dataset using the dataset config and processor
    dataset_train = get_custom_dataset(None, processor, "train")
    dataset_val = get_custom_dataset(None, processor, "test")

    data_collator = get_data_collator(processor)

    # Create train sampler using SimpleDistributedSampler
    train_sampler = SimpleDistributedSampler(
        dataset=dataset_train,
        num_replicas=dist.get_world_size(),
        rank=dist.get_rank(),
        shuffle=True
    )

    # Create train dataloader with batch_size parameter (instead of batch_sampler)
    train_dataloader = torch.utils.data.DataLoader(
        dataset_train,
        batch_size=train_config.batch_size_training,
        sampler=train_sampler,
        collate_fn=data_collator,
        num_workers=2,
        pin_memory=True,
    )

    # Create validation sampler
    val_sampler = SimpleDistributedSampler(
        dataset=dataset_val,
        num_replicas=dist.get_world_size(),
        rank=dist.get_rank(),
        shuffle=False
    )

    # Create validation dataloader
    valid_dataloader = torch.utils.data.DataLoader(
        dataset_val,
        batch_size=train_config.val_batch_size,
        sampler=val_sampler,
        collate_fn=data_collator,
        num_workers=2,
        pin_memory=True,
    )

    return train_dataloader, valid_dataloader

# Import functions from utils
from utils import check_header, replace_target