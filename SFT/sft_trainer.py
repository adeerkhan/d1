import torch
import torch.nn.functional as F
from transformers import Trainer, TrainerCallback
from transformers import DefaultDataCollator
import random
from tqdm import tqdm
import pickle
import torch.distributed as dist
import sys
import os
import math

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from rplan_data_coordtok import pretty_token, TokenizationSchema


class LogMetricsCallback(TrainerCallback):
    """A custom callback to log weight norm."""

    def on_log(self, args, state, control, model=None, logs=None, **kwargs):
        if logs is not None and state.is_world_process_zero and model is not None:
            weight_norm = 0.0
            for param in model.parameters():
                if param.requires_grad:
                    weight_norm += torch.norm(param.data).item() ** 2
            logs["weight_norm"] = weight_norm**0.5


class dLLMTrainer(Trainer):
    def compute_loss(self, model, inputs, num_items_in_batch=None, return_outputs=False):
        """
        Absorbing state diffusion loss computation
        """
        labels, t, num_prompt_tokens = (
            inputs.pop("labels"),
            inputs.pop("t"),
            inputs.pop("num_prompt_tokens"),
        )
        outputs = model(**inputs)
        logits = outputs.logits
        unscaled_loss = F.cross_entropy(
            logits.view(-1, logits.shape[-1]), labels.view(-1), reduction="none"
        ).view(logits.shape[0], -1)
        if (self.state.global_step + 1) % self.args.logging_steps == 0:
            self.log({"unscaled_loss": (unscaled_loss.sum() / (labels != -100).sum()).item()})
        loss = unscaled_loss / t
        loss = loss.sum() / (inputs["input_ids"].numel() - num_prompt_tokens)
        return loss if not return_outputs else (loss, outputs)

    def compute_metrics(self, eval_pred):
        logits, labels = eval_pred.predictions, eval_pred.label_ids
        if isinstance(logits, tuple):
            logits = logits[0]

        labels = torch.from_numpy(labels).to(logits.device).long()

        loss_fct = torch.nn.CrossEntropyLoss()
        loss = loss_fct(logits.reshape(-1, logits.shape[-1]), labels.reshape(-1))

        try:
            perplexity = math.exp(loss.item())
        except OverflowError:
            perplexity = float("inf")

        return {"perplexity": perplexity}


class dLLMSFTDataset(torch.utils.data.Dataset):
    """
    Similar to AR datasets, except in inference, we keep the timsteps fixed
    """

    def __init__(self, data, tokenizer, max_length, eval=False):
        super().__init__()
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.eval = eval
        if self.eval:
            self.t = torch.linspace(0, 1, len(self.data))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        out = self.data[idx]
        if self.eval:
            out["t"] = self.t[idx]
        return out


class dLLMDataCollator(DefaultDataCollator):
    """
    Adds the forward noising process to the batch.
    Modify forward_process to change the noise schedule
    """

    def __init__(self, *args, **kwargs):
        super().__init__()
        self.mask_token_id = kwargs["tokenizer"].mask_token_id
        self.tokenizer = kwargs["tokenizer"]
        if "max_length" in kwargs:
            self.max_length = kwargs["max_length"]
        if kwargs["tokenizer"].mask_token_id is None:
            assert (
                "mask_token_id" in kwargs
            ), "For dLLM models, pass a mask_token_id or set it equal to tokenizer.mask_token_id"
            self.mask_token_id = kwargs["mask_token_id"]

    def forward_process(self, batch, eps=1e-3):
        input_ids = batch["input_ids"]
        B, N = input_ids.shape
        if "t" not in batch:
            t = torch.rand((B,), device=input_ids.device)
        else:
            t = batch["t"]

        t = (1 - eps) * t + eps
        t = t[:, None].repeat(1, N)

        mask_indices = torch.rand((B, N), device=input_ids.device) < t
        noisy_batch = torch.where(mask_indices, self.mask_token_id, input_ids)
        return noisy_batch, t, mask_indices

    def __call__(self, batch):
        batch = super().__call__(batch)
        batch["labels"] = batch["input_ids"].clone()
        noisy_batch, batch["t"], mask_indices = self.forward_process(batch)
        batch["labels"][~mask_indices] = -100
        batch["num_prompt_tokens"] = 0
        if "prompt_lengths" in batch:
            prompt_lengths = batch.pop("prompt_lengths")
            prompt_length_indices = torch.arange(noisy_batch.shape[1]).unsqueeze(0)
            prompt_mask = prompt_length_indices < prompt_lengths
            noisy_batch[prompt_mask] = batch["input_ids"][prompt_mask].clone()
            batch["labels"][prompt_mask] = -100
            batch["num_prompt_tokens"] = prompt_mask.sum()
        batch["input_ids"] = noisy_batch.long()
        return batch


def preprocess_dataset(data: torch.utils.data.Dataset, tokenizer, max_length, test_split=0.01):
    preprocessed_data = []
    schema = data.schema

    for i in tqdm(range(len(data)), desc="Preprocessing dataset"):
        sample = data[i]
        question = sample["caption"]

        floorplan_tokens = sample["input_ids"]
        actual_tokens = floorplan_tokens[floorplan_tokens != schema.PAD_TOKEN].tolist()

        if actual_tokens and actual_tokens[0] == schema.BOS_TOKEN:
            actual_tokens.pop(0)
        if actual_tokens and actual_tokens[-1] == schema.EOS_TOKEN:
            actual_tokens.pop()

        answer = " ".join([pretty_token(tok, schema) for tok in actual_tokens])

        prompt = [{"role": "user", "content": question}]
        response = [{"role": "assistant", "content": answer}]

        inputs = tokenizer.apply_chat_template(prompt + response, tokenize=False)
        prompt_str = tokenizer.apply_chat_template(prompt, tokenize=False) + "\n"

        tokenized_input = tokenizer(
            inputs,
            return_tensors="pt",
            truncation=True,
            max_length=max_length,
            padding="max_length",
        ).input_ids.squeeze(0)

        tokenized_prompt = tokenizer(
            prompt_str, return_tensors="pt", truncation=True, max_length=max_length
        )

        preprocessed_data.append(
            {
                "input_ids": tokenized_input,
                "prompt_lengths": tokenized_prompt.attention_mask.sum(-1),
            }
        )

    random.shuffle(preprocessed_data)
    test_size = int(len(preprocessed_data) * test_split)
    test_data = preprocessed_data[:test_size]
    train_data = preprocessed_data[test_size:]
    return train_data, test_data
