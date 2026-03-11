import json
import torch
from PIL import Image
import torchvision.transforms as T
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoTokenizer

from dataset.instruction_format import build_prompt


# -----------------------------
# Image transform
# -----------------------------

transform = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor()
])


# -----------------------------
# Tokenizer
# -----------------------------

tokenizer = AutoTokenizer.from_pretrained(
    "gpt2",
    use_fast=True
)

tokenizer.pad_token = tokenizer.eos_token


# -----------------------------
# Tokenization helper
# -----------------------------

def tokenize(text):

    tokens = tokenizer(
        text,
        return_tensors="pt",
        padding=False,
        truncation=True,
        max_length=512
    )

    return tokens["input_ids"].squeeze(0)


# -----------------------------
# Dataset
# -----------------------------

class InstructionDataset:

    def __init__(self, json_path):

        with open(json_path) as f:
            self.data = json.load(f)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):

        sample = self.data[idx]

        # image
        image = Image.open(sample["image"]).convert("RGB")
        image = transform(image)

        # text
        question = sample["conversations"][0]["content"]
        answer = sample["conversations"][1]["content"]

        prompt = build_prompt(question)

        prompt_tokens = tokenize(prompt)
        answer_tokens = tokenize(answer)

        # Append EOS so the model learns to stop generating
        eos = torch.tensor([tokenizer.eos_token_id])
        answer_tokens = torch.cat([answer_tokens, eos])

        tokens = torch.cat([prompt_tokens, answer_tokens])

        # Standard labels: mask prompt with -100, keep answer tokens at their positions
        # The shift for next-token prediction is handled in the training loop
        labels = torch.cat([
            torch.full_like(prompt_tokens, -100),
            answer_tokens
        ])

        tokens = tokens.long()
        labels = labels.long()

        # clamp tokens to vocab
        tokens = torch.clamp(tokens, 0, tokenizer.vocab_size - 1)

        # preserve -100 mask
        valid = labels >= 0
        labels[valid] = torch.clamp(
            labels[valid],
            0,
            tokenizer.vocab_size - 1
        )

        return image, tokens, labels


# -----------------------------
# Collate function
# -----------------------------

def collate_fn(batch):

    images = []
    tokens = []
    labels = []

    for img, tok, lab in batch:
        images.append(img)
        tokens.append(tok)
        labels.append(lab)

    images = torch.stack(images)

    tokens = pad_sequence(
        tokens,
        batch_first=True,
        padding_value=tokenizer.pad_token_id
    )

    labels = pad_sequence(
        labels,
        batch_first=True,
        padding_value=-100
    )

    return {
        "image": images,
        "tokens": tokens,
        "labels": labels
    }