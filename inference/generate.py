import torch
from PIL import Image
import torchvision.transforms as T
from transformers import AutoTokenizer

from multimodal.vlm_model import VLM
from inference.paged_kv_cache import PagedKVCache
from inference.sampling import top_p
from utils.config import load_config
from dataset.instruction_format import build_prompt


class Generator:

    def __init__(self, checkpoint=None, vocab=None, device="cuda"):

        if vocab is None:
            model_cfg = load_config("configs/model.yaml")
            vocab = int(model_cfg["vocab_size"])

        self.device = device

        self.vocab = vocab
        self.model = VLM(vocab=vocab).to(device)

        if checkpoint is not None:

            ckpt = torch.load(checkpoint, map_location=device)

            self.model.load_state_dict(ckpt["model"])

        self.model.eval()

        self.tokenizer = AutoTokenizer.from_pretrained("gpt2", use_fast=True)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.eos_token_id = self.tokenizer.eos_token_id

        self.transform = T.Compose([
            T.Resize((224,224)),
            T.ToTensor()
        ])

        self.cache = PagedKVCache(
            layers=len(self.model.text.layers),
            heads=8,
            head_dim=96
        )


    def preprocess_image(self, path):

        img = Image.open(path).convert("RGB")

        img = self.transform(img)

        return img.unsqueeze(0).to(self.device)


    def tokenize(self, text):

        prompt = build_prompt(text)
        encoded = self.tokenizer(
            prompt,
            return_tensors="pt",
            padding=False,
            truncation=True,
            max_length=512
        )
        return encoded["input_ids"].to(self.device)


    def detokenize(self, tokens):

        return self.tokenizer.decode(tokens.tolist(), skip_special_tokens=True)


    @torch.no_grad()
    def generate(
        self,
        image_path,
        prompt,
        max_tokens=64,
        temperature=0.8,
        top_p_val=0.9
    ):

        image = self.preprocess_image(image_path)

        tokens = self.tokenize(prompt)
        prompt_len = tokens.shape[1]

        self.cache.reset()

        for _ in range(max_tokens):

            outputs = self.model(
                image=image,
                tokens=tokens,
                kv_cache=self.cache
            )

            # unpack model output
            if isinstance(outputs, tuple):
                logits = outputs[0]
            else:
                logits = outputs

            next_logits = logits[:, -1, :]

            next_token = top_p(
                next_logits[0],
                p=top_p_val,
                temp=temperature
            ).unsqueeze(0)

            tokens = torch.cat(
                [tokens, next_token],
                dim=1
            )

            if next_token.item() == self.eos_token_id:
                break

        # return only the generated portion, excluding the prompt
        generated = tokens[0][prompt_len:]
        return self.detokenize(generated)


if __name__ == "__main__":

    import sys
    import os
    import glob
    import readline  # enables backspace, arrow keys, and line editing in input()

    # Fix terminal so backspace/arrow keys work in VS Code and other terminals
    if sys.stdin.isatty():
        os.system("stty sane")

    ckpts = sorted(glob.glob("outputs/checkpoint_*.pt") + glob.glob("experiments/*/checkpoint_*.pt"))
    checkpoint = ckpts[-1] if ckpts else None
    if checkpoint:
        print("Using checkpoint:", checkpoint)

    gen = Generator(checkpoint=checkpoint)

    while True:
        try:
            img = input("image path: ").strip()
            prompt = input("prompt: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not img or not prompt:
            print("Please provide both image path and prompt.")
            continue
        out = gen.generate(img, prompt)
        print("\nModel:", out)
        try:
            cont = input("\nContinue? (y/n): ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if cont != "y":
            break