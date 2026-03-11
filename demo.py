import torch
import glob

from inference.generate import Generator

device = "cuda" if torch.cuda.is_available() else "cpu"

ckpts = sorted(glob.glob("outputs/checkpoint_*.pt") + glob.glob("experiments/*/checkpoint_*.pt"))

if len(ckpts) == 0:
    raise RuntimeError("No checkpoints found")

checkpoint = ckpts[-1]

print("Using checkpoint:", checkpoint)

generator = Generator(checkpoint)

prompt = "What animal is this?"

image_path = "images/dog.jpg"

output = generator.generate(image_path, prompt)

print("Model output:", output)