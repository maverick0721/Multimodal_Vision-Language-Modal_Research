import torch

from multimodal.vlm_model import VLM
from experiments.logger import Logger
from text.lora import apply_lora


torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


model = VLM(vocab=32000).cuda()

# Apply LoRA adapters
apply_lora(model)

# Freeze base model parameters (train only LoRA)
for name, p in model.named_parameters():
    if "lora_" not in name:
        p.requires_grad = False


optimizer = torch.optim.AdamW(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=1e-4
)


logger = Logger()


for step in range(10000):

    images = torch.randn(
        8,3,224,224
    ).cuda()

    tokens = torch.randint(
        0,32000,
        (8,128)
    ).cuda()

    logits = model(images, tokens)

    # Autoregressive language modeling loss
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = tokens[:, 1:].contiguous()

    loss = torch.nn.functional.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1)
    )

    optimizer.zero_grad()

    loss.backward()

    optimizer.step()

    logger.log(step, loss.item())

    print(step, loss.item())