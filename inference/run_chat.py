from inference.chat_vlm import VLMChat
import glob
import readline

ckpts = sorted(glob.glob("outputs/checkpoint_*.pt") + glob.glob("experiments/*/checkpoint_*.pt"))
checkpoint = ckpts[-1] if ckpts else None

chat = VLMChat(
    checkpoint=checkpoint
)

print("VLM Chat ready. Type 'quit' to exit.\n")

while True:
    image = input("Image path: ")
    if image.lower() == "quit":
        break
    question = input("Question: ")
    if question.lower() == "quit":
        break

    response = chat.chat(image=image, question=question)
    print("\nModel:", response, "\n")