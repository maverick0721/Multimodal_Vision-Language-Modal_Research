import json
import matplotlib.pyplot as plt

with open("logs.json") as f:

    logs = json.load(f)

steps = [l["step"] for l in logs]

loss = [l["loss"] for l in logs]

plt.plot(steps,loss)

plt.xlabel("step")
plt.ylabel("loss")

plt.savefig(
    "training_curve.png",
    dpi=300
)