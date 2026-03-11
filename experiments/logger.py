import json
import os


class Logger:

    def __init__(self, output_dir):

        self.output_dir = output_dir
        self.path = os.path.join(output_dir, "train_log.json")

        self.data = []

    def log(self, step, loss):

        self.data.append({
            "step": step,
            "loss": loss
        })

        with open(self.path, "w") as f:
            json.dump(self.data, f, indent=2)