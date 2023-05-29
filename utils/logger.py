import os
import json
from pathlib import Path

import numpy as np

class Logger(object):
    def __init__(self, path, resume, wandb=None):
        import shutil

        self.metrics_path = path
        self.metrics_file = os.path.join(path, "metrics.jsonl")
        self.interventions_path = os.path.join(path, "interventions.csv")

        if not resume:
            try:
                os.unlink(self.metrics_file)
                os.unlink(self.interventions_path)
            except FileNotFoundError:
                pass

        self.wandb = wandb

    def log_graphs(self, iteration, model, samples=100):
        graphs = []
        if model.ensemble:
            for i in range(len(model.all_graphs)):
                if getattr(model.all_graphs[0], "to_amat", False):
                    G = (model.all_graphs[i].to_amat() != 0).astype(np.uint8)
                else:
                    G = np.array(model.all_graphs[i])
                graphs.append(G)
        else:
            Gs = model.sample(samples)
            for G in Gs:
                graphs.append(G)
        np.savez_compressed(Path(self.metrics_path) / f'graphs_{iteration}.npz', graphs=np.stack(graphs))

    def log_interventions(self, iteration, nodes, values):
        with open(self.interventions_path, "a") as f:
            entry = {
                'nodes': (nodes.astype(np.uint8)).tolist(),
                'values': values[0].tolist()
            }
            entry = json.dumps(entry)
            f.write(f"{iteration},{entry}\n")
            f.close()

    def log_metrics(self, scalars):
        for key, value in scalars.items():
            if self.wandb is not None:
                self.wandb.log({key: value})
            print(f"{key}: {value}", flush = True)

        try:
            with open(self.metrics_file, "a") as f:
                f.write(json.dumps({**scalars}) + "\n")
                f.close()
        except:
            import pdb; pdb.set_trace()