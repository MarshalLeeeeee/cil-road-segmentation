import os

import torch

from cil.utilities.utils import ensure_dir


class CheckpointManager:

    def __init__(self, checkpoint_dir, keep_previous_checkpoints=True):
        self.keep_previous_checkpoints = keep_previous_checkpoints
        self.previous_path = None
        self.checkpoint_dir = checkpoint_dir

        ensure_dir(checkpoint_dir)

    def checkpoint_model(self, model, optimizer, epoch):
        checkpoint_name = type(model).__name__ + '-' + str(epoch)
        checkpoint_path = os.path.join(self.checkpoint_dir, checkpoint_name)

        torch.save({
            'model_state': model.state_dict(),
            'optimizer_state': optimizer.state_dict(),
            'epoch': epoch
        }, checkpoint_path)

        if not self.keep_previous_checkpoints:
            if self.previous_path is not None and os.path.exists(self.previous_path):
                os.remove(self.previous_path)
            self.previous_path = checkpoint_path

        print("Checkpoint saved: {}".format(checkpoint_path))

    # Load model for training
    def load_checkpoint(self, checkpoint_name, model, optimizer):
        checkpoint_path = os.path.join(self.checkpoint_dir, checkpoint_name)
        checkpoint = torch.load(checkpoint_path)

        model.load_state_dict(checkpoint['model_state'])
        optimizer.load_state_dict(checkpoint['optimizer_state'])
        # not a good design... any suggestions?
        return checkpoint['epoch']

    # Load model for evaluation
    def load_model(self, checkpoint_name, model):
        checkpoint_path = os.path.join(self.checkpoint_dir, checkpoint_name)
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state'])

    def cleanup(self):
        if not self.keep_previous_checkpoints:
            if self.previous_path is not None and os.path.exists(self.previous_path):
                os.remove(self.previous_path)
            self.previous_path = None
