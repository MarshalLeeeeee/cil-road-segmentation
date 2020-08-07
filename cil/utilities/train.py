import os

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from cil.models.unet_multi_dilation import UNetSegMultiDilation
from cil.utilities.checkpoint import CheckpointManager
from cil.utilities.dataset import get_training_set, get_validation_set
from cil.utilities.utils import ensure_dir, get_logger, OutputType, get_name

# Path settings
dataset_root_dir = 'dataset/'
model_dir = 'trained_models/'
valid_dir = 'validation/'
log_dir = 'log/'

# Training parameters
# if train_all == True, then we will not reserve training samples for validation
train_all = True

bn_eps = 1e-5
bn_momentum = 0.1
base_lr = 1e-2
weight_decay = 5e-4
batch_size = 4
max_epochs = 10000

# boundary loss
edge_penalty = True
base_edge_weight = 1.

# Checkpoint settings
checkpoint_interval = 250
keep_previous_checkpoints = True
# Resume learning from a checkpoint
resume_mode = False
resume_checkpoint = 'UNetMultiSeg-12800'


def _get_train_dataloader():
    dataset = get_training_set(dataset_root_dir, train_all=train_all)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)


def _get_valid_dataloader():
    dataset = get_validation_set(dataset_root_dir)
    return DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)


# Select appropriate criterion based on model names
class LossCalculator:
    def __init__(self, model):
        super(LossCalculator, self).__init__()
        self.output_type = model.output_type
        if self.output_type == OutputType.LOGIT:
            self.criterion = nn.CrossEntropyLoss()
        elif self.output_type == OutputType.PROBABILITY:
            self.criterion = nn.BCELoss()
        else:
            raise ValueError('No appropriate loss has been defined for model {}.'.format(get_name(model)))

    def calculate_loss(self, inputs, targets, edge_mask=None, edge_weight=0.):
        if self.output_type == OutputType.LOGIT:
            loss = self.criterion(inputs, targets)

            # Apply boundary loss
            if edge_penalty and edge_weight > 0:
                prob_foreground = nn.functional.log_softmax(inputs, dim=1)[:, 1]
                edge_loss_tensor = -(edge_mask * prob_foreground)
                edge_loss = torch.mean(edge_loss_tensor[edge_mask != 0])

                loss = loss + edge_weight * edge_loss

            return loss

        elif self.output_type == OutputType.PROBABILITY:
            # Remove channel dimension, which should always be one
            inputs = torch.squeeze(inputs, 1)
            # Match types
            targets = targets.float()
            return self.criterion(inputs, targets)


def _init_weight(model, bn_eps, bn_momentum):
    for param in model.parameters():
        if isinstance(param, (nn.Conv2d, nn.Conv3d)):
            nn.init.kaiming_normal_(param.weight, mode='fan_in', nonlinearity='relu')
        elif isinstance(param, nn.BatchNorm2d):
            param.eps = bn_eps
            param.momentum = bn_momentum
            nn.init.constant_(param.weight, 1)
            nn.init.constant_(param.bias, 0)


def _init_optimizer(model, learning_rate):
    params_with_weight_decay = list()
    params_without_weight_decay = list()

    for name, param in model.named_parameters():
        if 'bn' in name or 'bias' in name:
            params_without_weight_decay.append(param)
        else:
            params_with_weight_decay.append(param)

    optimizer = torch.optim.AdamW([
        {'params': params_without_weight_decay, 'weight_decay': 0.},
        {'params': params_with_weight_decay, 'weight_decay': weight_decay}
    ], learning_rate)
    return optimizer


def train():
    model = UNetSegMultiDilation(in_channels=3, out_channels=2, bn_eps=bn_eps, bn_momentum=bn_momentum)
    optimizer = _init_optimizer(model, base_lr)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Initialize training helpers
    loss_calculator = LossCalculator(model)
    train_loader = _get_train_dataloader()
    valid_loader = _get_valid_dataloader()

    # Initialize some system helpers
    logger = get_logger(log_dir, model)
    ensure_dir(valid_dir)
    ensure_dir(model_dir)
    checkpoint_manager = CheckpointManager(model_dir, keep_previous_checkpoints)

    # Check if model is in resume mode
    if resume_mode:
        assert resume_checkpoint.startswith(get_name(model))
        start_epoch = checkpoint_manager.load_checkpoint(resume_checkpoint, model, optimizer)
    else:
        _init_weight(model, bn_eps=bn_eps, bn_momentum=bn_momentum)
        start_epoch = 0

    # Log basic information
    edge_weight = base_edge_weight
    logger.info("model: {}, batch: {}, learning_rate {}".format(get_name(model), batch_size, base_lr))

    # start training process
    for epoch in range(start_epoch, max_epochs):

        # Edge weight decay strategy
        if epoch + 1 % 1000 == 0:
            edge_weight /= 2.

        if epoch > 4000:
            edge_weight = 0.

        # Training run
        model.train()
        running_loss = 0.0
        for sample in train_loader:
            # Load data
            images = sample['image'].to(device)
            groundtruths = sample['groundtruth'].to(device)
            edge_mask = sample['edge_mask'].to(device)

            # Forward pass
            outputs = model(images)
            loss = loss_calculator.calculate_loss(outputs, groundtruths, edge_mask, edge_weight)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # for logging purpose
            running_loss += loss.item()

        # Validation run
        model.eval()
        valid_loss = 0.0
        with torch.no_grad():
            for sample in valid_loader:
                # Load data
                images = sample['image'].to(device)
                groundtruths = sample['groundtruth'].to(device)
                edge_mask = sample['edge_mask'].to(device)

                # Evaluate
                outputs = model(images)
                loss = loss_calculator.calculate_loss(outputs, groundtruths, edge_mask, base_edge_weight)

                # Calculate loss
                valid_loss += loss.item()

        digest = "Epoch [{}/{}], Edge weight: {:.4f}, Running loss: {:.4f}, Validation loss: {:.4f}".format(
            epoch + 1, max_epochs, edge_weight, running_loss / len(train_loader), valid_loss / len(valid_loader))
        logger.info(digest)

        # Logging and persisting
        if (epoch + 1) % checkpoint_interval == 0:
            checkpoint_manager.checkpoint_model(model, optimizer, epoch + 1)

    # Finish training
    save_path = os.path.join(model_dir, get_name(model))
    torch.save(model.state_dict(), save_path)
    checkpoint_manager.cleanup()


if __name__ == "__main__":
    train()
