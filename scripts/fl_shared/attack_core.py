from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F
from aijack.attack.inversion import GradientInversionAttackServerManager
from aijack.collaborative.fedavg import FedAVGAPI
from torch.utils.data import DataLoader


def build_manager(
    num_trials: int,
    attack_iters: int,
    distance: str,
    attack_lr: float,
    device: torch.device,
    image_shape: tuple[int, int, int],
):
    kwargs = {
        "num_trial_per_communication": num_trials,
        "log_interval": 10,
        "num_iteration": attack_iters,
        "distancename": distance,
        "device": device,
        "lr": attack_lr,
    }

    try:
        manager = GradientInversionAttackServerManager(image_shape, **kwargs)
        return manager, kwargs
    except TypeError:
        kwargs_without_device = {
            "num_trial_per_communication": num_trials,
            "log_interval": 10,
            "num_iteration": attack_iters,
            "distancename": distance,
            "lr": attack_lr,
        }
        manager = GradientInversionAttackServerManager(image_shape, **kwargs_without_device)
        return manager, kwargs_without_device


def run_attack_api(
    server,
    client,
    dataloader: DataLoader,
    lr: float,
    device: torch.device,
):
    local_dataloaders = [dataloader]
    local_optimizers = [torch.optim.SGD(client.parameters(), lr=lr)]

    def criterion(outputs: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        targets = labels.view(-1).long()
        return F.nll_loss(outputs, targets)

    api_kwargs = dict(
        server=server,
        clients=[client],
        criterion=criterion,
        local_optimizers=local_optimizers,
        local_dataloaders=local_dataloaders,
        num_communication=1,
        local_epoch=1,
        use_gradients=True,
    )

    try:
        api = FedAVGAPI(device=device, **api_kwargs)
    except TypeError:
        api = FedAVGAPI(**api_kwargs)

    api.run()


def find_reconstruction_tensor(obj: Any) -> torch.Tensor | None:
    if torch.is_tensor(obj):
        if obj.ndim == 3:
            return obj.unsqueeze(0)
        if obj.ndim == 4:
            return obj
        if obj.ndim == 5:
            n_trials, batch_size, channels, height, width = obj.shape
            return obj.reshape(n_trials * batch_size, channels, height, width)
        return None

    if isinstance(obj, dict):
        for value in obj.values():
            found = find_reconstruction_tensor(value)
            if found is not None:
                return found
        return None

    if isinstance(obj, (list, tuple)):
        for item in obj:
            found = find_reconstruction_tensor(item)
            if found is not None:
                return found

    return None

