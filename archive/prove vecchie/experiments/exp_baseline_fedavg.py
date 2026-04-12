"""FedAVG + Gradient Inversion demo on MedMNIST BloodMNIST.

This script shows how to attach AIJack's GradientInversionAttackServerManager to
FedAVGServer, run FL rounds with FedAVGAPI, and inspect reconstruction results
from `server.attack_results` after training.
"""

import copy
import random

import medmnist
import numpy as np
import torch
import torch.nn as nn
from medmnist import INFO
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import transforms

from aijack.attack.inversion import GradientInversionAttackServerManager
from aijack.collaborative.fedavg import FedAVGAPI, FedAVGClient, FedAVGServer


# --------------------------
# 1. Configuration
# --------------------------
DATA_FLAG = "bloodmnist"
INPUT_SHAPE = (3, 28, 28)
SEED = 42
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

NUM_CLIENTS = 5
TARGET_BENIGN_CLIENT_ID = 0
COMMUNICATION_ROUNDS = 3
LOCAL_EPOCH = 1
BATCH_SIZE = 1  # keep batch size 1 to make gradient inversion feasible

CLIENT_LR = 0.01
SERVER_LR = 0.1


def set_seed(seed: int) -> None:
	random.seed(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)
	if torch.cuda.is_available():
		torch.cuda.manual_seed_all(seed)


# --------------------------
# 2. MedMNIST wrapper (safe label shape)
# --------------------------
class MedMNISTSafe(Dataset):
	def __init__(self, data_flag: str, split: str, transform):
		info = INFO[data_flag]
		data_class = getattr(medmnist, info["python_class"])
		self.dataset = data_class(split=split, download=True, transform=transform)

	def __getitem__(self, idx):
		img, target = self.dataset[idx]
		# BloodMNIST can return labels with shape [1], convert to scalar long.
		target = torch.as_tensor(target).long().squeeze()
		return img, target

	def __len__(self):
		return len(self.dataset)


# --------------------------
# 3. Simple model
# --------------------------
class SimpleCNN(nn.Module):
	def __init__(self, num_classes: int):
		super().__init__()
		self.conv = nn.Sequential(
			nn.Conv2d(3, 32, kernel_size=5, padding=2),
			nn.ReLU(),
			nn.Conv2d(32, 32, kernel_size=3, padding=1),
			nn.ReLU(),
			nn.AdaptiveAvgPool2d((7, 7)),
		)
		self.fc = nn.Linear(32 * 7 * 7, num_classes)

	def forward(self, x):
		x = self.conv(x)
		x = x.view(x.size(0), -1)
		return self.fc(x)


def split_indices_evenly(n_samples: int, n_clients: int, seed: int):
	g = torch.Generator().manual_seed(seed)
	perm = torch.randperm(n_samples, generator=g)
	chunks = torch.chunk(perm, n_clients)
	return [c.tolist() for c in chunks]


def main() -> None:
	set_seed(SEED)

	info = INFO[DATA_FLAG]
	num_classes = len(info["label"])

	transform = transforms.Compose(
		[
			transforms.ToTensor(),
			transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
		]
	)

	train_dataset = MedMNISTSafe(DATA_FLAG, split="train", transform=transform)
	client_indices = split_indices_evenly(len(train_dataset), NUM_CLIENTS, SEED)

	# One benign target client is explicitly defined (id=0).
	# Other clients are still standard FedAVG participants to keep a realistic FL topology.
	client_dataloaders = [
		DataLoader(Subset(train_dataset, idxs), batch_size=BATCH_SIZE, shuffle=False)
		for idxs in client_indices
	]

	base_model = SimpleCNN(num_classes).to(DEVICE)
	clients = [
		FedAVGClient(copy.deepcopy(base_model).to(DEVICE), user_id=i, lr=CLIENT_LR, device=DEVICE)
		for i in range(NUM_CLIENTS)
	]

	local_optimizers = [
		torch.optim.SGD(client.model.parameters(), lr=CLIENT_LR) for client in clients
	]

	# Attach gradient inversion logic to a FedAVG server class.
	manager = GradientInversionAttackServerManager(
		x_shape=INPUT_SHAPE,
		target_client_id=TARGET_BENIGN_CLIENT_ID,
		num_trial_per_communication=1,
		optimize_label=False,
		distancename="l2",
		optimizername="Adam",
		num_iteration=800,
		lr=0.08,
		tv_reg_coef=1e-3,
		l2_reg_coef=1e-6,
		clamp_range=(-1, 1),
		log_interval=100,
		early_stopping=250,
		device=DEVICE,
	)

	AttackFedAVGServer = manager.attach(FedAVGServer)
	server = AttackFedAVGServer(
		clients=clients,
		global_model=copy.deepcopy(base_model).to(DEVICE),
		lr=SERVER_LR,
		device=DEVICE,
	)

	criterion = nn.CrossEntropyLoss()
	api = FedAVGAPI(
		server,
		clients,
		criterion,
		local_optimizers,
		client_dataloaders,
		num_communication=COMMUNICATION_ROUNDS,
		local_epoch=LOCAL_EPOCH,
		use_gradients=True,
		device=DEVICE,
	)

	api.run()

	# --------------------------
	# 4. Access inversion outputs
	# --------------------------
	print(f"Total recorded rounds in server.attack_results: {len(server.attack_results)}")

	target_dataset_idx = client_indices[TARGET_BENIGN_CLIENT_ID][0]
	target_img, target_label = train_dataset[target_dataset_idx]

	for comm_round, trial_results in enumerate(server.attack_results, start=1):
		print(f"\nRound {comm_round}: {len(trial_results)} attack trial(s)")
		if len(trial_results) == 0:
			continue

		recon_x, recon_y = trial_results[0]
		pred_label = int(recon_y[0].item()) if recon_y.ndim == 1 else int(recon_y[0].argmax().item())
		mse = torch.mean((recon_x[0].detach().cpu() - target_img.detach().cpu()) ** 2).item()

		print(f"  target_client_id = {TARGET_BENIGN_CLIENT_ID}")
		print(f"  target_label     = {int(target_label)}")
		print(f"  reconstructed_y  = {pred_label}")
		print(f"  reconstruction MSE (normalized) = {mse:.6f}")


if __name__ == "__main__":
	main()
