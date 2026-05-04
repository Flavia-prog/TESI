import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from aijack.attack.inversion import GradientInversion_Attack
from torch.utils.data import DataLoader, TensorDataset


DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
DELTA = 1e-5


class SimpleClinicalTextModel(nn.Module):
    def __init__(self, vocab_size: int = 64, num_classes: int = 2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(vocab_size, 32),
            nn.ReLU(),
            nn.Linear(32, num_classes),
        )

    def forward(self, x):
        return self.net(x)


def build_synthetic_clinical_dataset(
    n_samples: int = 2048, seq_len: int = 12, vocab_size: int = 64, seed: int = 0
):
    rng = np.random.default_rng(seed)
    tokens = rng.integers(0, vocab_size, size=(n_samples, seq_len), endpoint=False, dtype=np.int64)
    labels = (tokens.mean(axis=1) > (vocab_size / 2.0)).astype(np.int64)
    x = np.zeros((n_samples, vocab_size), dtype=np.float32)
    for i in range(n_samples):
        counts = np.bincount(tokens[i], minlength=vocab_size).astype(np.float32)
        x[i] = counts / max(float(seq_len), 1.0)
    x = torch.tensor(x, dtype=torch.float32)
    y = torch.tensor(labels, dtype=torch.long)
    return TensorDataset(x, y)


def split_dataset(dataset, train_ratio: float = 0.8, seed: int = 0):
    n = len(dataset)
    n_train = int(n * train_ratio)
    gen = torch.Generator().manual_seed(seed)
    perm = torch.randperm(n, generator=gen)
    train_idx = perm[:n_train]
    test_idx = perm[n_train:]
    x = dataset.tensors[0]
    y = dataset.tensors[1]
    train_ds = TensorDataset(x[train_idx], y[train_idx])
    test_ds = TensorDataset(x[test_idx], y[test_idx])
    return train_ds, test_ds


def iterate_on_device(dataloader: DataLoader, device: torch.device = DEVICE):
    for x, y in dataloader:
        yield x.to(device, non_blocking=True), y.to(device, non_blocking=True)


def compute_dp_gradients(model, x, y, clip_bound, noise_sigma):
    model.train()
    params = [p for p in model.parameters() if p.requires_grad]
    batch_size = x.size(0)
    criterion = nn.CrossEntropyLoss()

    per_sample_sums = [torch.zeros_like(p) for p in params]
    for i in range(batch_size):
        model.zero_grad(set_to_none=True)
        logits_i = model(x[i : i + 1])
        loss_i = criterion(logits_i, y[i : i + 1])
        grads_i = torch.autograd.grad(loss_i, params, retain_graph=False, create_graph=False)
        total_norm = torch.sqrt(sum(torch.sum(g.detach() * g.detach()) for g in grads_i) + 1e-12)
        clip_coef = min(1.0, clip_bound / total_norm.item())
        for j, g in enumerate(grads_i):
            per_sample_sums[j].add_(g.detach() * clip_coef)

    noisy_gradients = []
    noise_std = noise_sigma * clip_bound
    for gsum in per_sample_sums:
        avg_clipped_grad = gsum / float(batch_size)
        noise = torch.normal(
            mean=0.0,
            std=noise_std / float(batch_size),
            size=gsum.shape,
            device=gsum.device,
        )
        noisy_gradients.append((avg_clipped_grad + noise).detach().clone())
    return noisy_gradients


def dp_step(model, x, y, optimizer, clip_bound, noise_sigma):
    params = [p for p in model.parameters() if p.requires_grad]
    grads = compute_dp_gradients(model, x, y, clip_bound, noise_sigma)
    optimizer.zero_grad(set_to_none=True)
    for p, g in zip(params, grads):
        p.grad = g
    optimizer.step()


def evaluate_accuracy(model, dataloader, max_batches=None):
    model.eval()
    total = 0
    correct = 0
    with torch.no_grad():
        for batch_id, (x, y) in enumerate(iterate_on_device(dataloader)):
            pred = model(x).argmax(dim=1)
            correct += (pred == y).sum().item()
            total += y.size(0)
            if max_batches is not None and (batch_id + 1) >= max_batches:
                break
    return correct / total if total > 0 else 0.0


class SignedAdam(torch.optim.Adam):
    def step(self, closure=None):
        loss = closure() if closure is not None else None
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is not None:
                    p.grad.data.copy_(p.grad.data.sign())
        super().step()
        return loss


def run_inversion_attack(model, intercepted_gradients, target_shape):
    x_shape = tuple(target_shape[1:]) if len(target_shape) > 1 else tuple(target_shape)
    attacker = GradientInversion_Attack(
        target_model=model.to(DEVICE),
        x_shape=x_shape,
        optimize_label=False,
        distancename="cossim",
        optimizer_class=SignedAdam,
        num_iteration=1200,
        lr=0.1,
        tv_reg_coef=0.0,
        device=DEVICE,
        log_interval=0,
    )
    try:
        rec_x, _ = attacker.attack([g.to(DEVICE) for g in intercepted_gradients], batch_size=1)
        return rec_x.detach()
    except UnboundLocalError:
        return torch.zeros((1,) + x_shape, dtype=torch.float32, device=DEVICE)


def bag_reconstruction_score(original_x: torch.Tensor, reconstructed_x: torch.Tensor):
    orig = original_x.view(original_x.size(0), -1).float()
    rec = reconstructed_x.view(reconstructed_x.size(0), -1).float()
    orig = orig / (orig.norm(dim=1, keepdim=True) + 1e-8)
    rec = rec / (rec.norm(dim=1, keepdim=True) + 1e-8)
    return float((orig * rec).sum(dim=1).mean().item())


def approximate_epsilon(noise_sigma: float, steps: int, delta: float = DELTA):
    if noise_sigma <= 0.0:
        return float("inf")
    return float((steps * (2.0 * torch.log(torch.tensor(1.25 / delta)).item()) ** 0.5) / noise_sigma)


def run_text_plan(plan_df: pd.DataFrame, output_csv: str = "results/text_pipeline_results.csv"):
    base_ds = build_synthetic_clinical_dataset()
    train_ds, test_ds = split_dataset(base_ds)
    results = []

    for _, row in plan_df.iterrows():
        batch_size = int(row["Batch_Size"])
        sigma = float(row["Noise_Sigma"])
        clip = float(row["Clip_C"])
        lr = float(row["Learning_Rate"])
        train_steps = 10

        model = SimpleClinicalTextModel().to(DEVICE)
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_ds, batch_size=256, shuffle=False)
        optimizer = torch.optim.SGD(model.parameters(), lr=lr)

        it = iterate_on_device(train_loader)
        first_batch = None
        for _ in range(train_steps):
            x_step, y_step = next(it)
            if first_batch is None:
                first_batch = (x_step.detach().clone(), y_step.detach().clone())
            dp_step(model, x_step, y_step, optimizer, clip, sigma)

        x0, y0 = first_batch
        grads = compute_dp_gradients(model, x0, y0, clip, sigma)
        reconstructed = run_inversion_attack(model, grads, target_shape=x0.shape)
        recon_score = bag_reconstruction_score(x0[:1], reconstructed[:1])
        acc = evaluate_accuracy(model, test_loader, max_batches=10)
        eps = approximate_epsilon(sigma, train_steps)

        out = row.to_dict()
        out["Modality"] = "text"
        out["Reconstruction_Score"] = recon_score
        out["Test_Accuracy"] = float(acc)
        out["Train_Steps"] = int(train_steps)
        out["Delta"] = float(DELTA)
        out["Epsilon"] = float(eps)
        results.append(out)

    df = pd.DataFrame(results)
    df.to_csv(output_csv, index=False)
    return df
