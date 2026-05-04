import torch
import torch.nn as nn
import pandas as pd
from aijack.attack.inversion import GradientInversion_Attack
from medmnist import BloodMNIST
from torchmetrics.image import StructuralSimilarityIndexMeasure
from torch.utils.data import DataLoader
from torchvision import transforms


DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
DELTA = 1e-5


class SimpleBloodCNN(nn.Module):
    """Lightweight CNN for BloodMNIST (28x28 RGB images, 8 classes).
    
    Architecture:
    - Input: 28x28x3 (RGB)
    - Conv1: 3 -> 16 channels, 3x3 kernel, ReLU, MaxPool 2x2 -> 14x14x16
    - Conv2: 16 -> 32 channels, 3x3 kernel, ReLU, MaxPool 2x2 -> 7x7x32
    - Conv3: 32 -> 64 channels, 3x3 kernel, ReLU, MaxPool 2x2 -> 3x3x64
    - Flatten: 576 features
    - FC: 576 -> 8 classes
    """
    
    def __init__(self):
        super(SimpleBloodCNN, self).__init__()
        
        # Convolutional blocks
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU(inplace=True)
        self.pool1 = nn.AvgPool2d(kernel_size=2, stride=2)
        
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU(inplace=True)
        self.pool2 = nn.AvgPool2d(kernel_size=2, stride=2)
        
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.relu3 = nn.ReLU(inplace=True)
        self.pool3 = nn.AvgPool2d(kernel_size=2, stride=2)
        
        # Fully connected layer (576 = 64 * 3 * 3)
        self.fc = nn.Linear(64 * 3 * 3, 8)
    
    def forward(self, x):
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = self.pool3(self.relu3(self.conv3(x)))
        x = x.view(x.size(0), -1)  # Flatten
        x = self.fc(x)
        return x


def get_bloodmnist_dataloader(batch_size: int, split: str = "train", num_workers: int = 0) -> DataLoader:
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    )

    dataset = BloodMNIST(split=split, download=True, transform=transform)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(split == "train"),
        num_workers=num_workers,
    )


def iterate_on_device(dataloader: DataLoader, device: torch.device = DEVICE):
    for x, y in dataloader:
        x = x.to(device, non_blocking=True)
        y = y.squeeze().long().to(device, non_blocking=True)
        yield x, y


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
        grads_i = torch.autograd.grad(
            loss_i,
            params,
            retain_graph=False,
            create_graph=False,
            allow_unused=False,
        )

        total_norm = torch.sqrt(
            sum(torch.sum(g.detach() * g.detach()) for g in grads_i) + 1e-12
        )
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
    dp_grads = compute_dp_gradients(
        model=model,
        x=x,
        y=y,
        clip_bound=clip_bound,
        noise_sigma=noise_sigma,
    )
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer.zero_grad(set_to_none=True)
    for p, g in zip(params, dp_grads):
        p.grad = g
    optimizer.step()


def evaluate_accuracy(model, dataloader, device=DEVICE, max_batches=None):
    model.eval()
    total = 0
    correct = 0
    with torch.no_grad():
        for batch_id, (x, y) in enumerate(iterate_on_device(dataloader, device=device)):
            logits = model(x)
            pred = logits.argmax(dim=1)
            correct += (pred == y).sum().item()
            total += y.size(0)
            if max_batches is not None and (batch_id + 1) >= max_batches:
                break
    return correct / total if total > 0 else 0.0


def approximate_epsilon(noise_sigma: float, steps: int, delta: float = DELTA) -> float:
    # Conservative closed-form upper bound for Gaussian mechanism under simple composition.
    if noise_sigma <= 0.0:
        return float("inf")
    return float((steps * (2.0 * torch.log(torch.tensor(1.25 / delta)).item()) ** 0.5) / noise_sigma)


class SignedAdam(torch.optim.Adam):
    def step(self, closure=None):
        loss = closure() if closure is not None else None
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is not None:
                    p.grad.data.copy_(p.grad.data.sign())
        super().step()
        return loss


def run_inversion_attack(model, intercepted_gradients, target_shape, device):
    model = model.to(device)
    received_gradients = [torch.nan_to_num(g.detach().to(device)) for g in intercepted_gradients]
    x_shape = tuple(target_shape[1:]) if len(target_shape) == 4 else tuple(target_shape)

    attacker = GradientInversion_Attack(
        target_model=model,
        x_shape=x_shape,
        optimize_label=False,
        distancename="cossim",
        optimizer_class=SignedAdam,
        tv_reg_coef=1e-4,
        num_iteration=4800,
        device=device,
        log_interval=0,
        lr=0.1,
    )

    try:
        reconstructed_x, _ = attacker.attack(received_gradients, batch_size=1)
        return reconstructed_x.detach()
    except UnboundLocalError:
        # AIJack can raise this when optimization never sets a valid best iteration.
        return torch.zeros((1,) + x_shape, device=device, dtype=torch.float32)


def _minmax_to_unit(t: torch.Tensor) -> torch.Tensor:
    t_min = t.amin(dim=(1, 2, 3), keepdim=True)
    t_max = t.amax(dim=(1, 2, 3), keepdim=True)
    return (t - t_min) / (t_max - t_min + 1e-8)


def run_image_plan(plan_df: pd.DataFrame, output_csv: str = "pipeline_a_results.csv"):
    ssim_metric = StructuralSimilarityIndexMeasure(data_range=1.0).to(DEVICE)
    results = []

    for _, row in plan_df.iterrows():
        model = SimpleBloodCNN().to(DEVICE)
        batch_size = int(row["Batch_Size"])
        noise_sigma = float(row["Noise_Sigma"])
        clip_bound = float(row["Clip_C"])
        learning_rate = float(row["Learning_Rate"])
        train_steps = 10

        train_loader = get_bloodmnist_dataloader(batch_size, split="train")
        test_loader = get_bloodmnist_dataloader(256, split="test")
        optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

        train_iter = iterate_on_device(train_loader, device=DEVICE)
        first_batch = None
        for _ in range(train_steps):
            x_step, y_step = next(train_iter)
            if first_batch is None:
                first_batch = (x_step.detach().clone(), y_step.detach().clone())
            dp_step(
                model=model,
                x=x_step,
                y=y_step,
                optimizer=optimizer,
                clip_bound=clip_bound,
                noise_sigma=noise_sigma,
            )

        x, y = first_batch

        intercepted_gradients = compute_dp_gradients(
            model=model,
            x=x,
            y=y,
            clip_bound=clip_bound,
            noise_sigma=noise_sigma,
        )

        reconstructed_x = run_inversion_attack(
            model=model,
            intercepted_gradients=intercepted_gradients,
            target_shape=x.shape,
            device=DEVICE,
        )

        x_eval = _minmax_to_unit(x[:1].detach())
        reconstructed_eval = _minmax_to_unit(reconstructed_x[:1].detach())
        ssim_score = ssim_metric(x_eval, reconstructed_eval).item()
        test_accuracy = evaluate_accuracy(model, test_loader, device=DEVICE, max_batches=10)
        epsilon = approximate_epsilon(noise_sigma=noise_sigma, steps=train_steps, delta=DELTA)

        result_row = row.to_dict()
        result_row["Modality"] = "image"
        result_row["SSIM"] = float(ssim_score)
        result_row["Reconstruction_Score"] = float(ssim_score)
        result_row["Test_Accuracy"] = float(test_accuracy)
        result_row["Train_Steps"] = int(train_steps)
        result_row["Delta"] = float(DELTA)
        result_row["Epsilon"] = float(epsilon)
        results.append(result_row)

    out_df = pd.DataFrame(results)
    out_df.to_csv(output_csv, index=False)
    return out_df


def main():
    plan_df = pd.read_csv("experiment_plan.csv")
    run_image_plan(plan_df, output_csv="pipeline_a_results.csv")


if __name__ == "__main__":
    main()
