from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from aijack.attack.inversion import GradientInversionAttackServerManager
from aijack.collaborative.fedavg import FedAVGAPI, FedAVGServer
from torch.utils.data import DataLoader, TensorDataset

from src.model import SmallCNN
from src.utils import DEVICE, ensure_dir


def _predict_label(model, image_batch):
    with torch.no_grad():
        logits = model(image_batch.to(DEVICE))
        return int(logits.argmax(dim=1).item())


def _save_reconstruction_grid(rows, output_path: Path):
    tiles = []
    for row in rows:
        tiles.append(np.concatenate([row["original"], row["reconstructed"]], axis=1))
    grid = np.concatenate(tiles, axis=0)
    plt.imsave(output_path, grid, cmap="gray")


def run_gradient_inversion_demo(
    trained_client,
    train_dataset,
    output_dir: str = "results/gradient_inversion",
    num_attacks: int = 10,
    attack_batch_size: int = 1,
    attack_iterations: int = 60,
    sample_seed: int | None = None,
):
    ensure_dir(output_dir)
    out_dir = Path(output_dir)
    records = []
    rows_for_grid = []

    trained_client.eval()
    attack_manager = GradientInversionAttackServerManager(
        x_shape=(1, 28, 28),
        device=DEVICE,
        num_iteration=attack_iterations,
        lr=1.0,
        log_interval=30,
        optimizer_class=torch.optim.LBFGS,
        distancename="l2",
    )
    AttackingServer = attack_manager.attach(FedAVGServer)
    criterion = torch.nn.CrossEntropyLoss()
    rng = np.random.default_rng(sample_seed)

    for attack_id in range(num_attacks):
        sampled_indices = rng.choice(len(train_dataset), size=attack_batch_size, replace=False)
        batch_x = []
        batch_y = []
        for idx in sampled_indices:
            x_i, y_i = train_dataset[idx]
            batch_x.append(x_i)
            batch_y.append(int(y_i))

        x_secret = torch.stack(batch_x, dim=0)
        y_secret = torch.tensor(batch_y, dtype=torch.long)
        secret_dataset = TensorDataset(x_secret, y_secret)
        secret_loader = DataLoader(secret_dataset, batch_size=attack_batch_size, shuffle=False)

        server = AttackingServer([trained_client], SmallCNN().to(DEVICE), device=DEVICE)
        local_optimizer = torch.optim.SGD(trained_client.parameters(), lr=0.1)

        api = FedAVGAPI(
            server=server,
            clients=[trained_client],
            criterion=criterion,
            local_optimizers=[local_optimizer],
            local_dataloaders=[secret_loader],
            num_communication=1,
            local_epoch=1,
            use_gradients=True,
            device=DEVICE,
        )
        api.run()

        recon_x, _ = server.attack_results[-1][0]
        secret_img = x_secret[0].squeeze().detach().cpu().numpy()
        recon_np = recon_x.detach().cpu().numpy()
        if recon_np.ndim == 4:
            candidates = [recon_np[i, 0] for i in range(recon_np.shape[0])]
        elif recon_np.ndim == 3:
            candidates = [recon_np[0]]
        else:
            candidates = [recon_np]

        losses = [float(np.mean((cand - secret_img) ** 2)) for cand in candidates]
        best_idx = int(np.argmin(losses))
        reconstructed = candidates[best_idx]
        mse = losses[best_idx]

        reconstructed_tensor = (
            torch.tensor(reconstructed, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        )
        pred_label = _predict_label(trained_client, reconstructed_tensor)

        original_path = out_dir / f"attack_{attack_id:02d}_original.png"
        reconstructed_path = out_dir / f"attack_{attack_id:02d}_reconstructed.png"
        comparison_path = out_dir / f"attack_{attack_id:02d}_comparison.png"
        plt.imsave(original_path, secret_img, cmap="gray")
        plt.imsave(reconstructed_path, reconstructed, cmap="gray")
        plt.imsave(comparison_path, np.concatenate([secret_img, reconstructed], axis=1), cmap="gray")

        records.append(
            {
                "attack_id": attack_id,
                "sample_index": int(sampled_indices[0]),
                "true_label": int(y_secret[0].item()),
                "predicted_label": pred_label,
                "mse": mse,
                "original_image": str(original_path),
                "reconstructed_image": str(reconstructed_path),
                "comparison_image": str(comparison_path),
            }
        )
        rows_for_grid.append({"original": secret_img, "reconstructed": reconstructed})
        print(f"Attack {attack_id + 1}/{num_attacks} MSE: {mse:.6f}")

    df = pd.DataFrame(records)
    df.to_csv(out_dir / "attack_metrics.csv", index=False)
    _save_reconstruction_grid(rows_for_grid, out_dir / "reconstruction_grid.png")

    avg_mse = float(df["mse"].mean())
    best_mse = float(df["mse"].min())
    worst_mse = float(df["mse"].max())
    return {"avg_mse": avg_mse, "best_mse": best_mse, "worst_mse": worst_mse}
