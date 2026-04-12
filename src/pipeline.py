from src.common import *


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@dataclass
class ExperimentConfig:
    dataset_name: str = "MNIST"
    model_name: str = "LeNet"
    activation: str = "sigmoid"

    num_clients: int = 1
    clients_per_round: int = 1
    num_rounds: int = 1
    local_epochs: int = 1
    batch_size: int = 1
    client_lr: float = 0.1

    use_dp: bool = False
    dp_sigma: float = 0.0
    dp_clip: float = 1.0

    attack_iterations: int = 100
    attack_lr: float = 1.0
    attack_distance: str = "l2"

    secret_index: int = 7
    seed: int = 0
    noniid_alpha: Optional[float] = None


class LeNet(nn.Module):
    def __init__(self, in_channels=1, num_classes=10, activation_cls=nn.Sigmoid):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(in_channels, 12, kernel_size=5, padding=2, stride=2),
            activation_cls(),
            nn.Conv2d(12, 12, kernel_size=5, padding=2, stride=2),
            activation_cls(),
        )
        self.fc = nn.Linear(12 * 7 * 7, num_classes)

    def forward(self, x):
        x = self.body(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)


def get_model(cfg: ExperimentConfig):
    activation_cls = nn.Sigmoid if cfg.activation.lower() == "sigmoid" else nn.ReLU

    if cfg.dataset_name == "MNIST":
        return LeNet(in_channels=1, num_classes=10, activation_cls=activation_cls).to(DEVICE)

    raise ValueError(f"Unsupported dataset: {cfg.dataset_name}")


def get_mnist():
    transform = transforms.Compose([transforms.ToTensor()])
    train_dataset = torchvision.datasets.MNIST(
        root="./data", train=True, download=True, transform=transform
    )
    test_dataset = torchvision.datasets.MNIST(
        root="./data", train=False, download=True, transform=transform
    )
    return train_dataset, test_dataset


def iid_partition(dataset, num_clients, seed=0):
    rng = np.random.default_rng(seed)
    indices = np.arange(len(dataset))
    rng.shuffle(indices)
    splits = np.array_split(indices, num_clients)
    return [Subset(dataset, split.tolist()) for split in splits]


def build_batch_with_secret(dataset, secret_index, batch_size, seed):
    rng = np.random.default_rng(seed + 1000 * secret_index + 10000 * batch_size)

    if batch_size == 1:
        batch_indices = [secret_index]
    else:
        batch_indices = [secret_index]
        while len(batch_indices) < batch_size:
            idx = int(rng.integers(0, len(dataset)))
            if idx != secret_index and idx not in batch_indices:
                batch_indices.append(idx)
        rng.shuffle(batch_indices)

    secret_pos = batch_indices.index(secret_index)
    x_list, y_list = [], []
    for idx in batch_indices:
        x_i, y_i = dataset[idx]
        x_list.append(x_i)
        y_list.append(int(y_i))

    x = torch.stack(x_list, dim=0)
    y = torch.tensor(y_list, dtype=torch.long)
    return x, y, secret_pos


@torch.no_grad()
def evaluate_model(model, dataloader, criterion=None):
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for x, y in dataloader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        logits = model(x)

        if criterion is not None:
            total_loss += criterion(logits, y).item() * x.size(0)

        preds = torch.argmax(logits, dim=1)
        total_correct += (preds == y).sum().item()
        total_samples += x.size(0)

    avg_loss = total_loss / total_samples if criterion is not None else None
    acc = total_correct / total_samples
    return avg_loss, acc


def make_client_and_optimizer(base_model, cfg: ExperimentConfig, local_dataset):
    if not cfg.use_dp:
        client = FedAVGClient(
            copy.deepcopy(base_model), user_id=0, lr=cfg.client_lr, device=DEVICE
        )
        optimizer = optim.SGD(client.parameters(), lr=cfg.client_lr)
        return client, optimizer

    local_size = len(local_dataset)
    lot_size = min(cfg.batch_size, local_size)
    batch_size = min(cfg.batch_size, lot_size)

    accountant = GeneralMomentAccountant(noise_type="Gaussian", backend="python")
    dp_manager = DPSGDManager(
        accountant=accountant,
        optimizer_cls=optim.SGD,
        l2_norm_clip=cfg.dp_clip,
        dataset=local_dataset,
        lot_size=lot_size,
        batch_size=batch_size,
        iterations=max(cfg.local_epochs, 1),
    )
    DPFedAVGClient, DPOptimizer = DPSGDClientManager(dp_manager, sigma=cfg.dp_sigma).attach(
        FedAVGClient
    )

    client = DPFedAVGClient(
        copy.deepcopy(base_model), user_id=0, lr=cfg.client_lr, device=DEVICE
    )

    for p in client.parameters():
        if p.requires_grad and p.grad is None:
            p.grad = torch.zeros_like(p)

    optimizer = DPOptimizer(client.parameters(), lr=cfg.client_lr)
    return client, optimizer


def reconstruction_metrics(secret_img, reconstructed_img):
    mse_score = float(np.mean((reconstructed_img - secret_img) ** 2))
    ssim_score = float(ssim(secret_img, reconstructed_img, data_range=1.0))
    return mse_score, ssim_score


def run_single_client_attack_trial(cfg: ExperimentConfig, train_dataset, test_loader):
    set_seed(cfg.seed)

    base_model = get_model(cfg)
    criterion = nn.CrossEntropyLoss()

    batch_x_cpu, batch_y_cpu, secret_pos = build_batch_with_secret(
        train_dataset, cfg.secret_index, cfg.batch_size, cfg.seed
    )
    local_dataset = TensorDataset(batch_x_cpu, batch_y_cpu)
    local_loader = DataLoader(local_dataset, batch_size=cfg.batch_size, shuffle=False)

    client, local_optimizer = make_client_and_optimizer(base_model, cfg, local_dataset)

    attack_manager = GradientInversionAttackServerManager(
        x_shape=(1, 28, 28),
        device=DEVICE,
        num_iteration=cfg.attack_iterations,
        lr=cfg.attack_lr,
        log_interval=20,
        optimizer_class=torch.optim.LBFGS,
        distancename=cfg.attack_distance,
    )
    AttackingServer = attack_manager.attach(FedAVGServer)
    server = AttackingServer([client], copy.deepcopy(base_model), device=DEVICE)

    api = FedAVGAPI(
        server=server,
        clients=[client],
        criterion=criterion,
        local_optimizers=[local_optimizer],
        local_dataloaders=[local_loader],
        num_communication=cfg.num_rounds,
        local_epoch=cfg.local_epochs,
        use_gradients=True,
        device=DEVICE,
    )

    api.run()

    test_loss, test_acc = evaluate_model(client, test_loader, criterion)

    recon_x, recon_y = server.attack_results[-1][0]
    secret_img = batch_x_cpu[secret_pos].detach().cpu().numpy().squeeze()

    recon_cpu = recon_x.detach().cpu()
    if recon_cpu.ndim == 4:
        candidates = [recon_cpu[i].numpy().squeeze() for i in range(recon_cpu.shape[0])]
    else:
        candidates = [recon_cpu.numpy().squeeze()]

    mses = [np.mean((cand - secret_img) ** 2) for cand in candidates]
    best_idx = int(np.argmin(mses))
    best_recon = candidates[best_idx]

    mse_score, ssim_score = reconstruction_metrics(secret_img, best_recon)

    return {
        **asdict(cfg),
        "secret_pos": int(secret_pos),
        "test_loss": float(test_loss) if test_loss is not None else None,
        "test_acc": float(test_acc),
        "attack_mse": float(mse_score),
        "attack_ssim": float(ssim_score),
    }


def run_grid_experiments(train_dataset, test_loader, batch_sizes, sigmas, seeds):
    results = []

    for batch_size, sigma, seed in product(batch_sizes, sigmas, seeds):
        cfg = ExperimentConfig(
            dataset_name="MNIST",
            activation="sigmoid",
            batch_size=batch_size,
            num_rounds=1,
            local_epochs=1,
            use_dp=(sigma > 0),
            dp_sigma=sigma,
            dp_clip=1.0,
            seed=seed,
            secret_index=7,
        )
        out = run_single_client_attack_trial(cfg, train_dataset, test_loader)
        results.append(out)

    return pd.DataFrame(results)