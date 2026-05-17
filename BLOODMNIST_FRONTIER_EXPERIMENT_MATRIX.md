# BloodMNIST Federated Frontier Experiment Matrix

This table defines the next BloodMNIST privacy-utility frontier experiments.
The attacker is fixed from calibration and must not be tuned inside these
experiments:

- `attack_batch_size=1`
- `distance=cossim`
- `attack_lr=0.05`
- `attack_iters=300`
- `num_trials=3`
- clients `0,1,2`
- samples `0,25,50`
- protocol:
  `configs/current/attack_protocols/bloodmnist_fixed_attacker_v1.yaml`

Leakage direction: lower reconstruction MSE means stronger leakage; higher
`leakage_score = -log10(MSE)` means stronger leakage.

Use macro-F1 as the main utility axis. A configuration is frontier-relevant if
it lowers median or worst-case leakage while keeping macro-F1 within an
acceptable drop from its matched no-DP baseline, for example within 5, 10, and
15 percentage points. Report those thresholds explicitly instead of choosing a
single hidden cutoff.

## Frontier Table

| Block | Purpose | Dataset | Split / alpha | Sigma values | Other FL parameters | Seeds | Model runs | Attack cells | Status | Priority |
| --- | --- | --- | --- | --- | --- | --- | ---: | ---: | --- | --- |
| A. Existing sigma frontier | Complete the already configured main DP frontier and identify the first useful sigma range. | BloodMNIST | IID, Dirichlet alpha 1.0, 0.5, 0.1 | 0, 0.25, 0.5, 0.75, 1.0, 2.0 | 5 clients, 20 rounds, local epochs 1, batch 64, clip norm 1.0, SGD lr 0.01 | 42 | 24 | 216 | Training exists; implemented by `scripts/run_bloodmnist_block_a_frontier.py`. Fixed-attacker attacks needed for all sigmas except completed sigma 0.5/no-DP table. | P0 |
| B. Refined low-noise sigma frontier | Resolve the likely useful region between weak privacy and large utility loss. | BloodMNIST | IID, alpha 1.0, alpha 0.5, alpha 0.1 | 0.1, 0.2, 0.3, 0.4, 0.6 | Same as Block A | 42 | 20 | 180 | New configs needed. Run after Block A shows where macro-F1 starts collapsing. | P1 |
| C. Clip norm by sigma | Test whether clipping can improve privacy without requiring as much noise. | BloodMNIST | IID and alpha 0.5 | 0.25, 0.5, 0.75 | clip norm 0.5, 1.0, 2.0; 5 clients, 20 rounds, local epochs 1, batch 64, lr 0.01 | 42 | 18 | 162 | New configs needed. Include clip norm 1.0 rows as comparable anchors if not reusing Block A artifacts. | P1 |
| D. Local epochs by sigma | Measure the privacy and utility effect of more client-side training before aggregation. | BloodMNIST | IID and alpha 0.5 | 0, 0.25, 0.5, 0.75 | local epochs 1, 2, 5; 5 clients, 20 rounds, batch 64, clip norm 1.0, lr 0.01 | 42 | 24 | 216 | New configs needed. Local epochs 1 can reuse Block A where all metadata matches. | P1 |
| E. Client count by sigma | Test whether smaller per-client data partitions and more clients change leakage and utility. | BloodMNIST | IID and alpha 0.5 | 0, 0.25, 0.5, 0.75 | num clients 5, 10, 20; 20 rounds, local epochs 1, batch 64, clip norm 1.0, lr 0.01 | 42 | 24 | 216 | New configs needed for 10 and 20 clients. Confirm every chosen attack client has enough samples. | P2 |
| F. Batch size by sigma | Test the training-batch-size side of the privacy-utility frontier under fixed attack batch size. | BloodMNIST | IID and alpha 0.5 | 0, 0.25, 0.5, 0.75 | training batch size 32, 64, 128; 5 clients, 20 rounds, local epochs 1, clip norm 1.0, lr 0.01 | 42 | 24 | 216 | New configs needed for 32 and 128. Batch 64 can reuse Block A where all metadata matches. | P2 |
| G. Stability seeds | Check whether selected frontier conclusions survive split/training randomness. | BloodMNIST | IID and alpha 0.5 | selected from Blocks A-C, likely 0, 0.25, 0.5, 0.75 | best candidate clip norm/local epoch setting plus baseline anchors | 42, 1337, 2026 | 12 to 24 | 108 to 216 | Run only after Blocks A-C identify candidate frontier points. | P2 |
| H. Validation dataset transfer | Test whether BloodMNIST conclusions generalize beyond one dataset. | PathMNIST, DermaMNIST | IID and alpha 0.1 by default | selected BloodMNIST frontier sigmas, likely 0, 0.25, 0.5 | Match the chosen BloodMNIST FL settings where feasible | 42 | 12 | 108 | Requires generalized fixed-attacker evaluation path for validation datasets if not already wired. | P3 |

If every block is run literally, this is roughly 158 to 170 model trainings and
1,422 to 1,530 fixed-attacker cells. The intended laptop-scale workflow is
sequential: complete P0, inspect the frontier, then narrow P1/P2 to the regions
that still have plausible utility.

## Recommended Execution Order

1. Finish Block A using the fixed attacker and write one complete
   `model_privacy_utility_table.csv`.
2. Plot Block A macro-F1 against median leakage score and median MSE.
3. Run Block B only in the sigma interval where Block A suggests the strongest
   utility/privacy trade-off.
4. Run Blocks C and D for the two most relevant splits: IID and
   `noniid_alpha_05`.
5. Use Blocks E and F as second-pass FL-design evidence if time remains.
6. Run Block G for the most promising frontier points before making thesis
   claims.
7. Run Block H as validation after the BloodMNIST frontier is stable.

## Reporting Table Schema

Every trained model should contribute one row to the frontier table with:

| Column | Meaning |
| --- | --- |
| `experiment_name` | Unique run name encoding dataset, split, defense, sigma, and varied FL parameter. |
| `dataset` | `bloodmnist`, then validation datasets. |
| `split_label` | `iid`, `noniid_alpha_1`, `noniid_alpha_05`, or `noniid_alpha_01`. |
| `alpha` | Dirichlet alpha; blank or 0 for IID. |
| `num_clients` | Total FL clients. |
| `num_rounds` | FedAvg communication rounds. |
| `local_epochs` | Local epochs per round. |
| `batch_size` | Training batch size. |
| `clip_norm` | DP clipping norm. |
| `sigma` | DP noise multiplier. Use 0 for no-DP baseline. |
| `epsilon` | DP accountant epsilon when available. |
| `test_accuracy` | Utility metric. |
| `test_macro_f1` | Main utility metric for frontier selection. |
| `attack_success_rate` | Positive-MSE attack cells divided by intended attack cells. |
| `median_mse` | Median reconstruction MSE across fixed-attacker cells. Lower means more leakage. |
| `median_leakage_score` | Median `-log10(MSE)`. Higher means more leakage. |
| `worst_mse` | Lowest positive MSE across attack cells. |
| `worst_leakage_score` | Highest leakage score across attack cells. |
| `n_failed_or_no_mse` | Failed/no-MSE attack cells retained for transparency. |
