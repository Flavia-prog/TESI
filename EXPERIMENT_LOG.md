# Experiment Log

## Utility experiments

| Name | Config | Result folder | Accuracy | Macro-F1 | Notes |
|---|---|---|---:|---:|---|
| IID baseline | configs/iid_baseline.yaml | results/iid_baseline/ | 0.6659 | 0.5787 | Working FedAvg baseline |
| Non-IID alpha 1.0 | configs/noniid_alpha_1.yaml | results/noniid_alpha_1/ | 0.6343 | 0.5260 | Mild non-IID |
| Non-IID alpha 0.5 | configs/noniid_alpha_05.yaml | results/noniid_alpha_05/ | 0.6352 | 0.5509 | Moderate non-IID |
| Non-IID alpha 0.1 | configs/noniid_alpha_01.yaml | results/noniid_alpha_01/ | 0.4826 | 0.3736 | Strong non-IID; utility drops |

## Attack experiments

| Name | Base model | Batch size | Attack settings | Output folder | MSE | Notes |
|---|---|---:|---|---|---:|---|
| IID batch 1 strong | iid_baseline | 1 | cossim, 1000 iters, 5 trials, lr 0.1 | results/iid_baseline/attacks/batch1_cossim_1000iters_5trials_lr01_client0_sample0 | 0.0099 | Successful reconstruction |
| IID batch 2 strong | iid_baseline | 2 | cossim, 1000 iters, 5 trials, lr 0.1 | results/iid_baseline/attacks/batch2_cossim_1000iters_5trials_lr01_client0_sample0 | N/A | Batched reconstruction; MSE not directly comparable |
| IID batch 8 strong | iid_baseline | 8 | cossim, 1000 iters, 5 trials, lr 0.1 | results/iid_baseline/attacks/batch8_cossim_1000iters_5trials_lr01_client0_sample0 | N/A | Qualitative only for now |
| IID batch 64 strong | iid_baseline | 64 | cossim, 1000 iters, 5 trials, lr 0.1 | results/iid_baseline/attacks/batch64_cossim_1000iters_5trials_lr01_client0_sample0 | N/A | More realistic setting; qualitative only for now |