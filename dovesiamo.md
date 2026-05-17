1. Attack-Parameter Screening
Goal: understand which attack settings make reconstruction stronger or weaker.

You ran exploratory attack-condition experiments and analyzed them with regression / feature-importance style methods.

Tested attack-side factors included things like:

client id
sample index
attack batch size
attack iterations
number of trials
attack learning rate
distance metric
Output idea:

attack settings -> reconstruction MSE / leakage
Important interpretation:

Regression / feature importance = screening, not causal proof.
Matched contrasts are more defensible than raw importance rankings.
Main use:

Choose a strong fixed attacker for later privacy-utility experiments.
2. Fixed Attacker Chosen
From that calibration, you fixed one attacker protocol:

dataset: BloodMNIST
attack batch size: 1
attack iterations: 300
trials: 3
learning rate: 0.05
distance: cossim
clients: 0, 1, 2
sample indices: 0, 25, 50
device: cpu
Why this matters:

Attacker settings are no longer varied.
They become the evaluation protocol.
This avoids mixing two questions:

Question A: what makes the attack strong?
Question B: which FL defenses reduce leakage?
After calibration, you focus on Question B.

3. Baseline FL Models
You trained BloodMNIST FedAvg models without DP defense.

Splits:

IID
Dirichlet alpha=1.0
Dirichlet alpha=0.5
Dirichlet alpha=0.1
These baselines give the utility and leakage reference point.

Measured utility:

test accuracy
test macro-F1
Measured leakage after attack:

MSE
leakage score = -log10(MSE)
attack success/failure
4. DP Sigma Frontier Models
Then you trained BloodMNIST FedAvg models with DP-style defense.

Main varied variable:

sigma / noise multiplier
Values:

0.25
0.5
0.75
1.0
2.0
For each split, you now have:

baseline sigma=0
DP sigma=0.25
DP sigma=0.5
DP sigma=0.75
DP sigma=1.0
DP sigma=2.0
Total:

4 splits x 6 sigma levels = 24 models
5. Block A Fixed-Attacker Evaluation
You then attacked all 24 models using the same fixed attacker.

Attack grid:

24 models
x 3 clients
x 3 sample indices
= 216 attack cells
Outputs:

attack_cell_summary.csv
model_privacy_utility_table.csv
block_a_model_privacy_utility_table.csv
block_a_report.md
privacy-utility plots
This is the first real privacy-utility frontier result.

6. What Block A Shows
High-level result:

No-DP baselines:
good utility, strong leakage

Low sigma, especially 0.25:
utility already drops, privacy benefit inconsistent

sigma=0.5:
successful attacks usually have much higher MSE, so reconstructions are weaker,
but macro-F1 drops substantially

sigma>=0.75:
many attacks fail/no-MSE, but model utility also becomes poor
Important caveat:

failed/no-MSE is not the same as proven privacy.
It means the attack did not produce a usable reconstruction metric.
Overall Thesis Flow So Far

Regression / screening
        |
        v
Identify strong attacker settings
        |
        v
Fix attacker protocol
        |
        v
Train FL models with different privacy defenses
        |
        v
Attack every model with the same attacker
        |
        v
Compare utility vs leakage
Current One-Sentence Status

You have moved from exploratory attack calibration into the first privacy-utility evaluation: BloodMNIST FedAvg with IID/non-IID splits and DP sigma variation, attacked by a fixed calibrated AIJack gradient inversion protocol.