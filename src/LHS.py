import numpy as np
import pandas as pd
from scipy.stats import qmc

def generate_experiment_plan(num_samples=50, modalities=("image", "text")):
    print(f"Generating Latin Hypercube Sample with {num_samples} variations...")
    
    # 1. Initialize Latin Hypercube Sampler for 4 variables
    sampler = qmc.LatinHypercube(d=4)
    sample = sampler.random(n=num_samples)

    # 2. Define the exact boundaries optimized for your M3 MacBook
    # Parameter Order: [Batch_Size, Noise_Sigma, Clip_C, Learning_Rate]
    l_bounds = [1, 0.0, 0.1, 0.0001]
    u_bounds = [16, 2.0, 2.0, 0.001]

    # 3. Scale the [0, 1] samples to your specific boundaries
    scaled_samples = qmc.scale(sample, l_bounds, u_bounds)

    # 4. Format into a pandas DataFrame
    df = pd.DataFrame(scaled_samples, columns=['Batch_Size', 'Noise_Sigma', 'Clip_C', 'Learning_Rate'])
    
    # 5. Fix Batch Size (Must be an integer between 1 and 16)
    df['Batch_Size'] = np.round(df['Batch_Size']).astype(int)

    # 6. Save to CSV in the current working directory
    if modalities:
        blocks = []
        for m in modalities:
            tmp = df.copy()
            tmp["Modality"] = m
            blocks.append(tmp)
        df = pd.concat(blocks, ignore_index=True)

    filename = "experiment_plan.csv"
    df.to_csv(filename, index=False)
    
    print(f"Success! Saved to {filename}.")
    print("\nHere is a preview of your first 5 experimental setups:")
    print(df.head())

if __name__ == "__main__":
    generate_experiment_plan()
