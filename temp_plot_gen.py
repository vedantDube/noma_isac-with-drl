import scipy.io
import matplotlib
matplotlib.use('Agg') # Set non-interactive backend
import matplotlib.pyplot as plt
import os
import numpy as np

def generate_plot():
    M_values = [50, 100, 200, 300, 400]
    crbs = {
        'td3': [],
        'ppo': [],
        'ddpg': []
    }
    
    for M in M_values:
        fpath = f'./results/ris_M{M}/results.mat'
        if os.path.exists(fpath):
            try:
                data = scipy.io.loadmat(fpath)
                # Handle potential scalar nesting or weirdness in .mat files
                crbs['td3'].append(float(np.atleast_1d(data['CRB_td3']).flatten()[0]))
                crbs['ppo'].append(float(np.atleast_1d(data['CRB_ppo']).flatten()[0]))
                crbs['ddpg'].append(float(np.atleast_1d(data['CRB_ddpg']).flatten()[0]))
            except Exception as e:
                print(f"Error reading {fpath}: {e}")
                for key in crbs: crbs[key].append(np.nan)
        else:
            print(f"File not found: {fpath}")
            for key in crbs: crbs[key].append(np.nan)

    # Set up the plot aesthetics
    plt.rcParams.update({
        "text.usetex": False,
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica", "Arial"],
        "axes.labelsize": 12,
        "axes.titlesize": 14,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "figure.titlesize": 16,
    })

    plt.figure(figsize=(10, 7), dpi=150)
    plt.style.use('bmh') # Use a clean, professional style

    plt.plot(M_values, crbs['td3'], marker='o', linewidth=3, markersize=10, label='TD3', color='#0072BD')
    plt.plot(M_values, crbs['ppo'], marker='s', linewidth=3, markersize=10, label='PPO', color='#D95319')
    plt.plot(M_values, crbs['ddpg'], marker='^', linewidth=3, markersize=10, label='DDPG', color='#77AC30')

    plt.xlabel('Number of RIS Elements (M)', fontweight='bold')
    plt.ylabel('Cramer-Rao Bound (CRB)', fontweight='bold')
    plt.title('Best CRB Trace vs RIS Element Count', fontweight='bold', pad=20)
    plt.grid(True, alpha=0.3)
    plt.legend(frameon=True, fancybox=True, shadow=True)
    plt.xticks(M_values)
    
    # Save to workspace and to a temporary location for embedding
    output_name = 'crb_vs_ris_premium.png'
    plt.savefig(output_name, bbox_inches='tight')
    plt.close()
    
    print(f"Successfully generated {output_name}")

if __name__ == "__main__":
    generate_plot()
