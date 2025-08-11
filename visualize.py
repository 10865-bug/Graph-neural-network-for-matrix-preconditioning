import matplotlib.pyplot as plt
import os
import numpy as np
plt.rcParams['text.usetex'] = True

def plot_loss(train_losses, val_losses=None, save_dir='./figures'):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    epochs = range(1, len(train_losses) + 1)
    
    plt.figure(figsize=(10, 5))
    plt.title(r"Loss over Epochs")
    plt.plot(epochs, train_losses, 'b', label=r'Train Loss')
    plt.plot(epochs, val_losses, 'r', label='Validation Loss')

    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)

    plt.savefig(os.path.join(save_dir, "loss_curves.png"))
    plt.close()

def show_progress(epoch, total_epochs):
    print(f'\rProgress: {epoch}/{total_epochs}', end='')

def plot_spectral_radius(spectral_data, save_dir='./figures'):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    epochs = range(1, len(spectral_data['max']) + 1)

    plt.figure(figsize=(10, 5))
    plt.title(r"Eigenvalues of $I - A^{-1}A$ over Epochs")
    plt.plot(epochs, spectral_data['max'], 'g', label=r'Max Eigenvalue of $I - A^{-1}A$')
    plt.plot(epochs, spectral_data['min'], 'b', label=r'Min Eigenvalue of $I - A^{-1}A$')
    plt.xlabel("Epochs")
    plt.ylabel("Eigenvalue")
    plt.legend()
    plt.grid(True)

    plt.savefig(os.path.join(save_dir, "eigenvalues_I_minus_A_inv_A_curve.png"))
    plt.close()
    
def plot_condition_numbers(original, preconditioned, save_dir):
    original = np.asarray(original)
    preconditioned = np.asarray(preconditioned)
    
    plt.figure(figsize=(10, 8))
    plt.scatter(original, preconditioned, alpha=0.6, s=50, color='#1f77b4', label='Data Points')
    
    min_val = max(1e-100, min(original.min(), preconditioned.min()))
    max_val = min(1e+500, max(original.max(), preconditioned.max()))
    
    ref_line = np.logspace(np.log10(min_val), np.log10(max_val), 100)
    plt.plot(ref_line, ref_line, 'r--', linewidth=1, label='y=x')
    
    plt.xscale('log')
    plt.yscale('log')
    plt.xlim(min_val/10, max_val*10)
    plt.ylim(min_val/10, max_val*10) 
    plt.xlabel('Original Condition Number (log scale)')
    plt.ylabel('Preconditioned Condition Number (log scale)')
    plt.legend()
    
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(os.path.join(save_dir, 'condition_number_comparison.png'), 
                bbox_inches='tight', 
                dpi=300)
    plt.close()