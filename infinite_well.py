import numpy as np
import matplotlib.pyplot as plt
import argparse

def main():
    # Command line arguments
    parser = argparse.ArgumentParser(description='Infinite Potential Well Schrödinger Equation')
    parser.add_argument('--mass', type=float, required=True, help='Electron mass (kg)')
    parser.add_argument('--width', type=float, required=True, help='Well width (meters)')
    args = parser.parse_args()

    m = args.mass
    L = args.width

    # Physical constants
    hbar = 1.0545718e-34    # Reduced Planck's constant (J·s)
    e_charge = 1.602176634e-19  # Elementary charge (C)

    # Numerical parameters
    N = 1000                # Number of grid points
    dx = L / (N + 1)        # Spatial step size
    x = np.linspace(dx, L - dx, N)  # Spatial grid

    # Construct Hamiltonian matrix
    K = hbar**2 / (2 * m * dx**2)
    main_diag = 2 * K * np.ones(N)
    off_diag = -K * np.ones(N-1)
    H = np.diag(main_diag) + np.diag(off_diag, 1) + np.diag(off_diag, -1)

    # Solve eigenvalue problem
    eigenvalues, eigenvectors = np.linalg.eigh(H)
    energies_eV = eigenvalues / e_charge  # Convert to eV

    # Print first 9 energy levels
    print("First 9 Energy Levels (eV):")
    for n in range(9):
        print(f"State {n+1}: {energies_eV[n]:.4f} eV")

    # Visualization parameters
    n_states = 9
    colors = plt.cm.viridis(np.linspace(0, 1, n_states))
    scaling_factor = 0.15 * L  # Wavefunction scaling for visualization

    # Creating figure
    plt.figure(figsize=(12, 8))

    # Drawing potential well
    plt.plot([0, L], [0, 0], 'k-', lw=3, label='Infinite Well')

    # Plotting energy levels and wavefunctions
    for n in range(n_states):
        energy = energies_eV[n]
        psi = eigenvectors[:, n]
        
        # Normalize and scale wavefunction
        psi_normalized = psi / np.max(np.abs(psi)) * scaling_factor
        psi_shifted = psi_normalized + energy
        
        # Plot energy level
        plt.hlines(energy, 0, L, colors='gray', linestyles='dotted', lw=1)
        
        # Plot wavefunction
        plt.plot(x, psi_shifted, 
                color=colors[n], 
                lw=1.5, 
                alpha=0.8,
                label=f'n = {n+1}' if n < 3 else "")

    # Format plot
    plt.title(f'Infinite Potential Well - First {n_states} Quantum States', fontsize=14)
    plt.xlabel('Position (m)', fontsize=12)
    plt.ylabel('Energy (eV)', fontsize=12)
    plt.ylim(-0.1 * energies_eV[0], energies_eV[n_states-1] * 1.2)
    plt.legend(loc='upper right')
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()