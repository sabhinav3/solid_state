import numpy as np
import matplotlib.pyplot as plt
import argparse

def main():
    # command line arguments
    parser = argparse.ArgumentParser(description='Solve two coupled finite potential wells')
    parser.add_argument('--mass', type=float, required=True, help='Electron mass (kg)')
    parser.add_argument('--width', type=float, required=True, help='Width of each well (m)')
    parser.add_argument('--depth', type=float, required=True, help='Well depth (eV)')
    parser.add_argument('--spacing', type=float, required=True, help='Well spacing (m)')
    args = parser.parse_args()

    m = args.mass
    w = args.width
    U0_eV = args.depth
    s = args.spacing

    # Physical constants
    hbar = 1.0545718e-34       # Reduced Planck's constant (J·s)
    e_charge = 1.602176634e-19 # Elementary charge (C)
    U0 = U0_eV * e_charge      # Convert depth to Joules

    # Simulation parameters
    total_width = 2*w + s
    L_sim = 4 * total_width    # Total simulation domain
    N = 2000                   # Number of grid points
    dx = L_sim / N
    x = np.linspace(0, L_sim, N)

    # Creating double well potential
    center = L_sim/2
    well1_start = center - (total_width/2)
    well1_end = well1_start + w
    well2_start = well1_end + s
    well2_end = well2_start + w

    U = np.zeros_like(x)
    U[(x < well1_start) | (x > well2_end)] = U0
    U[(x > well1_end) & (x < well2_start)] = U0

    # Constructing Hamiltonian matrix
    K = hbar**2 / (2 * m * dx**2)
    main_diag = 2*K + U
    off_diag = -K * np.ones(N-1)
    H = np.diag(main_diag) + np.diag(off_diag, 1) + np.diag(off_diag, -1)

    # Solving eigenvalue problem
    eigenvalues, eigenvectors = np.linalg.eigh(H)
    energies_eV = eigenvalues / e_charge

    # Finding bound states (E < U0_eV)
    bound_mask = energies_eV < U0_eV
    bound_energies = energies_eV[bound_mask]
    bound_wavefuncs = eigenvectors[:, bound_mask]
    n_states = len(bound_energies)

    if n_states == 0:
        print("No bound states found!")
        return

    print(f"Found {n_states} bound state(s):")
    for i, energy in enumerate(bound_energies):
        print(f"E_{i+1}: {energy:.4f} eV")

    # Visualization
    plt.figure(figsize=(14, 8))
    colors = plt.cm.viridis(np.linspace(0, 1, n_states))
    scaling = 0.15 * U0_eV  # Wavefunction scaling factor

    # Plotting potential
    plt.plot(x, U/e_charge, 'k-', lw=3, label='Potential')

    # Plotting energy levels and wavefunctions
    for i in range(n_states):
        energy = bound_energies[i]
        psi = bound_wavefuncs[:, i]
        
        # Normalize and shift wavefunction
        psi_normalized = psi / np.max(np.abs(psi)) * scaling
        psi_shifted = psi_normalized + energy
        
        # Plot components
        plt.hlines(energy, x[0], x[-1], colors='gray', linestyles='dotted', lw=1)
        plt.plot(x, psi_shifted, color=colors[i], lw=1.5, alpha=0.8, 
                label=f'State {i+1}' if i < 3 else "")

    # Highlighting the well regions
    plt.axvspan(well1_start, well1_end, facecolor='0.9', alpha=0.3)
    plt.axvspan(well2_start, well2_end, facecolor='0.9', alpha=0.3)

    plt.xlabel('Position (m)', fontsize=12)
    plt.ylabel('Energy (eV)', fontsize=12)
    plt.title(f'Coupled Wells: {w:.1e}m width, {s:.1e}m spacing, {U0_eV}eV depth', fontsize=14)
    plt.ylim(-0.1*U0_eV, 1.2*U0_eV)
    plt.legend(loc='upper right')
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()