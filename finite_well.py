import numpy as np
import matplotlib.pyplot as plt
import argparse

def main():
    # command line arguments
    parser = argparse.ArgumentParser(description='Solve finite potential well Schrödinger equation')
    parser.add_argument('--mass', type=float, required=True, help='Mass of electron (kg)')
    parser.add_argument('--width', type=float, required=True, help='Width of the well (m)')
    parser.add_argument('--depth', type=float, required=True, help='Depth of the well (eV)')
    args = parser.parse_args()

    m = args.mass
    L = args.width
    U0_eV = args.depth

    # Physical constants
    hbar = 1.0545718e-34       # Reduced Planck's constant (J·s)
    e_charge = 1.602176634e-19 # Elementary charge (C)
    U0_J = U0_eV * e_charge    # Convert depth to Joules

    # Numerical parameters
    L_sim = 3 * L              # Total simulation width
    N = 1000                   # Number of grid points
    dx = L_sim / (N + 1)
    x = np.linspace(dx, L_sim - dx, N)

    # Defining potential well
    well_center = L_sim / 2
    well_start = well_center - L/2
    well_end = well_center + L/2
    U = np.where((x >= well_start) & (x <= well_end), 0, U0_J)

    # Constructing Hamiltonian matrix
    K = hbar**2 / (2 * m * dx**2)
    main_diag = 2 * K + U
    off_diag = -K * np.ones(N-1)
    H = np.diag(main_diag) + np.diag(off_diag, 1) + np.diag(off_diag, -1)

    # Solving eigenvalue problem
    eigenvalues, eigenvectors = np.linalg.eigh(H)
    energies_eV = eigenvalues / e_charge  # Convert to eV

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

    # Visualization setup
    plt.figure(figsize=(12, 8))
    colors = plt.cm.viridis(np.linspace(0, 1, n_states))
    scaling = 0.15 * U0_eV  # Wavefunction scaling factor

    # Drawing potential well
    plt.plot([0, well_start], [U0_eV, U0_eV], 'k-', lw=3)
    plt.plot([well_start, well_start], [U0_eV, 0], 'k-', lw=3)
    plt.plot([well_start, well_end], [0, 0], 'k-', lw=3)
    plt.plot([well_end, well_end], [0, U0_eV], 'k-', lw=3)
    plt.plot([well_end, L_sim], [U0_eV, U0_eV], 'k-', lw=3, label='Potential')

    # Plotting energy levels and wavefunctions
    for i in range(n_states):
        energy = bound_energies[i]
        psi = bound_wavefuncs[:, i]
        
        # Normalize and shift wavefunction
        psi_normalized = psi / np.max(np.abs(psi)) * scaling
        psi_shifted = psi_normalized + energy
        
        # Plot components
        plt.hlines(energy, 0, L_sim, colors='gray', linestyles='dotted', lw=1)
        plt.plot(x, psi_shifted, color=colors[i], lw=1.5, alpha=0.8, 
                label=f'n = {i+1}' if i < 3 else "")

    plt.xlabel('Position (m)', fontsize=12)
    plt.ylabel('Energy (eV)', fontsize=12)
    plt.title(f'Finite Potential Well: {L:.1e} m width, {U0_eV} eV depth', fontsize=14)
    plt.ylim(-0.1 * U0_eV, 1.2 * U0_eV)
    plt.legend(loc='upper right')
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()