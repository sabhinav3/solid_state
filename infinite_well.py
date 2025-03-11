import numpy as np
from scipy.linalg import eigh
import matplotlib.pyplot as plt

def solve_infinite_well(mass_electron, width, N=1000):
    """
    Solves the infinite potential well problem for a given electron mass and well width.
    
    Parameters:
    mass_electron (float): Mass of electron in kg
    width (float): Width of the well in meters
    N (int): Number of grid points (default 1000)
    
    Returns:
    eigen_energies (np.array): Array of eigen energies in eV
    wavefunctions (np.array): Matrix of normalized wavefunctions (columns)
    """
    # Constants
    ħ = 1.0545718e-34  # Reduced Planck's constant (J·s)
    m = mass_electron  # Electron mass in kg
    L = width
    
    # Discretization
    dx = L / (N + 1)  # Grid spacing
    x = np.linspace(dx, L - dx, N)  # Interior points
    
    # Hamiltonian construction
    K = (ħ ** 2) / (2 * m * dx ** 2)
    H = np.zeros((N, N))
    
    # Fill diagonal and off-diagonal elements
    main_diag = 2 * K * np.ones(N)
    off_diag = -K * np.ones(N - 1)
    
    H = np.diag(main_diag) + np.diag(off_diag, k=1) + np.diag(off_diag, k=-1)
    
    # Solve eigenvalue problem
    eigenvalues, eigenvectors = eigh(H)
    
    # Convert energies to eV
    J_to_eV = 1 / 1.602176634e-19
    eigen_energies = eigenvalues * J_to_eV
    
    # Normalize wavefunctions
    wavefunctions = eigenvectors / np.sqrt(dx)
    
    # Plot results
    plt.figure(figsize=(10, 6))
    
    # Plot potential well
    plt.vlines(0, 0, 0.1, color='black', lw=2)
    plt.vlines(L, 0, 0.1, color='black', lw=2)
    plt.hlines(0, 0, L, colors='black', lw=2)
    
    # Plot first 5 energy levels and wavefunctions
    num_levels = min(5, len(eigen_energies))
    for n in range(num_levels):
        energy = eigen_energies[n]
        psi = wavefunctions[:, n]
        
        # Create full wavefunction with boundary zeros
        psi_full = np.zeros(N + 2)
        psi_full[1:-1] = psi
        
        # Scale wavefunction for visualization
        scaled_psi = 0.1 * psi_full / np.max(np.abs(psi_full)) + energy
        
        # Create x array including boundaries
        x_full = np.linspace(0, L, N + 2)
        
        plt.plot(x_full, scaled_psi, label=f'n = {n+1}')
        plt.hlines(energy, 0, L, colors='gray', linestyles='dashed', alpha=0.5)
    
    plt.xlabel('Position (m)')
    plt.ylabel('Energy (eV)')
    plt.title(f'Infinite Potential Well (L = {L:.1e} m)')
    plt.legend()
    plt.show()
    
    return eigen_energies, wavefunctions

# Example usage
if __name__ == "__main__":
    # Electron mass (kg), Well width (m)
    m_e = 9.10938356e-31  # kg
    L = 1e-10  # 1 angstrom
    
    energies, wavefuncs = solve_infinite_well(m_e, L)
    
    # Print first 5 energy levels
    print("First 5 energy levels (eV):")
    for i, E in enumerate(energies[:5]):
        print(f"n = {i+1}: {E:.3f} eV")