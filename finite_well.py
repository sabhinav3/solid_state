import numpy as np
from scipy.linalg import eigh
import matplotlib.pyplot as plt

def solve_finite_well(mass_electron, width, depth_eV, N=1000, L_extended=2e-10):
    """
    Solves the finite potential well problem.
    
    Parameters:
    mass_electron (float): Electron mass in kg
    width (float): Width of the well in meters
    depth_eV (float): Depth of the well in eV
    N (int): Number of grid points
    L_extended (float): Extension beyond well boundaries
    
    Returns:
    eigen_energies (np.array): Bound state energies in eV
    wavefunctions (np.array): Corresponding wavefunctions
    """
    
    # Constants and unit conversions
    ħ = 1.0545718e-34  # J·s
    eV_to_J = 1.602176634e-19
    V0 = depth_eV * eV_to_J  # Convert depth to Joules
    
    # Spatial grid setup
    L = width
    x_min = -L_extended
    x_max = L + L_extended
    x = np.linspace(x_min, x_max, N)
    dx = x[1] - x[0]
    
    # Potential function
    U = np.zeros_like(x)
    inside_well = (x >= 0) & (x <= L)
    U[inside_well] = -V0  # Well depth
    
    # Hamiltonian construction
    K = ħ**2 / (2 * mass_electron * dx**2)
    H = np.zeros((N, N))
    
    # Main diagonal
    np.fill_diagonal(H, 2*K + U)
    
    # Off-diagonals
    off_diag = -K * np.ones(N-1)
    H += np.diag(off_diag, k=1) + np.diag(off_diag, k=-1)
    
    # Solve eigenvalue problem
    eigenvalues, eigenvectors = eigh(H)
    
    # Convert energies to eV and filter bound states
    eigen_energies_J = eigenvalues
    eigen_energies_eV = eigen_energies_J / eV_to_J
    bound_states = eigen_energies_J < 0  # Only keep states with E < 0 (bound)
    
    eigen_energies = eigen_energies_eV[bound_states]
    wavefunctions = eigenvectors[:, bound_states]
    
    # Normalize wavefunctions
    wavefunctions /= np.sqrt(dx)
    
    # Plotting
    plt.figure(figsize=(12, 8))
    
    # Plot potential
    plt.plot(x, U/eV_to_J, 'k-', lw=2, label='Potential')
    
    # Plot energy levels and wavefunctions
    num_levels = min(3, len(eigen_energies))  # Plot first 3 states
    colors = ['r', 'b', 'g']
    
    for i in range(num_levels):
        energy = eigen_energies[i]
        psi = wavefunctions[:, i]
        
        # Scale wavefunction for visibility
        scaled_psi = 0.2 * psi / np.max(np.abs(psi)) + energy
        
        plt.plot(x, scaled_psi, color=colors[i], 
                label=f'n = {i+1}, E = {energy:.3f} eV')
        plt.hlines(energy, x_min, x_max, colors=colors[i], 
                 linestyles='dashed', alpha=0.5)
    
    plt.xlabel('Position (m)')
    plt.ylabel('Energy (eV)')
    plt.title(f'Finite Potential Well (Depth = {depth_eV} eV, Width = {width:.1e} m)')
    plt.ylim(-depth_eV*1.1, 0.5*depth_eV)
    plt.legend()
    plt.grid(True)
    plt.show()
    
    return eigen_energies, wavefunctions

# Example usage
if __name__ == "__main__":
    # Input parameters (modify these)
    m_e = 9.10938356e-31  # Electron mass (kg)
    width = 1e-10  # Well width (m)
    depth = 5.0  # Well depth (eV)
    
    # Solve and plot
    energies, wavefuncs = solve_finite_well(m_e, width, depth)
    
    # Print results
    print(f"Found {len(energies)} bound states:")
    for i, E in enumerate(energies):
        print(f"State {i+1}: {E:.3f} eV")