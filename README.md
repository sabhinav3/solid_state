# Numerical Solution of Schrödinger Equation for Electrons in Potential Wells

This repository contains Python scripts to numerically solve the Schrödinger equation for electrons in three types of potential wells: **infinite**, **finite**, and **two closely spaced finite wells**. The solutions provide eigenenergies, eigenvectors (wavefunctions), and plots of the potential wells with allowed energy bands.

## Features

1. **Infinite Potential Well**  
   - **Inputs**: Mass of electron, width of the well.  
   - **Outputs**: Eigenenergies, eigen wavefunctions, plot of the well with energy levels.  

2. **Finite Potential Well**  
   - **Inputs**: Mass of electron, width of the well, depth of the well (in eV).  
   - **Outputs**: Eigenenergies, eigen wavefunctions, plot of the well with energy levels.  

3. **Two Closely Spaced Finite Wells**  
   - **Inputs**: Mass of electron, width of each well, depth of the wells (in eV), spacing between wells.  
   - **Outputs**: Eigenenergies, eigen wavefunctions, plot of the coupled wells with energy bands.  

## Example Output
The results mimic the visualization from [falstad.com/qm1d](https://www.falstad.com/qm1d/), showing quantized energy levels and wavefunctions within the potential wells.

## Requirements
- Python 3.x
- NumPy
- SciPy
- Matplotlib

Install dependencies with:  
```bash
pip install numpy scipy matplotlib``

## Usage

### Infinite Potential Well
```python
python infinite_well.py --mass 9.1e-31 --width 1e-9
```

### Finite Potential Well
```python
python finite_well.py --mass 9.1e-31 --width 1e-9 --depth 10
```

### Two Closely Spaced Wells
```python
python double_well.py --mass 9.1e-31 --width 1e-9 --depth 10 --spacing 2e-10
```

## Methodology
1. **Finite Difference Algorithm**: Discretizes the Schrödinger equation into a matrix eigenvalue problem.  
2. **Hamiltonian Matrix**: Constructed using the second-order central difference approximation for the kinetic energy term and the potential energy profile.  
3. **Eigenvalue Solver**: Uses `scipy.linalg.eigh` to compute eigenvalues (energies) and eigenvectors (wavefunctions).  

## Repository Structure
```
├── infinite_well.py      # Solver for infinite well
├── finite_well.py        # Solver for finite well
├── double_well.py        # Solver for two coupled wells
├── README.md
└── requirements.txt
```

## References
- [Finite Difference Method Tutorial](https://www.youtube.com/watch?v=9fGaTU1-f-0)
- [Solving ODEs in Python](https://www.youtube.com/watch?v=gFXvfrAKuSo)
- [Eigenvalues in Python](https://www.youtube.com/watch?v=UDQehOF1IP0)

## Contributing
Contributions are welcome! Open an issue or submit a pull request for improvements or bug fixes.

## License
MIT License. See `LICENSE` for details.
``` 

This README provides a clear overview, setup instructions, usage examples, and links to relevant resources. It is structured to help users quickly understand the project’s purpose and how to use the code.
