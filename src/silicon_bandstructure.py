#!/usr/bin/env python3
"""
Silicon Band Structure Calculation

This script calculates the electronic band structure of bulk silicon
along high-symmetry paths in the Brillouin zone and plots it.

Steps:
1. Perform SCF calculation at Gamma point to get self-consistent potential
2. Compute eigenvalues along high-symmetry k-path
3. Plot the band structure
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.constants import HA_TO_EV, BOHR_TO_ANG, ANG_TO_BOHR, TWOPI
from src.lattice import Lattice
from src.crystal import Crystal
from src.gvector import GVector
from src.xc import lda_xc
from src.hartree import compute_hartree_potential
from src.hamiltonian import Hamiltonian
from src.eigensolver import PCGEigensolver, random_initial_guess
from src.mixing import LinearMixer
from src.ewald import Ewald


def create_silicon_crystal():
    """Create diamond silicon crystal structure."""
    a_ang = 5.43  # Angstrom
    a_bohr = a_ang * ANG_TO_BOHR
    
    lattice = Lattice.fcc(a_bohr)
    crystal = Crystal(lattice)
    crystal.add_atom('Si', [0.0, 0.0, 0.0], zion=4)
    crystal.add_atom('Si', [0.25, 0.25, 0.25], zion=4)
    
    return crystal, a_bohr


def create_silicon_potential(lattice, fft_shape, v_depth=1.0, r_c=1.5):
    """Create a simplified silicon-like Gaussian potential."""
    n1, n2, n3 = fft_shape
    v_ext = np.zeros(fft_shape, dtype=float)
    
    x = np.linspace(0, 1, n1, endpoint=False)
    y = np.linspace(0, 1, n2, endpoint=False)
    z = np.linspace(0, 1, n3, endpoint=False)
    
    atoms = [np.array([0.0, 0.0, 0.0]), np.array([0.25, 0.25, 0.25])]
    a, b, c = lattice.vectors
    
    for i, xi in enumerate(x):
        for j, yj in enumerate(y):
            for k, zk in enumerate(z):
                grid_frac = np.array([xi, yj, zk])
                for atom_frac in atoms:
                    d_frac = grid_frac - atom_frac
                    d_frac = d_frac - np.round(d_frac)
                    d_cart = d_frac[0]*a + d_frac[1]*b + d_frac[2]*c
                    r2 = np.sum(d_cart**2)
                    v_ext[i, j, k] += -v_depth * np.exp(-r2 / r_c**2)
    
    return v_ext


def get_high_symmetry_points_fcc():
    """
    High-symmetry points for FCC Brillouin zone.
    
    Returns fractional coordinates in the reciprocal lattice
    of the FCC primitive cell.
    """
    # For FCC primitive cell, the BZ is a truncated octahedron
    # Standard points (in units of 2pi/a for conventional cell):
    points = {
        'G': np.array([0.0, 0.0, 0.0]),       # Gamma
        'X': np.array([0.5, 0.0, 0.5]),       # X point
        'W': np.array([0.5, 0.25, 0.75]),     # W point
        'K': np.array([0.375, 0.375, 0.75]),  # K point
        'L': np.array([0.5, 0.5, 0.5]),       # L point
        'U': np.array([0.625, 0.25, 0.625]),  # U point
    }
    return points


def generate_kpath(lattice, path_labels, npts_per_segment=20):
    """
    Generate k-points along a path through high-symmetry points.
    
    Args:
        lattice: Lattice object
        path_labels: List of high-symmetry point labels, e.g., ['G', 'X', 'W', 'L', 'G', 'K']
        npts_per_segment: Number of k-points per segment
    
    Returns:
        k_frac: K-points in fractional coordinates (nk, 3)
        k_cart: K-points in Cartesian coordinates (nk, 3)  
        k_dist: Cumulative distance along path (for plotting x-axis)
        tick_positions: Positions of high-symmetry points on x-axis
        tick_labels: Labels for high-symmetry points
    """
    hs_points = get_high_symmetry_points_fcc()
    b = lattice.reciprocal_vectors
    
    k_frac = []
    k_cart = []
    k_dist = []
    tick_positions = []
    tick_labels = []
    
    cumulative_dist = 0.0
    
    for i, label in enumerate(path_labels):
        if label not in hs_points:
            raise ValueError(f"Unknown high-symmetry point: {label}")
        
        kf = hs_points[label]
        
        if i == 0:
            # First point
            k_frac.append(kf)
            kc = kf[0]*b[0] + kf[1]*b[1] + kf[2]*b[2]
            k_cart.append(kc)
            k_dist.append(0.0)
            tick_positions.append(0.0)
            tick_labels.append('$\\Gamma$' if label == 'G' else f'${label}$')
        else:
            # Generate points from previous to current
            kf_prev = hs_points[path_labels[i-1]]
            
            for j in range(1, npts_per_segment + 1):
                t = j / npts_per_segment
                kf_interp = (1 - t) * kf_prev + t * kf
                k_frac.append(kf_interp)
                
                kc = kf_interp[0]*b[0] + kf_interp[1]*b[1] + kf_interp[2]*b[2]
                k_cart.append(kc)
                
                # Distance increment
                dk_frac = kf_interp - k_frac[-2] if len(k_frac) > 1 else np.zeros(3)
                if len(k_frac) > 1:
                    prev_kc = k_cart[-2]
                    dk_cart = kc - prev_kc
                    cumulative_dist += np.linalg.norm(dk_cart)
                k_dist.append(cumulative_dist)
            
            tick_positions.append(cumulative_dist)
            tick_labels.append('$\\Gamma$' if label == 'G' else f'${label}$')
    
    return (np.array(k_frac), np.array(k_cart), np.array(k_dist), 
            tick_positions, tick_labels)


def compute_eigenvalues_at_k(k_cart, gvec, hamiltonian, n_bands, kg_base):
    """
    Compute eigenvalues at a specific k-point.
    
    Args:
        k_cart: K-point in Cartesian coordinates
        gvec: GVector object
        hamiltonian: Hamiltonian object with potential set
        n_bands: Number of bands
        kg_base: |G|^2 values (kinetic energy without k)
    
    Returns:
        eigenvalues: Array of eigenvalues
    """
    npw = gvec.npw
    
    # Compute |k+G|^2 for this k-point
    kg = np.zeros(npw)
    for i in range(npw):
        kpg = k_cart + gvec.cart[i]
        kg[i] = np.sum(kpg**2)
    
    # Create temporary Hamiltonian with updated kinetic energy
    # We need to update the diagonal (kinetic part)
    hamiltonian.kg = kg
    
    # Initialize eigenvectors
    evecs = random_initial_guess(npw, n_bands)
    evals = np.zeros(n_bands)
    
    # Solve eigenvalue problem
    eigensolver = PCGEigensolver(npw, n_bands)
    eigensolver.solve(
        ham_apply=hamiltonian.apply,
        ham_diag=hamiltonian.get_diagonal(),
        evecs=evecs,
        evals=evals,
        tol=1e-6,
        max_iter=100
    )
    
    return evals


class SiliconBandStructure:
    """Calculate and plot silicon band structure."""
    
    def __init__(self, ecut=5.0, n_bands=8):
        print("=" * 60)
        print("Silicon Band Structure Calculation")
        print("=" * 60)
        
        # Create crystal
        self.crystal, self.a = create_silicon_crystal()
        self.lattice = self.crystal.lattice
        self.volume = self.lattice.volume
        
        print(f"\nCrystal: Diamond Silicon")
        print(f"Lattice constant: {self.a:.4f} Bohr ({self.a * BOHR_TO_ANG:.4f} Ang)")
        
        self.n_electrons = 8
        self.ecut = ecut
        self.n_bands = n_bands
        
        # G-vectors
        self.gvec = GVector(self.lattice, ecut)
        self.npw = self.gvec.npw
        self.fft_shape = self.gvec.get_fft_grid_size()
        self.n_fft = np.prod(self.fft_shape)
        
        print(f"Energy cutoff: {ecut:.1f} Ha ({ecut * HA_TO_EV:.1f} eV)")
        print(f"Plane waves: {self.npw}")
        print(f"Bands: {n_bands}")
        
        # External potential
        self.v_ext = create_silicon_potential(self.lattice, self.fft_shape)
        
        # Hamiltonian
        self.hamiltonian = Hamiltonian(self.gvec, self.volume)
        
        # Storage
        self.rho_r = None
        self.rho_g = None
        self.v_local_r = None
    
    def run_scf(self, max_iter=50, tol=1e-4):
        """Run SCF to get self-consistent potential."""
        print("\n" + "-" * 60)
        print("Step 1: SCF calculation at Gamma point")
        print("-" * 60)
        
        # Initialize density
        rho_0 = self.n_electrons / self.volume
        self.rho_r = np.full(self.fft_shape, rho_0, dtype=float)
        self.rho_g = self._r_to_g(self.rho_r)
        
        # Initialize eigenvectors
        evecs = random_initial_guess(self.npw, self.n_bands)
        evals = np.zeros(self.n_bands)
        
        # Occupations
        occupations = np.array([2.0, 2.0, 2.0, 2.0, 0.0, 0.0, 0.0, 0.0])
        
        # Mixer
        mixer = LinearMixer(alpha=0.3)
        
        # Eigensolver
        eigensolver = PCGEigensolver(self.npw, self.n_bands)
        
        energy_old = 0.0
        energy_history = []
        
        print(f"{'Iter':>4} {'dE (Ha)':>12}")
        print("-" * 20)
        
        for scf_iter in range(1, max_iter + 1):
            # Build potential
            v_hartree_g = compute_hartree_potential(self.rho_g, self.gvec.norms)
            v_hartree_fft = self.gvec.map_to_fft_grid(v_hartree_g, self.fft_shape)
            v_hartree_r = np.real(np.fft.ifftn(v_hartree_fft) * self.n_fft)
            
            rho_real = np.maximum(np.real(self.rho_r), 1e-20)
            v_xc_r, _ = lda_xc(rho_real)
            
            self.v_local_r = v_hartree_r + v_xc_r + self.v_ext
            self.hamiltonian.set_local_potential(self.v_local_r)
            
            # Solve eigenvalue problem
            eigensolver.solve(
                ham_apply=self.hamiltonian.apply,
                ham_diag=self.hamiltonian.get_diagonal(),
                evecs=evecs,
                evals=evals,
                tol=1e-6,
                max_iter=50
            )
            
            # Compute new density
            rho_r_new = np.zeros(self.fft_shape, dtype=float)
            from src.hamiltonian import g_to_r
            for i in range(self.n_bands):
                if occupations[i] < 1e-10:
                    continue
                psi_r = g_to_r(evecs[:, i], self.gvec, self.fft_shape, self.volume)
                rho_r_new += occupations[i] * np.abs(psi_r)**2
            
            rho_g_new = self._r_to_g(rho_r_new)
            
            # Energy (band energy)
            energy = np.sum(occupations * evals)
            energy_history.append(energy)
            
            de = abs(energy - energy_old)
            print(f"{scf_iter:4d} {de:12.2e}")
            
            if de < tol:
                print("-" * 20)
                print(f"SCF converged in {scf_iter} iterations")
                break
            
            # Check oscillation
            if scf_iter > 10 and len(energy_history) >= 6:
                recent = energy_history[-6:]
                odd_std = np.std([recent[0], recent[2], recent[4]])
                even_std = np.std([recent[1], recent[3], recent[5]])
                if odd_std < tol and even_std < tol:
                    print("-" * 20)
                    print(f"SCF oscillating - converged after {scf_iter} iterations")
                    break
            
            # Mix
            rho_g_mixed = mixer.mix(self.rho_g, rho_g_new)
            self.rho_g = rho_g_mixed
            self.rho_r = self._g_to_r(self.rho_g)
            
            energy_old = energy
        
        print(f"Gamma point eigenvalues: {evals[:4] * HA_TO_EV} eV (valence)")
        return evals
    
    def compute_bands(self, path_labels=['L', 'G', 'X', 'W', 'K', 'G'], npts=15):
        """Compute band structure along k-path."""
        print("\n" + "-" * 60)
        print("Step 2: Computing band structure")
        print("-" * 60)
        
        # Generate k-path
        k_frac, k_cart, k_dist, ticks, labels = generate_kpath(
            self.lattice, path_labels, npts
        )
        nk = len(k_frac)
        
        print(f"K-path: {' -> '.join(path_labels)}")
        print(f"Total k-points: {nk}")
        
        # Make sure potential is set
        if self.v_local_r is None:
            print("Running SCF first...")
            self.run_scf()
        
        self.hamiltonian.set_local_potential(self.v_local_r)
        
        # Compute eigenvalues at each k-point
        bands = np.zeros((nk, self.n_bands))
        
        print("Computing eigenvalues...")
        for ik in range(nk):
            if ik % 10 == 0:
                print(f"  k-point {ik+1}/{nk}")
            
            k = k_cart[ik]
            
            # Update kinetic energy for this k-point
            kg = np.zeros(self.npw)
            for ig in range(self.npw):
                kpg = k + self.gvec.cart[ig]
                kg[ig] = np.sum(kpg**2)
            
            self.hamiltonian.kg = kg
            
            # Solve
            evecs = random_initial_guess(self.npw, self.n_bands)
            evals = np.zeros(self.n_bands)
            
            eigensolver = PCGEigensolver(self.npw, self.n_bands)
            eigensolver.solve(
                ham_apply=self.hamiltonian.apply,
                ham_diag=self.hamiltonian.get_diagonal(),
                evecs=evecs,
                evals=evals,
                tol=1e-5,
                max_iter=50
            )
            
            # Sort eigenvalues (they should already be sorted, but ensure it)
            bands[ik] = np.sort(evals)
        
        print("Done!")
        
        return k_dist, bands, ticks, labels
    
    def plot_bands(self, k_dist, bands, ticks, labels, filename='silicon_bands.png'):
        """Plot the band structure."""
        print("\n" + "-" * 60)
        print("Step 3: Plotting band structure")
        print("-" * 60)
        
        # Convert to eV
        bands_ev = bands * HA_TO_EV
        
        # Find Fermi level (top of valence band)
        # With 8 electrons and 2 per band, bands 0-3 are occupied
        e_fermi = np.max(bands_ev[:, 3])
        
        # Shift so Fermi level is at 0
        bands_ev_shifted = bands_ev - e_fermi
        
        # Create figure
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Plot each band
        for i in range(self.n_bands):
            color = 'blue' if i < 4 else 'red'  # Valence vs conduction
            ax.plot(k_dist, bands_ev_shifted[:, i], color=color, linewidth=1.5)
        
        # Add vertical lines at high-symmetry points
        for tick in ticks:
            ax.axvline(x=tick, color='gray', linestyle='--', linewidth=0.5)
        
        # Add horizontal line at Fermi level
        ax.axhline(y=0, color='green', linestyle='--', linewidth=1, label='$E_F$')
        
        # Labels and formatting
        ax.set_xlabel('Wave Vector', fontsize=12)
        ax.set_ylabel('Energy (eV)', fontsize=12)
        ax.set_title('Silicon Band Structure (Educational Model)', fontsize=14)
        ax.set_xlim(k_dist[0], k_dist[-1])
        ax.set_ylim(-15, 10)
        
        # Set x-tick labels
        ax.set_xticks(ticks)
        ax.set_xticklabels(labels, fontsize=11)
        
        # Legend
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], color='blue', label='Valence bands'),
            Line2D([0], [0], color='red', label='Conduction bands'),
            Line2D([0], [0], color='green', linestyle='--', label='Fermi level')
        ]
        ax.legend(handles=legend_elements, loc='upper right')
        
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(filename, dpi=150)
        print(f"Band structure saved to: {filename}")
        
        # Also show the plot
        plt.show()
        
        # Print band gap
        vbm = np.max(bands_ev[:, 3])
        cbm = np.min(bands_ev[:, 4])
        gap = cbm - vbm
        print(f"\nBand gap: {gap:.3f} eV")
        print(f"VBM: {vbm:.3f} eV")
        print(f"CBM: {cbm:.3f} eV")
        
        return fig, ax
    
    def _r_to_g(self, arr_r):
        arr_fft = np.fft.fftn(arr_r) / self.n_fft
        return self.gvec.map_from_fft_grid(arr_fft)
    
    def _g_to_r(self, arr_g):
        arr_fft = self.gvec.map_to_fft_grid(arr_g, self.fft_shape)
        return np.real(np.fft.ifftn(arr_fft) * self.n_fft)


def main():
    """Main function to calculate and plot Si band structure."""
    print("\n" + "#" * 60)
    print("# Silicon Electronic Band Structure")
    print("#" * 60)
    
    # Create calculator
    si = SiliconBandStructure(ecut=5.0, n_bands=8)
    
    # Run SCF
    si.run_scf(max_iter=40, tol=1e-3)
    
    # Compute bands along path L-G-X-W-K-G with dense k-points
    k_dist, bands, ticks, labels = si.compute_bands(
        path_labels=['L', 'G', 'X', 'W', 'K', 'G'],
        npts=30  # More dense k-points for smoother bands
    )
    
    # Plot
    output_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 
                               'silicon_bands.png')
    si.plot_bands(k_dist, bands, ticks, labels, filename=output_path)
    
    print("\n" + "#" * 60)
    print("# Calculation Complete!")
    print("#" * 60 + "\n")


if __name__ == "__main__":
    main()
