#!/usr/bin/env python3
"""
Silicon DFT Calculation with Real UPF Pseudopotential.

This script performs a plane-wave DFT calculation for bulk silicon
using a norm-conserving pseudopotential from a UPF file.

Supports both Gamma-only and k-point sampled SCF calculations.
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.constants import HA_TO_EV, BOHR_TO_ANG, ANG_TO_BOHR, TWOPI
from src.lattice import Lattice
from src.gvector import GVector
from src.xc import lda_xc
from src.hartree import compute_hartree_potential, compute_hartree_energy
from src.hamiltonian import Hamiltonian, g_to_r
from src.eigensolver import PCGEigensolver, random_initial_guess
from src.mixing import LinearMixer
from src.ewald import Ewald
from src.upf_reader import UPFPseudopotential
from src.nonlocal_potential import NonlocalPotential
from src.kpoint_scf import KPointSCF


def find_upf_file():
    """Find the Si UPF file in the project."""
    possible_paths = [
        os.path.join(os.path.dirname(__file__), '..', '..', 'test_example', 
                     'si-oncv', 'scf', 'pot', 'Si-sr.upf'),
        os.path.join(os.path.dirname(__file__), '..', '..', 'atompsp', 
                     'src', 'Si-sr.upf'),
    ]
    
    for path in possible_paths:
        if os.path.exists(path):
            return os.path.abspath(path)
    
    raise FileNotFoundError("Could not find Si-sr.upf pseudopotential file")


class SiliconUPF:
    """
    Silicon DFT calculation with UPF pseudopotential.
    
    Includes:
    - Local potential from UPF
    - Non-local potential (Kleinman-Bylander)
    """
    
    def __init__(self, ecut=10.0, n_bands=8):
        print("=" * 70)
        print("Silicon DFT Calculation with UPF Pseudopotential")
        print("=" * 70)
        
        # Load pseudopotential
        upf_path = find_upf_file()
        self.psp = UPFPseudopotential(upf_path)
        self.psp.display()
        
        # Create crystal
        a_ang = 5.43  # Angstrom
        self.a = a_ang * ANG_TO_BOHR
        
        self.lattice = Lattice.fcc(self.a)
        self.volume = self.lattice.volume
        
        # Atom positions (fractional, in FCC primitive cell)
        self.atom_positions = np.array([
            [0.0, 0.0, 0.0],
            [0.25, 0.25, 0.25]
        ])
        self.natoms = 2
        
        print(f"\nCrystal: Diamond Silicon")
        print(f"Lattice constant: {self.a:.4f} Bohr ({self.a * BOHR_TO_ANG:.4f} Ang)")
        print(f"Primitive cell volume: {self.volume:.4f} Bohr^3")
        
        self.n_electrons = int(self.psp.zion * self.natoms)  # 4 * 2 = 8
        self.ecut = ecut
        self.n_bands = n_bands
        
        # G-vectors
        self.gvec = GVector(self.lattice, ecut)
        self.npw = self.gvec.npw
        self.fft_shape = self.gvec.get_fft_grid_size()
        self.n_fft = np.prod(self.fft_shape)
        
        print(f"\nEnergy cutoff: {ecut:.1f} Ha ({ecut * HA_TO_EV:.1f} eV)")
        print(f"Plane waves: {self.npw}")
        print(f"FFT grid: {self.fft_shape}")
        print(f"Bands: {n_bands}")
        print(f"Electrons: {self.n_electrons}")
        
        # Set up non-local potential handler
        print("\nSetting up pseudopotential...")
        self.nlpot = NonlocalPotential(
            psp=self.psp,
            atom_positions=self.atom_positions,
            gvec=self.gvec,
            volume=self.volume
        )
        self.vloc_g_atomic = self.nlpot.get_vloc_g()
        print(f"  V_loc(G=0) contribution: {np.real(self.vloc_g_atomic[0]):.6f} Ha")
        print(f"  {self.nlpot.nbeta} non-local projectors")
        
        # Ewald energy
        charges = np.array([self.psp.zion, self.psp.zion])
        self.ewald = Ewald(self.lattice, self.atom_positions, charges, self.gvec)
        print(f"\nEwald energy: {self.ewald.energy:.6f} Ha")
        
        # Hamiltonian (will add non-local later)
        self.hamiltonian = Hamiltonian(self.gvec, self.volume)
        
        # Storage
        self.rho_r = None
        self.rho_g = None
        self.v_local_r = None
        self.evecs = None
        self.evals = None
    
    def _apply_vnl(self, psi_g, beta_kg=None):
        """
        Apply non-local potential to wavefunction.
        
        Uses the NonlocalPotential class for the calculation.
        """
        return self.nlpot.apply_vnl(psi_g, beta_kg)
    
    def run_scf(self, max_iter=60, tol=1e-6, kmesh=None):
        """
        Run SCF calculation.
        
        Args:
            max_iter: Maximum SCF iterations
            tol: Energy convergence tolerance
            kmesh: K-point mesh as tuple (nk1, nk2, nk3). 
                   If None, uses Gamma-only calculation.
                   
        Returns:
            Total energy in Hartree
        """
        if kmesh is not None:
            return self._run_scf_kpoints(max_iter, tol, kmesh)
        else:
            return self._run_scf_gamma(max_iter, tol)
    
    def _run_scf_kpoints(self, max_iter, tol, kmesh):
        """Run SCF with k-point sampling using modular KPointSCF."""
        print("\n" + "-" * 70)
        print("SCF Calculation with K-point Sampling")
        print("-" * 70)
        
        # Create KPointSCF solver
        self.kscf = KPointSCF(
            lattice=self.lattice,
            gvec=self.gvec,
            volume=self.volume,
            n_electrons=self.n_electrons,
            n_bands=self.n_bands,
            vloc_g=self.vloc_g_atomic,
            nlpot=self.nlpot,
            ewald_energy=self.ewald.energy
        )
        
        # Set up k-points
        self.kscf.setup_kpoints(kmesh[0], kmesh[1], kmesh[2], use_symmetry=True)
        
        # Run SCF
        e_total = self.kscf.run(max_iter=max_iter, tol=tol, mixing_alpha=0.2)
        
        # Store converged potential for band structure
        self.v_local_r = self.kscf.get_converged_potential()
        self.rho_r = self.kscf.rho_r
        self.rho_g = self.kscf.rho_g
        
        # Store eigenvalues from Gamma point (or first k-point) for compatibility
        self.evals = self.kscf.kpoints[0].evals
        self.evecs = self.kscf.kpoints[0].evecs
        
        # Print results
        self.kscf.print_results()
        
        return e_total
    
    def _run_scf_gamma(self, max_iter, tol):
        """Run Gamma-only SCF calculation (original implementation)."""
        print("\n" + "-" * 70)
        print("SCF Calculation (Gamma-only)")
        print("-" * 70)
        
        # Initialize density (uniform)
        rho_0 = self.n_electrons / self.volume
        self.rho_r = np.full(self.fft_shape, rho_0, dtype=float)
        self.rho_g = self._r_to_g(self.rho_r)
        
        # Initialize eigenvectors
        self.evecs = random_initial_guess(self.npw, self.n_bands)
        self.evals = np.zeros(self.n_bands)
        
        # Occupations (4 bands fully occupied)
        occupations = np.zeros(self.n_bands)
        n_occ = self.n_electrons // 2
        occupations[:n_occ] = 2.0
        
        # Mixer - use linear mixing for stability
        mixer = LinearMixer(alpha=0.2)
        
        energy_old = 0.0
        energy_history = []
        
        print(f"\n{'Iter':>4} {'E_band (Ha)':>14} {'dE (Ha)':>12}")
        print("-" * 35)
        
        for scf_iter in range(1, max_iter + 1):
            # Build local potential
            self._build_potential()
            
            # Create custom Hamiltonian that includes non-local
            def ham_apply_with_vnl(psi):
                # Local part (kinetic + local potential)
                hpsi = self.hamiltonian.apply(psi)
                # Add non-local
                hpsi += self._apply_vnl(psi)
                return hpsi
            
            # Solve eigenvalue problem
            eigensolver = PCGEigensolver(self.npw, self.n_bands)
            eigensolver.solve(
                ham_apply=ham_apply_with_vnl,
                ham_diag=self.hamiltonian.get_diagonal(),
                evecs=self.evecs,
                evals=self.evals,
                tol=1e-7,
                max_iter=100
            )
            
            # Compute new density
            rho_r_new = np.zeros(self.fft_shape, dtype=float)
            for i in range(self.n_bands):
                if occupations[i] < 1e-10:
                    continue
                psi_r = g_to_r(self.evecs[:, i], self.gvec, self.fft_shape, self.volume)
                rho_r_new += occupations[i] * np.abs(psi_r)**2
            
            rho_g_new = self._r_to_g(rho_r_new)
            
            # Band energy
            e_band = np.sum(occupations * self.evals)
            energy_history.append(e_band)
            
            de = abs(e_band - energy_old)
            print(f"{scf_iter:4d} {e_band:14.8f} {de:12.2e}")
            
            if de < tol and scf_iter > 3:
                print("-" * 35)
                print(f"SCF converged in {scf_iter} iterations")
                break
            
            # Check oscillation
            if scf_iter > 10 and len(energy_history) >= 6:
                recent = energy_history[-6:]
                if np.std(recent) < tol * 10:
                    print("-" * 35)
                    print(f"SCF converged (oscillation) in {scf_iter} iterations")
                    break
            
            # Mix densities
            rho_g_mixed = mixer.mix(self.rho_g, rho_g_new)
            self.rho_g = rho_g_mixed
            self.rho_r = self._g_to_r(self.rho_g)
            
            energy_old = e_band
        
        # Compute total energy
        e_total = self._compute_total_energy(occupations)
        
        self._print_results(e_total, occupations)
        
        return e_total
    
    def _build_potential(self):
        """Build effective local potential."""
        # Hartree potential
        v_hartree_g = compute_hartree_potential(self.rho_g, self.gvec.norms)
        v_hartree_fft = self.gvec.map_to_fft_grid(v_hartree_g, self.fft_shape)
        v_hartree_r = np.real(np.fft.ifftn(v_hartree_fft) * self.n_fft)
        
        # XC potential
        rho_real = np.maximum(np.real(self.rho_r), 1e-20)
        v_xc_r, self._exc_r = lda_xc(rho_real)
        
        # Local pseudopotential in real space
        vloc_fft = self.gvec.map_to_fft_grid(self.vloc_g_atomic, self.fft_shape)
        vloc_r = np.real(np.fft.ifftn(vloc_fft) * self.n_fft)
        
        # Total local potential
        self.v_local_r = v_hartree_r + v_xc_r + vloc_r
        
        self._v_hartree_r = v_hartree_r
        self._v_xc_r = v_xc_r
        
        self.hamiltonian.set_local_potential(self.v_local_r)
    
    def _compute_total_energy(self, occupations):
        """Compute total energy."""
        e_band = np.sum(occupations * self.evals)
        
        e_hartree = compute_hartree_energy(self.rho_g, self.gvec.norms, self.volume)
        
        rho_real = np.maximum(np.real(self.rho_r), 1e-20)
        from src.xc import compute_xc_energy, compute_xc_potential_energy
        e_xc = compute_xc_energy(rho_real, self._exc_r, self.volume, self.n_fft)
        e_vxc = compute_xc_potential_energy(rho_real, self._v_xc_r, self.volume, self.n_fft)
        
        e_total = e_band - e_hartree + e_xc - e_vxc + self.ewald.energy
        
        return e_total
    
    def _print_results(self, e_total, occupations):
        """Print final results."""
        print("\n" + "=" * 70)
        print("RESULTS")
        print("=" * 70)
        
        print(f"\nTotal energy: {e_total:.8f} Ha ({e_total * HA_TO_EV:.6f} eV)")
        print(f"Ewald energy: {self.ewald.energy:.8f} Ha")
        
        print(f"\nEigenvalues (Gamma point):")
        print(f"{'Band':>6} {'E (Ha)':>14} {'E (eV)':>14} {'Occ':>8}")
        print("-" * 45)
        for i in range(self.n_bands):
            print(f"{i+1:6d} {self.evals[i]:14.6f} {self.evals[i]*HA_TO_EV:14.6f} "
                  f"{occupations[i]:8.2f}")
        
        n_occ = int(np.sum(occupations) / 2)
        if n_occ < self.n_bands:
            vbm = self.evals[n_occ - 1]
            cbm = self.evals[n_occ]
            gap = (cbm - vbm) * HA_TO_EV
            print(f"\nDirect gap at Gamma: {gap:.4f} eV")
        
        print("=" * 70)
    
    def compute_bands(self, path_labels=['L', 'G', 'X', 'W', 'K', 'G'], npts=40):
        """Compute band structure along k-path."""
        print("\n" + "-" * 70)
        print("Band Structure Calculation")
        print("-" * 70)
        
        # Make sure SCF is done
        if self.v_local_r is None:
            print("Running SCF first...")
            self.run_scf()
        
        # High-symmetry points for FCC
        hs_points = {
            'G': np.array([0.0, 0.0, 0.0]),
            'X': np.array([0.5, 0.0, 0.5]),
            'W': np.array([0.5, 0.25, 0.75]),
            'K': np.array([0.375, 0.375, 0.75]),
            'L': np.array([0.5, 0.5, 0.5]),
            'U': np.array([0.625, 0.25, 0.625]),
        }
        
        # Generate k-path
        b = self.lattice.reciprocal_vectors
        k_frac, k_cart, k_dist = [], [], []
        ticks, labels = [], []
        
        cumulative_dist = 0.0
        
        for i, label in enumerate(path_labels):
            kf = hs_points[label]
            
            if i == 0:
                k_frac.append(kf)
                kc = kf[0]*b[0] + kf[1]*b[1] + kf[2]*b[2]
                k_cart.append(kc)
                k_dist.append(0.0)
                ticks.append(0.0)
                labels.append('$\\Gamma$' if label == 'G' else f'${label}$')
            else:
                kf_prev = hs_points[path_labels[i-1]]
                for j in range(1, npts + 1):
                    t = j / npts
                    kf_interp = (1 - t) * kf_prev + t * kf
                    k_frac.append(kf_interp)
                    kc = kf_interp[0]*b[0] + kf_interp[1]*b[1] + kf_interp[2]*b[2]
                    k_cart.append(kc)
                    
                    if len(k_cart) > 1:
                        dk = np.linalg.norm(k_cart[-1] - k_cart[-2])
                        cumulative_dist += dk
                    k_dist.append(cumulative_dist)
                
                ticks.append(cumulative_dist)
                labels.append('$\\Gamma$' if label == 'G' else f'${label}$')
        
        k_cart = np.array(k_cart)
        k_dist = np.array(k_dist)
        nk = len(k_cart)
        
        print(f"K-path: {' -> '.join(path_labels)}")
        print(f"Total k-points: {nk}")
        
        # Compute eigenvalues
        bands = np.zeros((nk, self.n_bands))
        
        print("Computing eigenvalues...")
        for ik in range(nk):
            if ik % 20 == 0:
                print(f"  k-point {ik+1}/{nk}")
            
            k = k_cart[ik]
            
            # Update |k+G|^2 for this k-point
            kg = np.zeros(self.npw)
            for ig in range(self.npw):
                kpg = k + self.gvec.cart[ig]
                kg[ig] = np.sum(kpg**2)
            
            self.hamiltonian.kg = kg
            
            # Recompute beta(|k+G|) for this k-point using NonlocalPotential
            beta_kg_k = self.nlpot.get_beta_kg(k)
            
            # Hamiltonian with k-dependent non-local
            def ham_apply_k(psi, beta_kg=beta_kg_k):
                hpsi = self.hamiltonian.apply(psi)
                hpsi += self._apply_vnl(psi, beta_kg)
                return hpsi
            
            # Solve with tighter convergence for smooth bands
            evecs = random_initial_guess(self.npw, self.n_bands)
            evals = np.zeros(self.n_bands)
            
            eigensolver = PCGEigensolver(self.npw, self.n_bands)
            eigensolver.solve(
                ham_apply=ham_apply_k,
                ham_diag=self.hamiltonian.get_diagonal(),
                evecs=evecs,
                evals=evals,
                tol=1e-8,
                max_iter=200
            )
            
            bands[ik] = np.sort(evals)
        
        print("Done!")
        
        return k_dist, bands, ticks, labels
    
    def plot_bands(self, k_dist, bands, ticks, labels, filename='silicon_upf_bands.png'):
        """Plot band structure with smooth interpolation."""
        from scipy.interpolate import make_interp_spline
        
        bands_ev = bands * HA_TO_EV
        
        n_occ = self.n_electrons // 2
        # Use a more robust Fermi level: shift valence band maximum to 0
        # VBM is the highest energy of the highest occupied band (index n_occ-1)
        vbm = np.max(bands_ev[:, n_occ - 1])
        cbm = np.min(bands_ev[:, n_occ])
        
        # Shift so VBM is at 0
        e_ref = vbm
        bands_ev_shifted = bands_ev - e_ref
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Create smooth interpolated k-path for plotting
        k_smooth = np.linspace(k_dist[0], k_dist[-1], 500)
        
        for i in range(self.n_bands):
            color = 'blue' if i < n_occ else 'red'
            
            # Use cubic spline interpolation for smoother curves
            try:
                spline = make_interp_spline(k_dist, bands_ev_shifted[:, i], k=3)
                bands_smooth = spline(k_smooth)
                
                # Clip smooth bands to avoid artifacts if needed, but here we just plot
                ax.plot(k_smooth, bands_smooth, color=color, linewidth=1.5)
            except Exception:
                # Fall back to raw data if interpolation fails
                ax.plot(k_dist, bands_ev_shifted[:, i], color=color, linewidth=1.5)
        
        for tick in ticks:
            ax.axvline(x=tick, color='gray', linestyle='--', linewidth=0.5)
        
        ax.axhline(y=0, color='green', linestyle='--', linewidth=1)
        
        ax.set_xlabel('Wave Vector', fontsize=12)
        ax.set_ylabel('Energy (eV)', fontsize=12)
        ax.set_title(f'Silicon Band Structure (UPF, ecut={self.ecut} Ha)', fontsize=14)
        ax.set_xlim(k_dist[0], k_dist[-1])
        
        # Adjust y-limits to see the gap clearly
        y_min = np.min(bands_ev_shifted) - 1.0
        y_max = np.max(bands_ev_shifted) + 1.0
        ax.set_ylim(max(-15, y_min), min(10, y_max))
        ax.set_xticks(ticks)
        ax.set_xticklabels(labels)
        
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], color='blue', label='Valence'),
            Line2D([0], [0], color='red', label='Conduction'),
            Line2D([0], [0], color='green', linestyle='--', label='$E_{VBM}$')
        ]
        ax.legend(handles=legend_elements, loc='upper right')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(filename, dpi=150)
        print(f"\nBand structure saved to: {filename}")
        
        # Band gap (corrected calculation)
        vbm_path = np.max(bands_ev[:, n_occ - 1])
        cbm_path = np.min(bands_ev[:, n_occ])
        gap_path = cbm_path - vbm_path
        
        # If path gap is negative, it might be due to low ecut causing overlap
        # but let's report the actual gap from the data provided to this function
        print(f"Indirect band gap (on path): {gap_path:.3f} eV")
        
        plt.show()
        return fig, ax
    
    def _r_to_g(self, arr_r):
        arr_fft = np.fft.fftn(arr_r) / self.n_fft
        return self.gvec.map_from_fft_grid(arr_fft)
    
    def _g_to_r(self, arr_g):
        arr_fft = self.gvec.map_to_fft_grid(arr_g, self.fft_shape)
        return np.real(np.fft.ifftn(arr_fft) * self.n_fft)


def main():
    print("\n" + "#" * 70)
    print("# Silicon DFT with Real Pseudopotential")
    print("#" * 70)
    
    # Create calculator
    si = SiliconUPF(ecut=10.0, n_bands=8)
    
    # Run SCF with k-point sampling (2x2x2 mesh)
    # Use kmesh=None for Gamma-only (faster but less accurate)
    si.run_scf(max_iter=60, tol=1e-5, kmesh=(2, 2, 2))
    
    # Compute bands
    k_dist, bands, ticks, labels = si.compute_bands(
        path_labels=['L', 'G', 'X', 'W', 'K', 'G'],
        npts=40
    )
    
    # Plot
    output_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 
                               'silicon_upf_bands.png')
    si.plot_bands(k_dist, bands, ticks, labels, filename=output_path)
    
    print("\n" + "#" * 70)
    print("# Calculation Complete!")
    print("#" * 70 + "\n")


if __name__ == "__main__":
    main()
