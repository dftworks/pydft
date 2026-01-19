#!/usr/bin/env python3
"""
K-point Sampled SCF Solver for Plane-Wave DFT.

This module provides a modular SCF solver that properly handles
k-point sampling for periodic systems.
"""

import numpy as np
from .constants import HA_TO_EV, TWOPI
from .gvector import GVector
from .xc import lda_xc, compute_xc_energy, compute_xc_potential_energy
from .hartree import compute_hartree_potential, compute_hartree_energy
from .hamiltonian import Hamiltonian, g_to_r
from .eigensolver import PCGEigensolver, random_initial_guess
from .mixing import LinearMixer, BroydenMixer


def generate_monkhorst_pack(nk1, nk2, nk3, shift=(0.0, 0.0, 0.0)):
    """
    Generate Monkhorst-Pack k-point mesh.
    
    Args:
        nk1, nk2, nk3: Number of k-points along each reciprocal lattice direction
        shift: Optional shift (0.5, 0.5, 0.5) for shifted grids
        
    Returns:
        kpoints_frac: Array of fractional k-points (nk, 3)
        weights: Array of k-point weights (nk,)
    """
    kpoints = []
    for i in range(nk1):
        for j in range(nk2):
            for k in range(nk3):
                kx = (2*i - nk1 + 1) / (2*nk1) + shift[0] / nk1
                ky = (2*j - nk2 + 1) / (2*nk2) + shift[1] / nk2
                kz = (2*k - nk3 + 1) / (2*nk3) + shift[2] / nk3
                kpoints.append([kx, ky, kz])
    
    kpoints = np.array(kpoints)
    nk = len(kpoints)
    weights = np.ones(nk) / nk
    
    return kpoints, weights


def reduce_kpoints_by_symmetry(kpoints_frac, weights, tol=1e-8):
    """
    Reduce k-points using time-reversal symmetry: k and -k are equivalent.
    
    This is a simple reduction; full space group symmetry would be more complete.
    
    Args:
        kpoints_frac: Fractional k-points
        weights: K-point weights
        tol: Tolerance for comparing k-points
        
    Returns:
        kpoints_reduced: Reduced k-point set
        weights_reduced: Updated weights
    """
    nk = len(kpoints_frac)
    used = np.zeros(nk, dtype=bool)
    kpoints_reduced = []
    weights_reduced = []
    
    for i in range(nk):
        if used[i]:
            continue
        
        k = kpoints_frac[i]
        w = weights[i]
        
        # Check if -k is in the set (and not the same as k)
        for j in range(i+1, nk):
            if used[j]:
                continue
            
            # Check if k[j] == -k[i] (mod 1)
            diff = kpoints_frac[j] + k
            diff = diff - np.round(diff)  # Bring to [-0.5, 0.5]
            
            if np.allclose(diff, 0, atol=tol):
                w += weights[j]
                used[j] = True
        
        kpoints_reduced.append(k)
        weights_reduced.append(w)
        used[i] = True
    
    return np.array(kpoints_reduced), np.array(weights_reduced)


class KPointData:
    """
    Stores data for a single k-point.
    
    Attributes:
        k_frac: Fractional coordinates
        k_cart: Cartesian coordinates
        weight: K-point weight
        evecs: Eigenvectors (npw, nbands)
        evals: Eigenvalues (nbands,)
        occupations: Band occupations (nbands,)
    """
    
    def __init__(self, k_frac, k_cart, weight, npw, nbands):
        self.k_frac = np.array(k_frac)
        self.k_cart = np.array(k_cart)
        self.weight = weight
        self.npw = npw
        self.nbands = nbands
        
        # Will be initialized during SCF
        self.evecs = None
        self.evals = None
        self.occupations = None
        
        # |k+G|^2 for kinetic energy
        self.kg_squared = None
        
        # Beta projectors for this k-point
        self.beta_kg = None


class KPointSCF:
    """
    SCF solver with proper k-point sampling.
    
    This class handles:
    - Monkhorst-Pack k-point generation
    - Eigenvalue problems at each k-point
    - Density summation over k-points
    - Total energy with k-point integration
    """
    
    def __init__(self, lattice, gvec, volume, n_electrons, n_bands,
                 vloc_g, nlpot=None, ewald_energy=0.0):
        """
        Initialize the k-point SCF solver.
        
        Args:
            lattice: Lattice object
            gvec: GVector object
            volume: Cell volume
            n_electrons: Number of electrons
            n_bands: Number of bands to compute
            vloc_g: Local potential in G-space (complex array)
            nlpot: NonlocalPotential object (optional)
            ewald_energy: Ion-ion energy
        """
        self.lattice = lattice
        self.gvec = gvec
        self.volume = volume
        self.n_electrons = n_electrons
        self.n_bands = n_bands
        self.vloc_g = vloc_g
        self.nlpot = nlpot
        self.ewald_energy = ewald_energy
        
        self.npw = gvec.npw
        self.fft_shape = gvec.get_fft_grid_size()
        self.n_fft = np.prod(self.fft_shape)
        
        # Reciprocal lattice vectors
        self.b = lattice.reciprocal_vectors
        
        # K-point data
        self.kpoints = []
        self.nk = 0
        
        # Density
        self.rho_r = None
        self.rho_g = None
        
        # Potentials
        self.v_local_r = None
        self._v_hartree_r = None
        self._v_xc_r = None
        self._exc_r = None
        
        # Hamiltonian template
        self.hamiltonian = Hamiltonian(gvec, volume)
    
    def setup_kpoints(self, nk1, nk2, nk3, shift=(0.0, 0.0, 0.0), 
                      use_symmetry=True):
        """
        Set up Monkhorst-Pack k-point mesh.
        
        Args:
            nk1, nk2, nk3: K-mesh dimensions
            shift: Grid shift
            use_symmetry: Whether to reduce by time-reversal symmetry
        """
        # Generate k-points
        kpoints_frac, weights = generate_monkhorst_pack(nk1, nk2, nk3, shift)
        
        # Reduce by symmetry
        if use_symmetry:
            kpoints_frac, weights = reduce_kpoints_by_symmetry(kpoints_frac, weights)
        
        self.nk = len(kpoints_frac)
        self.kpoints = []
        
        print(f"\nK-point mesh: {nk1}x{nk2}x{nk3}")
        print(f"Total k-points: {nk1*nk2*nk3}")
        print(f"Irreducible k-points: {self.nk}")
        
        for ik in range(self.nk):
            k_frac = kpoints_frac[ik]
            k_cart = k_frac[0]*self.b[0] + k_frac[1]*self.b[1] + k_frac[2]*self.b[2]
            
            kdata = KPointData(k_frac, k_cart, weights[ik], self.npw, self.n_bands)
            
            # Compute |k+G|^2
            kdata.kg_squared = np.zeros(self.npw)
            for ig in range(self.npw):
                kpg = k_cart + self.gvec.cart[ig]
                kdata.kg_squared[ig] = np.sum(kpg**2)
            
            # Compute beta projectors for this k-point
            if self.nlpot is not None:
                kdata.beta_kg = self.nlpot.get_beta_kg(k_cart)
            
            # Initialize eigenvectors
            kdata.evecs = random_initial_guess(self.npw, self.n_bands)
            kdata.evals = np.zeros(self.n_bands)
            
            self.kpoints.append(kdata)
    
    def _build_potential(self):
        """Build effective local potential from current density."""
        # Hartree potential
        v_hartree_g = compute_hartree_potential(self.rho_g, self.gvec.norms)
        v_hartree_fft = self.gvec.map_to_fft_grid(v_hartree_g, self.fft_shape)
        self._v_hartree_r = np.real(np.fft.ifftn(v_hartree_fft) * self.n_fft)
        
        # XC potential
        rho_real = np.maximum(np.real(self.rho_r), 1e-20)
        self._v_xc_r, self._exc_r = lda_xc(rho_real)
        
        # Local pseudopotential in real space
        vloc_fft = self.gvec.map_to_fft_grid(self.vloc_g, self.fft_shape)
        vloc_r = np.real(np.fft.ifftn(vloc_fft) * self.n_fft)
        
        # Total local potential
        self.v_local_r = self._v_hartree_r + self._v_xc_r + vloc_r
        
        self.hamiltonian.set_local_potential(self.v_local_r)
    
    def _apply_hamiltonian(self, psi, kdata):
        """
        Apply Hamiltonian at a specific k-point.
        
        Args:
            psi: Wavefunction in G-space
            kdata: KPointData object
            
        Returns:
            H|psi> in G-space
        """
        # Set kinetic energy for this k-point
        self.hamiltonian.kg = kdata.kg_squared
        
        # Local part (kinetic + local potential)
        hpsi = self.hamiltonian.apply(psi)
        
        # Non-local part
        if self.nlpot is not None and kdata.beta_kg is not None:
            hpsi += self.nlpot.apply_vnl(psi, kdata.beta_kg)
        
        return hpsi
    
    def _solve_eigenvalue(self, kdata, tol=1e-7, max_iter=100):
        """
        Solve eigenvalue problem for a single k-point.
        
        Args:
            kdata: KPointData object
            tol: Convergence tolerance
            max_iter: Maximum iterations
        """
        # Set kinetic energy for this k-point
        self.hamiltonian.kg = kdata.kg_squared
        
        def ham_apply(psi):
            return self._apply_hamiltonian(psi, kdata)
        
        eigensolver = PCGEigensolver(self.npw, self.n_bands)
        eigensolver.solve(
            ham_apply=ham_apply,
            ham_diag=self.hamiltonian.get_diagonal(),
            evecs=kdata.evecs,
            evals=kdata.evals,
            tol=tol,
            max_iter=max_iter
        )
    
    def _compute_density(self):
        """
        Compute electron density from wavefunctions at all k-points.
        
        rho(r) = sum_k w_k sum_n f_nk |psi_nk(r)|^2
        """
        rho_r_new = np.zeros(self.fft_shape, dtype=float)
        
        for kdata in self.kpoints:
            for n in range(self.n_bands):
                if kdata.occupations[n] < 1e-10:
                    continue
                
                psi_r = g_to_r(kdata.evecs[:, n], self.gvec, 
                              self.fft_shape, self.volume)
                rho_r_new += kdata.weight * kdata.occupations[n] * np.abs(psi_r)**2
        
        return rho_r_new
    
    def _set_occupations(self):
        """
        Set band occupations (fixed occupations for insulators).
        
        For metals, would use smearing and Fermi level search.
        """
        n_occ = self.n_electrons // 2
        
        for kdata in self.kpoints:
            kdata.occupations = np.zeros(self.n_bands)
            kdata.occupations[:n_occ] = 2.0
    
    def _compute_band_energy(self):
        """Compute total band energy sum_k w_k sum_n f_nk e_nk."""
        e_band = 0.0
        for kdata in self.kpoints:
            e_band += kdata.weight * np.sum(kdata.occupations * kdata.evals)
        return e_band
    
    def _compute_total_energy(self):
        """Compute total energy."""
        e_band = self._compute_band_energy()
        
        e_hartree = compute_hartree_energy(self.rho_g, self.gvec.norms, self.volume)
        
        rho_real = np.maximum(np.real(self.rho_r), 1e-20)
        e_xc = compute_xc_energy(rho_real, self._exc_r, self.volume, self.n_fft)
        e_vxc = compute_xc_potential_energy(rho_real, self._v_xc_r, self.volume, self.n_fft)
        
        e_total = e_band - e_hartree + e_xc - e_vxc + self.ewald_energy
        
        return e_total
    
    def _r_to_g(self, arr_r):
        """Transform from real space to G-space."""
        arr_fft = np.fft.fftn(arr_r) / self.n_fft
        return self.gvec.map_from_fft_grid(arr_fft)
    
    def _g_to_r(self, arr_g):
        """Transform from G-space to real space."""
        arr_fft = self.gvec.map_to_fft_grid(arr_g, self.fft_shape)
        return np.real(np.fft.ifftn(arr_fft) * self.n_fft)
    
    def run(self, max_iter=60, tol=1e-6, mixing_alpha=0.2, verbose=True):
        """
        Run SCF calculation with k-point sampling.
        
        Args:
            max_iter: Maximum SCF iterations
            tol: Energy convergence tolerance
            mixing_alpha: Density mixing parameter
            verbose: Print progress
            
        Returns:
            Total energy in Hartree
        """
        if self.nk == 0:
            raise RuntimeError("No k-points set up. Call setup_kpoints() first.")
        
        if verbose:
            print("\n" + "-" * 70)
            print("SCF Calculation with K-point Sampling")
            print("-" * 70)
            print(f"K-points: {self.nk}")
            print(f"Bands: {self.n_bands}")
            print(f"Electrons: {self.n_electrons}")
        
        # Initialize density (uniform)
        rho_0 = self.n_electrons / self.volume
        self.rho_r = np.full(self.fft_shape, rho_0, dtype=float)
        self.rho_g = self._r_to_g(self.rho_r)
        
        # Set fixed occupations
        self._set_occupations()
        
        # Mixer
        mixer = LinearMixer(alpha=mixing_alpha)
        
        energy_old = 0.0
        energy_history = []
        
        if verbose:
            print(f"\n{'Iter':>4} {'E_total (Ha)':>14} {'dE (Ha)':>12}")
            print("-" * 35)
        
        for scf_iter in range(1, max_iter + 1):
            # Build potential
            self._build_potential()
            
            # Solve eigenvalue problem at each k-point
            for kdata in self.kpoints:
                self._solve_eigenvalue(kdata)
            
            # Compute new density
            rho_r_new = self._compute_density()
            rho_g_new = self._r_to_g(rho_r_new)
            
            # Compute total energy
            # (temporarily update density for energy calculation)
            rho_g_old = self.rho_g
            rho_r_old = self.rho_r
            self.rho_g = rho_g_new
            self.rho_r = rho_r_new
            self._build_potential()
            
            e_total = self._compute_total_energy()
            energy_history.append(e_total)
            
            de = abs(e_total - energy_old)
            
            if verbose:
                print(f"{scf_iter:4d} {e_total:14.8f} {de:12.2e}")
            
            # Check convergence
            if de < tol and scf_iter > 3:
                if verbose:
                    print("-" * 35)
                    print(f"SCF converged in {scf_iter} iterations")
                break
            
            # Check oscillation
            if scf_iter > 10 and len(energy_history) >= 6:
                recent = energy_history[-6:]
                if np.std(recent) < tol * 10:
                    if verbose:
                        print("-" * 35)
                        print(f"SCF converged (oscillation) in {scf_iter} iterations")
                    e_total = np.mean(recent)
                    break
            
            # Mix densities
            self.rho_g = mixer.mix(rho_g_old, rho_g_new)
            self.rho_r = self._g_to_r(self.rho_g)
            
            energy_old = e_total
        
        self.total_energy = e_total
        return e_total
    
    def print_results(self):
        """Print SCF results."""
        print("\n" + "=" * 70)
        print("RESULTS")
        print("=" * 70)
        
        print(f"\nTotal energy: {self.total_energy:.8f} Ha "
              f"({self.total_energy * HA_TO_EV:.6f} eV)")
        print(f"Ewald energy: {self.ewald_energy:.8f} Ha")
        
        print(f"\nEigenvalues at each k-point:")
        for ik, kdata in enumerate(self.kpoints):
            print(f"\n  K-point {ik+1}: ({kdata.k_frac[0]:.4f}, "
                  f"{kdata.k_frac[1]:.4f}, {kdata.k_frac[2]:.4f}), "
                  f"weight = {kdata.weight:.4f}")
            print(f"  {'Band':>6} {'E (Ha)':>12} {'E (eV)':>12} {'Occ':>6}")
            print("  " + "-" * 38)
            for n in range(min(8, self.n_bands)):
                print(f"  {n+1:6d} {kdata.evals[n]:12.6f} "
                      f"{kdata.evals[n]*HA_TO_EV:12.6f} "
                      f"{kdata.occupations[n]:6.2f}")
        
        # Compute band gap
        n_occ = self.n_electrons // 2
        vbm = -np.inf
        cbm = np.inf
        
        for kdata in self.kpoints:
            vbm = max(vbm, kdata.evals[n_occ - 1])
            cbm = min(cbm, kdata.evals[n_occ])
        
        gap = (cbm - vbm) * HA_TO_EV
        print(f"\nBand gap: {gap:.4f} eV")
        print("=" * 70)
        
        return gap
    
    def get_converged_potential(self):
        """Return the converged local potential in real space."""
        return self.v_local_r
