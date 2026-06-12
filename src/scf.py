"""
Self-Consistent Field (SCF) driver.

This module implements a minimal Gamma-point Kohn-Sham cycle:
    1. rho -> V_eff (Hartree + XC + external)
    2. solve H psi = eps psi
    3. psi -> rho_new
    4. mix rho and rho_new
    5. check energy convergence

The code is intentionally explicit (not heavily abstracted) so that readers
can trace how each DFT equation maps to concrete arrays.

=============================================================================
PEDAGOGICAL NOTES
=============================================================================

The Self-Consistent Field (SCF) method solves the Kohn-Sham equations:

    [-ℏ²/(2m)∇² + V_eff(r)] ψ_i(r) = ε_i ψ_i(r)

The challenge is that V_eff depends on the electron density ρ(r), which in
turn depends on the wavefunctions ψ_i(r):

    ρ(r) = Σ_i f_i |ψ_i(r)|²

This creates a self-consistency problem: we need ρ to find V_eff, but we
need V_eff to find the ψ_i that give us ρ!

SOLUTION: Iterative SCF procedure:
    1. Start with an initial guess for ρ(r)
    2. Compute V_eff = V_H[ρ] + V_xc[ρ] + V_ext
    3. Solve the eigenvalue problem to get new ψ_i
    4. Compute new ρ_new from the ψ_i
    5. Mix ρ_old and ρ_new to get ρ_next (prevents oscillations)
    6. Check if energy has converged; if not, go to step 2

KEY INSIGHT: The mixing step (5) is crucial! Without it, the density can
oscillate wildly and never converge. Broyden mixing uses history to
accelerate convergence.

WHY G-SPACE (RECIPROCAL SPACE)?
- Kinetic energy T = -½∇² becomes diagonal: T|G⟩ = ½|G|² |G⟩
- Hartree potential V_H is trivial: V_H(G) = 4πρ(G)/|G|²
- Periodic boundary conditions are naturally satisfied
- Cutoff in |G| provides systematic basis set convergence
=============================================================================
"""

import numpy as np
from .constants import HA_TO_EV
from .lattice import Lattice
from .gvector import GVector
from .pwbasis import PWBasis
from .xc import lda_xc, compute_xc_energy, compute_xc_potential_energy
from .hartree import compute_hartree_potential, compute_hartree_energy
from .hamiltonian import Hamiltonian, g_to_r
from .eigensolver import PCGEigensolver, random_initial_guess
from .mixing import BroydenMixer, LinearMixer


class SCFSolver:
    """
    Self-Consistent Field solver for plane-wave DFT.
    
    Implements the basic SCF loop:
    1. Build potential from density
    2. Solve eigenvalue problem
    3. Build new density from wavefunctions
    4. Mix densities
    5. Check convergence
    
    Attributes:
        lattice: Crystal lattice
        gvec: G-vector grid
        n_bands: Number of bands
        n_electrons: Number of electrons
    """
    
    def __init__(self, lattice, ecut, n_bands, n_electrons,
                 external_potential=None, mixer='linear'):
        """
        Initialize SCF solver.

        Args:
            lattice: Lattice object
            ecut: Energy cutoff in Hartree
            n_bands: Number of bands to compute
            n_electrons: Number of electrons
            external_potential: Optional external potential V_ext(r) on FFT grid
            mixer: 'linear' (default; simplest, needs a small alpha) or
                   'broyden' (faster convergence using SCF history)
        """
        self.lattice = lattice
        self.ecut = ecut
        self.n_bands = n_bands
        self.n_electrons = n_electrons
        self.volume = lattice.volume
        
        # Generate G-vectors
        self.gvec = GVector(lattice, ecut)
        self.npw = self.gvec.npw
        
        # FFT grid
        self.fft_shape = self.gvec.get_fft_grid_size()
        self.n_fft = np.prod(self.fft_shape)
        
        # Plane wave basis (Gamma point only)
        self.pwbasis = PWBasis(self.gvec)
        
        # Hamiltonian
        self.hamiltonian = Hamiltonian(self.gvec, self.volume)
        
        # External local potential term used by this pedagogical SCF driver.
        if external_potential is not None:
            self.v_ext = external_potential
        else:
            # Default to a flat zero potential (jellium background)
            self.v_ext = np.zeros(self.fft_shape)
        
        # Eigensolver
        self.eigensolver = PCGEigensolver(self.npw, self.n_bands)
        
        # Mixer
        if mixer == 'broyden':
            self.mixer = BroydenMixer(alpha=0.7, n_history=8)
        else:
            self.mixer = LinearMixer(alpha=0.3)
        
        # Occupations use a simple aufbau-like filling with spin pairing.
        self.occupations = self._compute_occupations()
        
        # Storage
        self.evecs = None
        self.evals = None
        self.rho_r = None
        self.rho_g = None
    
    def _create_harmonic_potential(self, omega=0.1):
        """
        Create the default harmonic confining potential.
        
        V(r) = 0.5 * omega^2 * |r - r_center|^2
        """
        n1, n2, n3 = self.fft_shape
        
        # Real-space grid
        x = np.linspace(0, 1, n1, endpoint=False)
        y = np.linspace(0, 1, n2, endpoint=False)
        z = np.linspace(0, 1, n3, endpoint=False)
        
        X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
        
        # Center at (0.5, 0.5, 0.5) in fractional coordinates
        dx = X - 0.5
        dy = Y - 0.5
        dz = Z - 0.5
        
        # Convert to Cartesian distances
        # For simplicity, use lattice vectors
        a, b, c = self.lattice.vectors
        
        r2 = np.zeros_like(X)
        for i in range(n1):
            for j in range(n2):
                for k in range(n3):
                    r_cart = dx[i,j,k] * a + dy[i,j,k] * b + dz[i,j,k] * c
                    r2[i,j,k] = np.sum(r_cart**2)
        
        return 0.5 * omega**2 * r2
    
    def _compute_occupations(self):
        """
        Compute occupation numbers by filling the lowest bands.

        Convention in this module:
        - each spatial orbital can host up to 2 electrons (spin-paired),
        - occupations are therefore in [0, 2].

        PEDAGOGICAL NOTE: Aufbau Principle
        ----------------------------------
        In the ground state, electrons fill orbitals from lowest to highest
        energy. With spin-pairing (no spin polarization), each spatial
        orbital can hold 2 electrons (one spin-up, one spin-down).

        Example: 8 electrons with 6 bands
            Band 1: occ = 2.0  (filled)
            Band 2: occ = 2.0  (filled)
            Band 3: occ = 2.0  (filled)
            Band 4: occ = 2.0  (filled)
            Band 5: occ = 0.0  (empty - above Fermi level)
            Band 6: occ = 0.0  (empty)

        For metals, we would use Fermi-Dirac smearing instead of this
        simple step-function filling. See smearing.py for details.
        """
        occ = np.zeros(self.n_bands)
        n_filled = self.n_electrons // 2  # Assuming spin-paired
        remainder = self.n_electrons % 2

        for i in range(min(n_filled, self.n_bands)):
            occ[i] = 2.0  # Spin-paired

        if remainder > 0 and n_filled < self.n_bands:
            occ[n_filled] = float(remainder)

        return occ
    
    def run(self, max_iter=50, tol=1e-6, verbose=True):
        """
        Run SCF calculation.
        
        Args:
            max_iter: Maximum SCF iterations
            tol: Energy convergence tolerance (Hartree)
            verbose: Print progress
        
        Returns:
            Total energy (Hartree)
        """
        if verbose:
            print("=" * 60)
            print("SCF Calculation")
            print("=" * 60)
            print(f"  Lattice volume: {self.volume:.4f} Bohr^3")
            print(f"  Energy cutoff: {self.ecut:.2f} Ha ({self.ecut * HA_TO_EV:.2f} eV)")
            print(f"  Plane waves: {self.npw}")
            print(f"  FFT grid: {self.fft_shape}")
            print(f"  Bands: {self.n_bands}")
            print(f"  Electrons: {self.n_electrons}")
            print("-" * 60)
        
        # Phase 0: start from a neutral uniform density.
        self._initialize_density()
        
        # Random but orthonormal initial orbitals.
        self.evecs = random_initial_guess(self.npw, self.n_bands)
        self.evals = np.zeros(self.n_bands)
        
        # SCF loop
        energy_old = 0.0
        converged = False
        
        if verbose:
            print(f"{'Iter':>4} {'Energy (Ha)':>16} {'dE (Ha)':>12} {'dE (eV)':>12}")
            print("-" * 60)
        
        for scf_iter in range(1, max_iter + 1):
            # Phase 1: rho -> V_eff
            self._build_potential()
            
            # Phase 2: solve Kohn-Sham eigenproblem in current potential.
            self.eigensolver.solve(
                ham_apply=self.hamiltonian.apply,
                ham_diag=self.hamiltonian.get_diagonal(),
                evecs=self.evecs,
                evals=self.evals,
                tol=1e-8,
                max_iter=50
            )
            
            # Phase 3: build rho_new from occupied orbitals.
            rho_g_new = self._compute_density()
            
            # Evaluate the Harris-Foulkes-like total energy expression.
            energy = self._compute_total_energy()
            
            # Check convergence
            de = abs(energy - energy_old)
            
            if verbose:
                print(f"{scf_iter:4d} {energy:16.8f} {de:12.2e} {de * HA_TO_EV:12.2e}")
            
            if de < tol:
                converged = True
                if verbose:
                    print("-" * 60)
                    print(f"SCF converged in {scf_iter} iterations")
                break
            
            # Phase 4: mix old/new densities to damp fixed-point oscillations.
            rho_mixed = self.mixer.mix(self.rho_g, rho_g_new)
            self.rho_g = rho_mixed
            
            # Keep real-space and reciprocal-space densities in sync.
            self.rho_r = self._g_to_r_density(self.rho_g)
            
            energy_old = energy
        
        if not converged and verbose:
            print("-" * 60)
            print(f"SCF did not converge in {max_iter} iterations")
        
        # Print final results
        if verbose:
            print("-" * 60)
            print("Final eigenvalues (Ha / eV):")
            for i, (e, occ) in enumerate(zip(self.evals, self.occupations)):
                print(f"  Band {i+1}: {e:12.6f} / {e * HA_TO_EV:12.6f}  occ: {occ:.1f}")
            print(f"\nTotal energy: {energy:.8f} Ha ({energy * HA_TO_EV:.6f} eV)")
            print("=" * 60)
        
        return energy
    
    def _initialize_density(self):
        """
        Initialize electron density (uniform).

        PEDAGOGICAL NOTE: Initial Density Guess
        ---------------------------------------
        The initial density strongly affects convergence speed. Options:

        1. UNIFORM (used here): ρ(r) = N_electrons / Volume
           - Simple but far from final answer
           - Works for jellium-like systems

        2. ATOMIC SUPERPOSITION (production codes): ρ(r) = Σ_atom ρ_atom(r)
           - Much closer to final density
           - Faster convergence for real materials

        The uniform guess integrates to N_electrons by construction:
            ∫ ρ(r) dr = (N_e/V) × V = N_e  ✓
        """
        # Uniform density integrates to n_electrons by construction.
        rho_0 = self.n_electrons / self.volume

        # Real-space density on the FFT grid
        self.rho_r = np.full(self.fft_shape, rho_0)

        # G-space density (Fourier transform of rho_r)
        # For uniform density: only ρ(G=0) is nonzero
        self.rho_g = self._r_to_g_density(self.rho_r)
    
    def _r_to_g_density(self, rho_r):
        """
        Transform density from real space to G-space.

        PEDAGOGICAL NOTE: FFT Conventions
        ----------------------------------
        The continuous Fourier transform of density is:
            ρ(G) = (1/Ω) ∫ ρ(r) e^{-iG·r} dr

        On a discrete grid with N points:
            ρ(G) ≈ (1/N) Σ_r ρ(r) e^{-iG·r}  (Riemann sum)

        NumPy's fftn computes: Σ_r ρ(r) e^{-iG·r}  (no 1/N factor)
        So we divide by N to get proper Fourier coefficients.

        The result ρ(G=0) equals the average density:
            ρ(G=0) = (1/N) Σ_r ρ(r) = ⟨ρ⟩
        """
        # NumPy FFT is unnormalized; divide by N to represent Fourier coefficients.
        rho_fft = np.fft.fftn(rho_r) / self.n_fft

        # Map from full FFT grid to our G-vector list (subset within cutoff)
        rho_g = self.gvec.map_from_fft_grid(rho_fft)

        return rho_g

    def _g_to_r_density(self, rho_g):
        """
        Transform density from G-space to real space.

        PEDAGOGICAL NOTE: Inverse FFT
        ------------------------------
        The inverse transform recovers real-space density:
            ρ(r) = Σ_G ρ(G) e^{+iG·r}

        NumPy's ifftn computes: (1/N) Σ_G ρ(G) e^{+iG·r}
        So we multiply by N to undo the 1/N factor.

        Combined with _r_to_g_density, we have:
            ρ(r) → ρ(G) → ρ(r)  is identity (up to FFT precision)
        """
        # Map from G-vector list to full FFT grid
        rho_fft = self.gvec.map_to_fft_grid(rho_g, self.fft_shape)

        # Inverse of the scaling used in _r_to_g_density.
        rho_r = np.fft.ifftn(rho_fft) * self.n_fft

        # Density must be real (imaginary part is numerical noise)
        return np.real(rho_r)
    
    def _build_potential(self):
        """
        Build effective potential V_eff = V_H + V_xc + V_ext.

        PEDAGOGICAL NOTE: The Three Potential Components
        -------------------------------------------------
        The Kohn-Sham effective potential has three parts:

        1. HARTREE POTENTIAL V_H(r):
           - Describes electron-electron Coulomb repulsion
           - Computed in G-space: V_H(G) = 4π ρ(G) / |G|²
           - This is just solving Poisson's equation: ∇²V_H = -4πρ

        2. EXCHANGE-CORRELATION POTENTIAL V_xc(r):
           - Captures quantum many-body effects (exchange + correlation)
           - In LDA: V_xc(r) = dE_xc/dρ evaluated at local density ρ(r)
           - This is the "magic" of DFT - approximates many-body physics!

        3. EXTERNAL POTENTIAL V_ext(r):
           - Usually the ionic potential from nuclei/pseudopotentials
           - In this pedagogical code: harmonic well or jellium background

        WHY DIFFERENT SPACES?
        - V_H is natural in G-space (algebraic vs solving PDE)
        - V_xc is natural in r-space (depends on local density)
        - We transform V_H to r-space so H|ψ⟩ can use a single FFT
        """
        # ===== STEP 1: Hartree potential (G-space calculation) =====
        # V_H(G) = 4πρ(G)/|G|² is trivial in reciprocal space!
        # Compare to solving ∇²V_H = -4πρ in real space (much harder)
        v_hartree_g = compute_hartree_potential(self.rho_g, self.gvec.norms)

        # ===== STEP 2: XC potential (real-space calculation) =====
        # LDA: V_xc depends only on ρ(r) at each point
        # Protect against negative/zero density (numerical noise)
        rho_real = np.maximum(np.real(self.rho_r), 1e-20)
        v_xc_r, self._exc_r = lda_xc(rho_real)

        # ===== STEP 3: Transform Hartree to real space =====
        # We need V_H(r) to apply the potential via FFT convolution
        v_hartree_fft = self.gvec.map_to_fft_grid(v_hartree_g, self.fft_shape)
        v_hartree_r = np.real(np.fft.ifftn(v_hartree_fft) * self.n_fft)

        # ===== STEP 4: Sum all contributions =====
        # V_eff(r) = V_H(r) + V_xc(r) + V_ext(r)
        v_local_r = v_hartree_r + v_xc_r + self.v_ext

        # Store for energy calculation (need individual components)
        self._v_hartree_r = v_hartree_r
        self._v_xc_r = v_xc_r

        # Pass to Hamiltonian for H|ψ⟩ application
        self.hamiltonian.set_local_potential(v_local_r)
    
    def _compute_density(self):
        """
        Compute updated density from occupied Kohn-Sham orbitals.

        Returns:
            rho_g: Density in reciprocal space, matching mixer input.
        """
        # Real-space density
        rho_r = np.zeros(self.fft_shape, dtype=float)
        
        for i in range(self.n_bands):
            if self.occupations[i] < 1e-10:
                continue
            
            # Build |psi_n(r)|^2 on the FFT grid.
            psi_r = g_to_r(self.evecs[:, i], self.gvec, 
                          self.fft_shape, self.volume)
            
            # Add to density
            rho_r += self.occupations[i] * np.abs(psi_r)**2
        
        # Mixer and Hartree routines work with rho(G).
        rho_g = self._r_to_g_density(rho_r)
        
        return rho_g
    
    def _compute_total_energy(self):
        """
        Compute total energy.

        E_tot = E_band - E_H + E_xc - E_Vxc

        This base-class implementation covers the pedagogical jellium /
        harmonic-well case.  The UPF-enabled SCF solver (see silicon_upf.py)
        adds ion-ion Ewald and pseudopotential energy terms.

        PEDAGOGICAL NOTE: Double-Counting Correction
        ---------------------------------------------
        This is one of the trickiest parts of DFT to understand!

        The band energy is:
            E_band = Σ_i f_i ε_i = Σ_i f_i ⟨ψ_i|H|ψ_i⟩
                   = Σ_i f_i ⟨ψ_i| T + V_H + V_xc + V_ext |ψ_i⟩

        But this OVER-COUNTS the Hartree and XC energies!

        WHY? Consider Hartree:
        - E_H = (1/2) ∫∫ ρ(r)ρ(r')/|r-r'| dr dr' = (1/2) ∫ V_H(r)ρ(r) dr
        - But E_band contains ∫ V_H(r)ρ(r) dr (the full integral, not half!)

        So we must subtract the extra Hartree:
            E_tot = E_band - E_H  (removes the double-counted half)

        Similar for XC:
        - E_band contains ∫ V_xc(r)ρ(r) dr = E_Vxc
        - But true XC energy is E_xc = ∫ ε_xc(r)ρ(r) dr ≠ E_Vxc
        - (They differ because V_xc = d(ρε_xc)/dρ ≠ ε_xc)

        Final formula:
            E_tot = E_band - E_H + E_xc - E_Vxc + E_ion-ion

        This correctly counts each interaction exactly once!
        """
        # ===== Band energy: sum of occupied eigenvalues =====
        # E_band = Σ_i f_i ε_i (includes T + V_H + V_xc + V_ext)
        e_band = np.sum(self.occupations * self.evals)

        # ===== Hartree energy (for double-counting correction) =====
        # E_H = (Ω/2) Σ_{G≠0} 4π|ρ(G)|²/|G|²
        e_hartree = compute_hartree_energy(self.rho_g, self.gvec.norms, self.volume)

        # ===== XC energy =====
        # E_xc = ∫ ρ(r) ε_xc(ρ(r)) dr
        rho_real = np.maximum(np.real(self.rho_r), 1e-20)
        e_xc = compute_xc_energy(rho_real, self._exc_r, self.volume, self.n_fft)

        # ===== XC potential energy (for double-counting correction) =====
        # E_Vxc = ∫ ρ(r) V_xc(r) dr  (this is what E_band contains)
        e_vxc = compute_xc_potential_energy(rho_real, self._v_xc_r,
                                            self.volume, self.n_fft)

        # Diagnostic only: e_ext is already included implicitly in e_band.
        # We keep this explicit term for teaching and consistency checks.
        e_ext = (self.volume / self.n_fft) * np.sum(rho_real * self.v_ext)

        # ===== Total energy with double-counting corrections =====
        # E_tot = E_band - E_H + E_xc - E_Vxc
        # The -E_H removes the extra half of Hartree in E_band
        # The +E_xc - E_Vxc replaces ∫ρV_xc with ∫ρε_xc
        e_total = e_band - e_hartree + e_xc - e_vxc

        return e_total
    
    def get_eigenvalues(self):
        """Return eigenvalues in Hartree."""
        return self.evals.copy() if self.evals is not None else None
    
    def get_density(self):
        """Return electron density in real space."""
        return self.rho_r.copy() if self.rho_r is not None else None


def create_jellium_box(a, n_electrons, ecut, n_bands=None):
    """
    Create a jellium box calculation.
    
    Jellium: uniform positive background with electrons.
    
    Args:
        a: Box side length (Bohr)
        n_electrons: Number of electrons
        ecut: Energy cutoff (Hartree)
        n_bands: Number of bands (default: n_electrons // 2 + 4)
    
    Returns:
        SCFSolver configured for jellium box
    """
    lattice = Lattice.cubic(a)
    
    if n_bands is None:
        n_bands = n_electrons // 2 + 4
    
    # No external potential (just uniform background)
    solver = SCFSolver(
        lattice=lattice,
        ecut=ecut,
        n_bands=n_bands,
        n_electrons=n_electrons,
        external_potential=None  # Will create harmonic potential by default
    )
    
    # Actually set V_ext to zero for true jellium
    solver.v_ext = np.zeros(solver.fft_shape)
    
    return solver
