#!/usr/bin/env python3
"""
Graphene DFT Calculation Example

This script demonstrates a plane-wave DFT calculation for graphene,
a two-dimensional material with remarkable electronic properties.

The crystal structure is:
- Hexagonal lattice with 2 carbon atoms per unit cell
- In-plane lattice constant: a = 2.46 Angstrom = 4.648 Bohr
- Large vacuum spacing along c to simulate a 2D sheet
- 8 valence electrons (4 per carbon atom)

Graphene is a semimetal with linear (Dirac) dispersion at the K point
of the hexagonal Brillouin zone, producing a zero band gap.
"""

import numpy as np
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.constants import HA_TO_EV, BOHR_TO_ANG, ANG_TO_BOHR, TWOPI
from src.lattice import Lattice
from src.crystal import Crystal
from src.gvector import GVector
from src.pwbasis import PWBasis
from src.kpoints import monkhorst_pack, gamma_only
from src.xc import lda_xc, compute_xc_energy, compute_xc_potential_energy
from src.hartree import compute_hartree_potential, compute_hartree_energy
from src.ewald import Ewald
from src.smearing import create_smearing, find_fermi_level
from src.hamiltonian import Hamiltonian, g_to_r
from src.eigensolver import PCGEigensolver, random_initial_guess
from src.mixing import LinearMixer


def create_graphene_crystal():
    """
    Create graphene crystal structure.

    Graphene has a hexagonal lattice with a two-atom basis:
    - Atom A (sublattice A): (1/3, 2/3, 0) in fractional coordinates
    - Atom B (sublattice B): (2/3, 1/3, 0)

    The two sublattices give rise to the honeycomb pattern.
    A large vacuum spacing along c prevents interaction between
    periodic images, effectively modelling an isolated 2D sheet.

    Returns:
        Crystal object, in-plane lattice constant in Bohr
    """
    # Graphene in-plane lattice constant
    a_ang = 2.46  # Angstrom (experimental)
    a_bohr = a_ang * ANG_TO_BOHR

    # Vacuum spacing along c (must be large enough to decouple layers)
    c_bohr = 20.0  # Bohr (~10.6 Angstrom)

    # Hexagonal primitive cell
    lattice = Lattice.hexagonal(a_bohr, c_bohr)

    # Create crystal
    crystal = Crystal(lattice)

    # Two carbon atoms in the honeycomb basis (fractional coordinates)
    crystal.add_atom('C', [1.0/3.0, 2.0/3.0, 0.0], zion=4)
    crystal.add_atom('C', [2.0/3.0, 1.0/3.0, 0.0], zion=4)

    return crystal, a_bohr


def create_graphene_potential(lattice, fft_shape, v_depth=1.2, r_c=1.2):
    """
    Create a simplified carbon-like model potential for graphene.

    Uses a Gaussian potential centered on each atom, similar to the
    silicon example.  The parameters are adjusted for carbon's smaller
    ionic radius and stronger potential.

    V(r) = -V_depth * exp(-r^2 / r_c^2) for each atom

    Args:
        lattice: Lattice object
        fft_shape: FFT grid shape
        v_depth: Depth of potential well (Hartree)
        r_c: Characteristic radius (Bohr)
    """
    n1, n2, n3 = fft_shape
    v_ext = np.zeros(fft_shape, dtype=float)

    # Fractional coordinates grid
    x = np.linspace(0, 1, n1, endpoint=False)
    y = np.linspace(0, 1, n2, endpoint=False)
    z = np.linspace(0, 1, n3, endpoint=False)

    # Atom positions in fractional coordinates (honeycomb basis)
    atoms = [
        np.array([1.0/3.0, 2.0/3.0, 0.0]),
        np.array([2.0/3.0, 1.0/3.0, 0.0])
    ]

    a, b, c = lattice.vectors

    for i, xi in enumerate(x):
        for j, yj in enumerate(y):
            for k, zk in enumerate(z):
                grid_frac = np.array([xi, yj, zk])

                for atom_frac in atoms:
                    # Minimum image convention
                    d_frac = grid_frac - atom_frac
                    d_frac = d_frac - np.round(d_frac)

                    # Convert to Cartesian
                    d_cart = d_frac[0]*a + d_frac[1]*b + d_frac[2]*c
                    r2 = np.sum(d_cart**2)

                    # Gaussian model potential
                    v_ext[i, j, k] += -v_depth * np.exp(-r2 / r_c**2)

    return v_ext


class GrapheneSCF:
    """
    SCF solver for graphene.

    This example highlights several features unique to 2D materials:
    - Hexagonal lattice with vacuum along the c direction
    - K-point mesh is dense in-plane but uses only 1 point along c
    - The Dirac cone at the K point produces a zero (or very small) gap
    """

    def __init__(self, ecut=5.0, n_bands=8, kpts_mesh=(4, 4, 1)):
        """
        Initialize graphene SCF calculation.

        Args:
            ecut: Energy cutoff in Hartree
            n_bands: Number of bands to compute
            kpts_mesh: K-point mesh (n1, n2, n3); n3 should be 1 for 2D
        """
        print("=" * 60)
        print("Graphene DFT Calculation (Educational)")
        print("=" * 60)

        # Create crystal structure
        self.crystal, self.a = create_graphene_crystal()
        self.lattice = self.crystal.lattice
        self.volume = self.lattice.volume

        print(f"\nCrystal: Graphene (Honeycomb)")
        print(f"In-plane lattice constant: {self.a:.4f} Bohr "
              f"({self.a * BOHR_TO_ANG:.4f} Ang)")
        print(f"Cell volume: {self.volume:.4f} Bohr^3")
        print(f"Atoms: {self.crystal.natoms}")

        # Number of electrons (4 valence per C)
        self.n_electrons = 8

        # Parameters
        self.ecut = ecut
        self.n_bands = n_bands

        # Generate G-vectors
        self.gvec = GVector(self.lattice, ecut)
        self.npw = self.gvec.npw

        # FFT grid
        self.fft_shape = self.gvec.get_fft_grid_size()
        self.n_fft = np.prod(self.fft_shape)

        print(f"\nEnergy cutoff: {ecut:.1f} Ha ({ecut * HA_TO_EV:.1f} eV)")
        print(f"Plane waves: {self.npw}")
        print(f"FFT grid: {self.fft_shape}")
        print(f"Bands: {n_bands}")
        print(f"Electrons: {self.n_electrons}")

        # K-points (dense in-plane, single point along c)
        self.kpts = monkhorst_pack(self.lattice, *kpts_mesh)
        self.nk = self.kpts.nk
        print(f"K-points: {kpts_mesh} mesh = {self.nk} points")

        # External potential (model pseudopotential)
        print("\nBuilding model pseudopotential...")
        self.v_ext = create_graphene_potential(self.lattice, self.fft_shape)

        # Ion-ion energy (Ewald)
        positions = self.crystal.get_positions()
        charges = self.crystal.get_charges()
        self.ewald = Ewald(self.lattice, positions, charges, self.gvec)
        print(f"Ewald ion-ion energy: {self.ewald.energy:.6f} Ha")

        # Hamiltonian
        self.hamiltonian = Hamiltonian(self.gvec, self.volume)

        # Eigensolver
        self.eigensolver = PCGEigensolver(self.npw, self.n_bands)

        # Mixer - simple linear mixing with a small alpha, consistent
        # with the other examples (converges in ~20 iterations).
        self.mixer = LinearMixer(alpha=0.1)

        # Smearing (Gaussian, broader than for Si because graphene is a semimetal)
        self.smearing = create_smearing('gaussian', sigma=0.02)

        # Storage
        self.evecs = None
        self.evals = None
        self.rho_r = None
        self.rho_g = None
        self.occupations = None

    def run(self, max_iter=50, tol=1e-4, verbose=True):
        """
        Run SCF calculation.

        Returns:
            Total energy in Hartree
        """
        print("\n" + "-" * 60)
        print("Starting SCF iteration...")
        print("-" * 60)

        # Initialize density (uniform)
        rho_0 = self.n_electrons / self.volume
        self.rho_r = np.full(self.fft_shape, rho_0, dtype=float)
        self.rho_g = self._r_to_g_density(self.rho_r)

        # Initialize eigenvectors
        self.evecs = random_initial_guess(self.npw, self.n_bands)
        self.evals = np.zeros(self.n_bands)

        # Fixed occupations (4 occupied bands with 2 electrons each)
        self.occupations = np.zeros(self.n_bands)
        n_occ = self.n_electrons // 2
        self.occupations[:n_occ] = 2.0

        # SCF loop
        energy_old = 0.0
        converged = False
        energy_history = []

        if verbose:
            print(f"{'Iter':>4} {'E_total (Ha)':>16} {'dE (Ha)':>12} "
                  f"{'dE (eV)':>12}")
            print("-" * 50)

        for scf_iter in range(1, max_iter + 1):
            # Build potentials
            self._build_potential()

            # Solve eigenvalue problem
            self.eigensolver.solve(
                ham_apply=self.hamiltonian.apply,
                ham_diag=self.hamiltonian.get_diagonal(),
                evecs=self.evecs,
                evals=self.evals,
                tol=1e-8,
                max_iter=200
            )

            # Compute new density
            rho_new = self._compute_density()

            # Compute total energy
            energy = self._compute_total_energy()
            energy_history.append(energy)

            # Check convergence
            de = abs(energy - energy_old)

            if verbose:
                print(f"{scf_iter:4d} {energy:16.8f} {de:12.2e} "
                      f"{de * HA_TO_EV:12.2e}")

            if de < tol:
                converged = True
                print("-" * 50)
                print(f"SCF CONVERGED in {scf_iter} iterations")
                break

            # Check for oscillation
            if scf_iter > 10:
                recent = energy_history[-6:]
                if len(recent) == 6:
                    odd_vals = [recent[0], recent[2], recent[4]]
                    even_vals = [recent[1], recent[3], recent[5]]
                    odd_std = np.std(odd_vals)
                    even_std = np.std(even_vals)
                    if odd_std < tol and even_std < tol:
                        energy = np.mean(recent)
                        converged = True
                        print("-" * 50)
                        print(f"SCF oscillating - using average energy "
                              f"after {scf_iter} iterations")
                        break

            # Mix densities
            rho_mixed = self.mixer.mix(self.rho_g, rho_new)
            self.rho_g = rho_mixed
            self.rho_r = self._g_to_r_density(self.rho_g)

            energy_old = energy

        if not converged:
            print("-" * 50)
            print(f"Warning: SCF not converged after {max_iter} iterations")
            if len(energy_history) > 5:
                energy = np.mean(energy_history[-6:])

        # Print final results
        self._print_results(energy)

        return energy

    def _r_to_g_density(self, rho_r):
        """Transform density from real space to G-space."""
        rho_fft = np.fft.fftn(rho_r) / self.n_fft
        return self.gvec.map_from_fft_grid(rho_fft)

    def _g_to_r_density(self, rho_g):
        """Transform density from G-space to real space."""
        rho_fft = self.gvec.map_to_fft_grid(rho_g, self.fft_shape)
        rho_r = np.fft.ifftn(rho_fft) * self.n_fft
        return np.real(rho_r)

    def _build_potential(self):
        """Build effective potential."""
        # Hartree potential
        v_hartree_g = compute_hartree_potential(self.rho_g, self.gvec.norms)

        # Transform to real space
        v_hartree_fft = self.gvec.map_to_fft_grid(v_hartree_g, self.fft_shape)
        v_hartree_r = np.real(np.fft.ifftn(v_hartree_fft) * self.n_fft)

        # XC potential
        rho_real = np.maximum(np.real(self.rho_r), 1e-20)
        v_xc_r, self._exc_r = lda_xc(rho_real)

        # Total local potential
        v_local_r = v_hartree_r + v_xc_r + self.v_ext

        # Store for energy calculation
        self._v_hartree_r = v_hartree_r
        self._v_xc_r = v_xc_r

        # Set in Hamiltonian
        self.hamiltonian.set_local_potential(v_local_r)

    def _compute_density(self):
        """Compute new density from wavefunctions."""
        rho_r = np.zeros(self.fft_shape, dtype=float)

        for i in range(self.n_bands):
            if self.occupations[i] < 1e-10:
                continue

            psi_r = g_to_r(self.evecs[:, i], self.gvec,
                           self.fft_shape, self.volume)
            rho_r += self.occupations[i] * np.abs(psi_r)**2

        return self._r_to_g_density(rho_r)

    def _compute_total_energy(self):
        """Compute total energy."""
        # Band energy
        e_band = np.sum(self.occupations * self.evals)

        # Hartree energy
        e_hartree = compute_hartree_energy(
            self.rho_g, self.gvec.norms, self.volume)

        # XC energy
        rho_real = np.maximum(np.real(self.rho_r), 1e-20)
        e_xc = compute_xc_energy(rho_real, self._exc_r,
                                 self.volume, self.n_fft)

        # XC potential energy (double-counting)
        e_vxc = compute_xc_potential_energy(
            rho_real, self._v_xc_r, self.volume, self.n_fft)

        # Total (Harris-Foulkes formula)
        e_total = e_band - e_hartree + e_xc - e_vxc + self.ewald.energy

        return e_total

    def _print_results(self, energy):
        """Print final calculation results."""
        print("\n" + "=" * 60)
        print("CALCULATION RESULTS")
        print("=" * 60)

        print(f"\nMaterial: Graphene")
        print(f"Total energy: {energy:.8f} Ha ({energy * HA_TO_EV:.6f} eV)")
        print(f"Energy per atom: {energy/2:.8f} Ha "
              f"({energy/2 * HA_TO_EV:.6f} eV)")
        print(f"Ewald energy: {self.ewald.energy:.8f} Ha")

        print(f"\nEigenvalues (Gamma point):")
        print(f"{'Band':>6} {'Energy (Ha)':>14} {'Energy (eV)':>14} "
              f"{'Occ':>8}")
        print("-" * 44)
        for i in range(self.n_bands):
            print(f"{i+1:6d} {self.evals[i]:14.6f} "
                  f"{self.evals[i]*HA_TO_EV:14.6f} "
                  f"{self.occupations[i]:8.2f}")

        # Band gap (graphene should have zero or very small gap)
        n_occ = self.n_electrons // 2
        if n_occ < self.n_bands:
            vbm = self.evals[n_occ - 1]
            cbm = self.evals[n_occ]
            gap = cbm - vbm
            print(f"\nBand gap at Gamma: {gap:.4f} Ha "
                  f"({gap * HA_TO_EV:.4f} eV)")
            print("(Note: Graphene's zero gap occurs at K, not Gamma)")

        print("=" * 60)


def print_brillouin_zone_info():
    """Print high-symmetry points for the hexagonal Brillouin zone."""
    print("\n" + "=" * 60)
    print("Hexagonal Brillouin Zone")
    print("=" * 60)
    print("\nHigh-symmetry points (fractional reciprocal coordinates):")
    print(f"  Gamma = (0,     0,     0  )")
    print(f"  M     = (1/2,   0,     0  )")
    print(f"  K     = (1/3,   1/3,   0  )")
    print(f"  K'    = (2/3,   1/3,   0  )  [= -K by time reversal]")
    print(f"\nCommon band structure path: Gamma -> M -> K -> Gamma")
    print(f"\nThe Dirac cone (linear dispersion, zero gap) is at K and K'.")
    print("=" * 60)


def main():
    """Run graphene calculation."""
    # Fixed seed for reproducible SCF output (see silicon example).
    np.random.seed(42)
    print("\n" + "#" * 60)
    print("# Educational Graphene DFT Calculation")
    print("#" * 60)

    # Print Brillouin zone information
    print_brillouin_zone_info()

    # Create and run graphene SCF
    gr_scf = GrapheneSCF(
        ecut=5.0,                # Moderate cutoff for speed
        n_bands=8,               # 4 valence + 4 conduction
        kpts_mesh=(4, 4, 1)      # Dense in-plane, 1 along vacuum direction
    )

    # Run SCF with educational-level tolerance
    energy = gr_scf.run(max_iter=60, tol=1e-4)

    print("\n" + "#" * 60)
    print("# Calculation Complete!")
    print("#" * 60 + "\n")

    return energy


if __name__ == "__main__":
    main()
