"""
Plane wave basis set for a specific k-point.

Extends GVector with k-point specific information.

=============================================================================
PEDAGOGICAL NOTES
=============================================================================

THE PLANE WAVE BASIS AT A SPECIFIC k-POINT
------------------------------------------
In a periodic crystal, the Kohn-Sham orbitals according to Bloch's theorem are:
    ψ_{nk}(r) = e^{ik·r} u_{nk}(r)

The periodic part u_{nk}(r) is expanded in plane waves using the reciprocal 
lattice vectors (G-vectors):
    u_{nk}(r) = Σ_G c_{nk}(G) e^{iG·r}

Therefore, the full wavefunction is represented as:
    ψ_{nk}(r) = Σ_G c_{nk}(G) e^{i(k+G)·r}

This means that for each k-point in the Brillouin zone, the effective plane 
wave basis vectors are |k+G⟩, with wavevectors (k+G).

KINETIC ENERGY
--------------
The kinetic energy operator is T = -½∇². When applied to our basis functions:
    T |k+G⟩ = ½|k+G|² |k+G⟩

For a given energy cutoff E_cut, we keep all G-vectors such that:
    ½|k+G|² < E_cut

Note: In this simple pedagogical code, we often use the same G-vector set for 
all k-points (defined by ½|G|² < E_cut for the Gamma point), but their 
kinetic energies ½|k+G|² are correctly computed specifically for each k-point.

CROSS-REFERENCE: 
See `pydft-book`, Chapter "Basis", Section "Plane Wave Basis" 
(`planewavebasis.tex`) and Section "Brillouin zone sampling" (`ksampling.tex`)
for detailed theory.
=============================================================================
"""

import numpy as np
from .gvector import GVector


class PWBasis:
    """
    Plane wave basis for a specific k-point.
    
    This is essentially a wrapper around GVector that adds k-point
    specific quantities like |k+G|^2 kinetic energies.
    
    Attributes:
        gvec: GVector object (shared among k-points)
        k_cart: k-point in Cartesian coordinates
        kg: |k+G|^2 / 2 kinetic energies
        npw: Number of plane waves
    """
    
    def __init__(self, gvec, k_cart=None):
        """
        Create plane wave basis for a k-point.
        
        Args:
            gvec: GVector object
            k_cart: k-point in Cartesian coordinates (default: Gamma)
        """
        self.gvec = gvec
        self.k_cart = np.zeros(3) if k_cart is None else np.array(k_cart)
        
        # Compute |k+G|^2 / 2
        self.kg = self._compute_kinetic()
        self.npw = gvec.npw
    
    def _compute_kinetic(self):
        """Compute kinetic energies |k+G|^2 / 2."""
        k_plus_g = self.gvec.cart + self.k_cart
        return 0.5 * np.sum(k_plus_g**2, axis=1)
    
    def get_kg(self):
        """Return kinetic energies."""
        return self.kg
    
    def get_npw(self):
        """Return number of plane waves."""
        return self.npw
    
    def get_gindex(self):
        """Return indices in the shared G-vector list."""
        return np.arange(self.npw)
    
    def __repr__(self):
        return f"PWBasis(npw={self.npw}, k={self.k_cart})"


def create_fft_workspace(gvec, factor=2.0):
    """
    Create FFT workspace arrays.
    
    Args:
        gvec: GVector object
        factor: FFT grid size factor (2.0 for products)
    
    Returns:
        fft_shape: (n1, n2, n3) grid dimensions
        work_r: Real-space workspace array
        work_g: G-space workspace array
    """
    fft_shape = gvec.get_fft_grid_size(factor)
    work_r = np.zeros(fft_shape, dtype=complex)
    work_g = np.zeros(fft_shape, dtype=complex)
    return fft_shape, work_r, work_g
