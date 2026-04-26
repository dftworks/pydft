"""
G-vector generation for plane wave calculations.

G-vectors are reciprocal lattice vectors used as the plane wave basis.

=============================================================================
PEDAGOGICAL NOTES
=============================================================================

THE PLANE WAVE BASIS
--------------------
In a periodic crystal, wavefunctions satisfy Bloch's theorem:
    ψ_{nk}(r) = e^{ik·r} u_{nk}(r)

where u_{nk}(r) has the periodicity of the lattice. We expand u in plane waves:
    u_{nk}(r) = Σ_G c_{nk}(G) e^{iG·r}

So the full wavefunction is:
    ψ_{nk}(r) = Σ_G c_{nk}(G) e^{i(k+G)·r}

G-VECTORS ARE RECIPROCAL LATTICE VECTORS
----------------------------------------
The reciprocal lattice vectors are:
    G = m₁b₁ + m₂b₂ + m₃b₃

where b_i are the reciprocal lattice basis vectors and m_i are integers
(called Miller indices).

THE ENERGY CUTOFF
-----------------
We can't include infinitely many plane waves! The kinetic energy of |k+G⟩ is:
    T(G) = ½|k+G|²

We truncate the basis by keeping only G-vectors with:
    ½|k+G|² < E_cut  (or ½|G|² < E_cut for Gamma point)

This gives a FINITE, systematically improvable basis:
- Higher E_cut → more plane waves → better accuracy
- Typical values: 20-100 Ry (1 Ry = 13.6 eV)

WHY SORT BY |G|?
----------------
G=0 should be first because:
- V_H(G=0) is handled specially (charge neutrality)
- ρ(G=0) is the average density
- Makes coding cleaner

MAPPING TO FFT GRIDS
--------------------
FFT requires a regular 3D grid of size (N₁, N₂, N₃). Our G-vectors are a
SUBSET of this grid (those within cutoff). We need functions to:
- map_to_fft_grid: G-vector list → 3D FFT array
- map_from_fft_grid: 3D FFT array → G-vector list

Negative Miller indices wrap around: m = -1 maps to index N-1.

CROSS-REFERENCE: 
See `pydft-book`, Chapter "Crystal Lattice", Section "G Space" (`gspace.tex`)
and Chapter "Basis", Section "FFT Grid" (`fftgrid.tex`) for detailed derivations.
=============================================================================
"""

import numpy as np
from .lattice import Lattice


class GVector:
    """
    G-vector grid for plane wave basis.
    
    Generates all G-vectors within a kinetic energy cutoff and provides
    mapping to/from FFT grids.
    
    Attributes:
        lattice: Lattice object
        ecut: Energy cutoff in Hartree
        miller: (N, 3) array of Miller indices
        cart: (N, 3) array of Cartesian G-vectors
        norms: (N,) array of |G| values
        npw: Number of plane waves
    """
    
    def __init__(self, lattice, ecut):
        """
        Generate G-vectors within energy cutoff.
        
        Args:
            lattice: Lattice object
            ecut: Energy cutoff in Hartree (kinetic energy: 0.5*|G|^2 < ecut)
        """
        self.lattice = lattice
        self.ecut = ecut
        
        # Generate G-vectors
        self._generate_gvectors()
    
    def _generate_gvectors(self):
        """
        Generate all G-vectors with 0.5*|G|^2 < ecut.

        PEDAGOGICAL NOTE: The Algorithm
        --------------------------------
        1. Compute G_max from energy cutoff: ½|G_max|² = E_cut → |G_max| = √(2E_cut)
        2. Estimate maximum Miller indices needed
        3. Loop over all (m₁, m₂, m₃) within bounds
        4. For each, compute G = m₁b₁ + m₂b₂ + m₃b₃
        5. Keep only those with ½|G|² < E_cut
        6. Sort by |G| so that G=0 is first
        """
        b = self.lattice.reciprocal_vectors

        # Maximum |G| from energy cutoff: ½|G|² = E_cut → |G| = √(2E_cut)
        gmax = np.sqrt(2.0 * self.ecut)

        # Estimate max Miller indices using triangle inequality:
        # |G| = |m₁b₁ + m₂b₂ + m₃b₃| ≤ |m₁||b₁| + |m₂||b₂| + |m₃||b₃|
        # So |m_i| ≤ G_max / |b_i| is a safe upper bound
        b_norms = np.linalg.norm(b, axis=1)
        n_max = (gmax / b_norms).astype(int) + 1

        # Generate all candidate Miller indices
        miller_list = []
        cart_list = []
        norm_list = []

        # Triple loop over all possible Miller indices
        # This is O(N_max³) but N_max is typically small (~10-30)
        for n1 in range(-n_max[0], n_max[0] + 1):
            for n2 in range(-n_max[1], n_max[1] + 1):
                for n3 in range(-n_max[2], n_max[2] + 1):
                    # G = m₁b₁ + m₂b₂ + m₃b₃  (Cartesian coordinates)
                    g_cart = n1 * b[0] + n2 * b[1] + n3 * b[2]
                    g_norm = np.linalg.norm(g_cart)

                    # Apply energy cutoff: keep only if ½|G|² < E_cut
                    if 0.5 * g_norm**2 < self.ecut:
                        miller_list.append([n1, n2, n3])
                        cart_list.append(g_cart)
                        norm_list.append(g_norm)

        # Convert to NumPy arrays for efficient operations
        self.miller = np.array(miller_list, dtype=int)
        self.cart = np.array(cart_list)
        self.norms = np.array(norm_list)
        self.npw = len(self.norms)  # Number of plane waves

        # Sort by |G| so that G=0 is first
        # This is important for Hartree potential (V_H(G=0) = 0)
        order = np.argsort(self.norms)
        self.miller = self.miller[order]
        self.cart = self.cart[order]
        self.norms = self.norms[order]

        # G=0 should now be at index 0 (verify with g0_index)
        self.g0_index = np.where(self.norms < 1e-10)[0][0]
    
    def get_fft_grid_size(self, factor=2.0):
        """
        Determine FFT grid size to avoid aliasing.

        For proper representation of products (like V*psi), the FFT grid
        must be at least 2x the maximum Miller index in each direction.

        Args:
            factor: Safety factor (default 2.0 for products)

        Returns:
            (n1, n2, n3): FFT grid dimensions

        PEDAGOGICAL NOTE: Why Factor of 2?
        -----------------------------------
        When we compute V(r)×ψ(r) in real space and FFT back to G-space,
        the product can have G-vectors up to 2×G_max (convolution theorem).

        If the FFT grid is too small, these high-G components "wrap around"
        and contaminate low-G components. This is called ALIASING.

        To avoid aliasing for products, we need:
            N_i ≥ 2 × max(|m_i|)

        Production codes may use factor=3 or 4 for higher accuracy.

        WHY POWERS OF 2?
        ----------------
        FFT is fastest for sizes that are powers of 2 (or products of small
        primes like 2, 3, 5). Rounding up to the next power of 2 trades a
        bit of memory for faster computation.
        """
        # Find maximum Miller index in each direction
        max_miller = np.max(np.abs(self.miller), axis=0)

        # FFT size should be at least 2×max + 1, rounded up to next suitable FFT size
        n_fft = (factor * max_miller + 1).astype(int)

        # Round up to next power of 2 or product of small primes for efficiency
        n_fft = np.array([self._next_fft_size(n) for n in n_fft])

        return tuple(n_fft)
    
    @staticmethod
    def _next_fft_size(n):
        """Find next efficient FFT size >= n."""
        # For simplicity, round up to next power of 2
        return int(2 ** np.ceil(np.log2(max(n, 4))))
    
    def map_to_fft_grid(self, data_g, fft_shape):
        """
        Map 1D G-space data to 3D FFT grid.
        
        Args:
            data_g: (npw,) array of data in G-space
            fft_shape: (n1, n2, n3) FFT grid dimensions
        
        Returns:
            (n1, n2, n3) complex array on FFT grid
        """
        n1, n2, n3 = fft_shape
        grid = np.zeros(fft_shape, dtype=complex)
        
        for ig, (m1, m2, m3) in enumerate(self.miller):
            # Handle negative indices (wrap around)
            i1 = m1 % n1
            i2 = m2 % n2
            i3 = m3 % n3
            grid[i1, i2, i3] = data_g[ig]
        
        return grid
    
    def map_from_fft_grid(self, grid):
        """
        Extract 1D G-space data from 3D FFT grid.
        
        Args:
            grid: (n1, n2, n3) complex array on FFT grid
        
        Returns:
            (npw,) array of data in G-space
        """
        n1, n2, n3 = grid.shape
        data_g = np.zeros(self.npw, dtype=complex)
        
        for ig, (m1, m2, m3) in enumerate(self.miller):
            i1 = m1 % n1
            i2 = m2 % n2
            i3 = m3 % n3
            data_g[ig] = grid[i1, i2, i3]
        
        return data_g
    
    def get_kinetic_energies(self, k_cart=None):
        """
        Get kinetic energies |k+G|^2/2 for each G-vector.
        
        Args:
            k_cart: k-point in Cartesian coordinates (default: Gamma point)
        
        Returns:
            (npw,) array of kinetic energies
        """
        if k_cart is None:
            k_cart = np.zeros(3)
        
        kg = self.cart + k_cart
        return 0.5 * np.sum(kg**2, axis=1)
    
    def __repr__(self):
        return f"GVector(npw={self.npw}, ecut={self.ecut:.2f} Ha)"
