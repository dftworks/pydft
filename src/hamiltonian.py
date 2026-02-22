"""
Hamiltonian application: H|psi>.

The Kohn-Sham Hamiltonian consists of:
- Kinetic energy: T = -1/2 * nabla^2
- Local potential: V_loc = V_H + V_xc + V_ps_loc
- Non-local potential: V_nl (applied externally by caller-specific wrappers)

=============================================================================
PEDAGOGICAL NOTES
=============================================================================

THE CENTRAL OPERATION IN DFT: Applying H|ψ⟩

The Kohn-Sham Hamiltonian in atomic units (ℏ = m = e = 1):

    H = T + V_loc + V_nl

where:
    T = -½∇²           (kinetic energy)
    V_loc = V_H + V_xc + V_ps_loc  (local potentials)
    V_nl = Σ|β⟩D⟨β|    (non-local pseudopotential, Kleinman-Bylander form)

PLANE WAVE REPRESENTATION
-------------------------
In a plane wave basis, wavefunctions are:
    ψ(r) = Σ_G c(G) e^{iG·r} / √Ω

where c(G) are the expansion coefficients stored as a 1D array.

KEY INSIGHT: Different operators are diagonal in different spaces!
    - Kinetic T: diagonal in G-space → T|G⟩ = ½|G|² |G⟩
    - Local V_loc: diagonal in r-space → V|r⟩ = V(r)|r⟩

THE MIXED-SPACE METHOD (used here):
    1. Start with ψ(G) in G-space
    2. For kinetic: simply multiply by ½|G|²
    3. For V_loc:
       a. FFT⁻¹: ψ(G) → ψ(r)  [go to real space]
       b. Multiply: V(r)·ψ(r) [apply potential]
       c. FFT: V·ψ(r) → (Vψ)(G) [back to G-space]
    4. Sum: H|ψ⟩ = T|ψ⟩ + V_loc|ψ⟩

This is O(N log N) thanks to FFT, vs O(N²) for direct matrix multiplication!
=============================================================================
"""

import numpy as np


class Hamiltonian:
    """
    Kohn-Sham Hamiltonian in plane wave basis.
    
    Applies H|psi> = (T + V_loc)|psi> using FFT for the local potential.
    
    Attributes:
        gvec: GVector object
        fft_shape: FFT grid dimensions
        volume: Cell volume
    """
    
    def __init__(self, gvec, volume):
        """
        Initialize Hamiltonian.

        Args:
            gvec: GVector object
            volume: Cell volume

        PEDAGOGICAL NOTE: Kinetic Energy in G-Space
        -------------------------------------------
        The kinetic operator T = -½∇² acts on plane waves as:

            T e^{iG·r} = -½∇² e^{iG·r}
                       = -½(iG)·(iG) e^{iG·r}
                       = ½|G|² e^{iG·r}

        So T is DIAGONAL in the plane wave basis with eigenvalue ½|G|²!
        This makes applying the kinetic energy trivial: just multiply.

        For k-points (Bloch's theorem): T|k+G⟩ = ½|k+G|² |k+G⟩
        """
        self.gvec = gvec
        self.volume = volume
        self.fft_shape = gvec.get_fft_grid_size()
        self.n_fft = np.prod(self.fft_shape)

        # Kinetic energy: diagonal in G-space
        # For Gamma point (k=0): T_GG' = ½|G|² δ_{GG'}
        # For general k: T_GG' = ½|k+G|² δ_{GG'}
        self.kg = 0.5 * np.sum(gvec.cart**2, axis=1)

        # Local potential in real space (set later by SCF)
        self.vloc_r = None

        # Workspace arrays (pre-allocated for efficiency)
        # These avoid repeated memory allocation in the inner loop
        self._work_g = np.zeros(self.fft_shape, dtype=complex)
        self._work_r = np.zeros(self.fft_shape, dtype=complex)
    
    def set_local_potential(self, vloc_r):
        """
        Set the local potential in real space.
        
        Args:
            vloc_r: Local potential V_loc(r) on FFT grid
        """
        self.vloc_r = vloc_r
    
    def apply(self, psi_g):
        """
        Apply Hamiltonian to wavefunction: H|psi>.
        
        H = T + V_loc
        
        Args:
            psi_g: Wavefunction in G-space (npw,) complex array
        
        Returns:
            h_psi_g: H|psi> in G-space
        """
        h_psi_g = np.zeros_like(psi_g, dtype=complex)
        
        # Kinetic: T|psi> = (|G|^2 / 2) * psi(G)
        h_psi_g += self.kg * psi_g
        
        # Local potential: V_loc|psi> via FFT
        if self.vloc_r is not None:
            h_psi_g += self._apply_vloc(psi_g)
        
        return h_psi_g
    
    def _apply_vloc(self, psi_g):
        """
        Apply local potential via FFT (mixed-space method).

        1. Map psi(G) to FFT grid
        2. IFFT to get psi(r)
        3. Multiply: V_loc(r) * psi(r)
        4. FFT to get (V_loc * psi)(G)
        5. Map back to G-vector list

        PEDAGOGICAL NOTE: Why FFT for Potential Application?
        -----------------------------------------------------
        The matrix element ⟨G|V|G'⟩ for a local potential is:

            V_{GG'} = (1/Ω) ∫ V(r) e^{-i(G-G')·r} dr = V(G-G')

        This is a CONVOLUTION in G-space, which is O(N²) naively.

        But convolution in G-space = multiplication in r-space!
            (V·ψ)(G) = FFT[ V(r) × FFT⁻¹[ψ(G)] ]

        This reduces O(N²) to O(N log N) - a huge speedup!

        FFT NORMALIZATION (tricky but important):
        -----------------------------------------
        Physics convention:  ψ(r) = (1/√Ω) Σ_G c(G) e^{iG·r}
        NumPy FFT:           ifftn(c) = (1/N) Σ_G c(G) e^{iG·r}

        The factors of √Ω and N must be carefully tracked!

        Our approach:
            1. ifftn gives (1/N)Σc(G)e^{iGr} → multiply by N
            2. After real-space multiply, divide by √Ω for normalization
            3. fftn gives Σ[...]e^{-iGr} → divide by N
            4. Multiply by √Ω to restore c(G) normalization
        """
        # ===== STEP 1: Map ψ(G) to 3D FFT grid =====
        # Our G-vectors are a subset (within cutoff) of the full FFT grid
        self._work_g.fill(0)
        self._map_to_fft(psi_g, self._work_g)

        # ===== STEP 2: IFFT to real space =====
        # NumPy ifftn divides by N; we multiply by N to undo this
        # Result: ψ(r) on the real-space grid (unnormalized)
        self._work_r = np.fft.ifftn(self._work_g) * self.n_fft

        # ===== STEP 3: Multiply by potential in real space =====
        # This is where the "magic" happens - O(N) instead of O(N²)!
        # The 1/√Ω factor handles the plane wave normalization
        self._work_r *= self.vloc_r / np.sqrt(self.volume)

        # ===== STEP 4: FFT back to G-space =====
        # Divide by N to match our convention
        self._work_g = np.fft.fftn(self._work_r) / self.n_fft

        # ===== STEP 5: Extract result for our G-vectors =====
        vloc_psi_g = self._map_from_fft(self._work_g)

        # Restore √Ω factor for final result
        return vloc_psi_g * np.sqrt(self.volume)
    
    def _map_to_fft(self, data_g, grid):
        """
        Map 1D G-space data to 3D FFT grid.
        
        Uses an explicit loop over G-vectors for pedagogical clarity.
        A production code would use vectorised fancy indexing for speed.
        """
        n1, n2, n3 = self.fft_shape
        for ig, (m1, m2, m3) in enumerate(self.gvec.miller):
            i1 = m1 % n1
            i2 = m2 % n2
            i3 = m3 % n3
            grid[i1, i2, i3] = data_g[ig]
    
    def _map_from_fft(self, grid):
        """
        Extract 1D G-space data from 3D FFT grid.
        
        Mirrors _map_to_fft; also uses an explicit loop for clarity.
        """
        n1, n2, n3 = self.fft_shape
        data_g = np.zeros(self.gvec.npw, dtype=complex)
        for ig, (m1, m2, m3) in enumerate(self.gvec.miller):
            i1 = m1 % n1
            i2 = m2 % n2
            i3 = m3 % n3
            data_g[ig] = grid[i1, i2, i3]
        return data_g
    
    def get_diagonal(self):
        """
        Get diagonal of Hamiltonian (kinetic energy).
        
        Used for preconditioning in iterative eigensolvers.
        
        Returns:
            (npw,) array of diagonal elements
        """
        return self.kg.copy()
    
    def compute_expectation(self, psi_g):
        """
        Compute <psi|H|psi>.
        
        Args:
            psi_g: Normalized wavefunction in G-space
        
        Returns:
            Expectation value (energy)
        """
        h_psi = self.apply(psi_g)
        return np.real(np.vdot(psi_g, h_psi))


def g_to_r(psi_g, gvec, fft_shape, volume):
    """
    Transform wavefunction from G-space to real space.

    psi(r) = (1/sqrt(V)) * sum_G psi(G) * exp(i*G*r)

    Args:
        psi_g: Wavefunction in G-space
        gvec: GVector object
        fft_shape: FFT grid dimensions
        volume: Cell volume

    Returns:
        psi_r: Wavefunction on real-space grid

    PEDAGOGICAL NOTE: Wavefunction Normalization
    ---------------------------------------------
    In a periodic cell, wavefunctions are normalized as:
        ∫_cell |ψ(r)|² dr = 1

    With plane wave expansion:
        ψ(r) = (1/√Ω) Σ_G c(G) e^{iG·r}

    The 1/√Ω factor ensures:
        ∫|ψ|²dr = (1/Ω) ∫ |Σ_G c(G) e^{iG·r}|² dr = Σ_G |c(G)|² = 1

    (The last step uses orthogonality of plane waves)
    """
    n1, n2, n3 = fft_shape
    n_fft = n1 * n2 * n3

    # Map coefficients c(G) to the 3D FFT grid
    # Negative Miller indices wrap around via modulo operation
    work = np.zeros(fft_shape, dtype=complex)
    for ig, (m1, m2, m3) in enumerate(gvec.miller):
        i1 = m1 % n1  # e.g., -1 % 8 = 7 (wraps to end)
        i2 = m2 % n2
        i3 = m3 % n3
        work[i1, i2, i3] = psi_g[ig]

    # NumPy ifftn: (1/N) Σ_G c(G) e^{iG·r}
    # We want:     (1/√Ω) Σ_G c(G) e^{iG·r}
    # So: multiply by N, divide by √Ω
    psi_r = np.fft.ifftn(work) * n_fft / np.sqrt(volume)

    return psi_r


def r_to_g(psi_r, gvec, fft_shape, volume):
    """
    Transform wavefunction from real space to G-space.
    
    psi(G) = (1/sqrt(V)) * integral psi(r) * exp(-i*G*r) dr
    
    Args:
        psi_r: Wavefunction on real-space grid
        gvec: GVector object
        fft_shape: FFT grid dimensions
        volume: Cell volume
    
    Returns:
        psi_g: Wavefunction in G-space
    """
    n1, n2, n3 = fft_shape
    n_fft = n1 * n2 * n3
    
    # Inverse normalization of g_to_r.
    work = np.fft.fftn(psi_r * np.sqrt(volume)) / n_fft
    
    # Map to G-vector list
    psi_g = np.zeros(gvec.npw, dtype=complex)
    for ig, (m1, m2, m3) in enumerate(gvec.miller):
        i1 = m1 % n1
        i2 = m2 % n2
        i3 = m3 % n3
        psi_g[ig] = work[i1, i2, i3]
    
    return psi_g
