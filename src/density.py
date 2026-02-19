"""
Electron Density Computation.

Computes the electron density from Kohn-Sham wavefunctions,
including support for k-point sampling and spin polarization.

The density is:
    rho(r) = sum_{n,k} w_k * f_{n,k} * |psi_{n,k}(r)|^2

where w_k are k-point weights and f_{n,k} are occupation numbers.
"""

import numpy as np
from .hamiltonian import g_to_r


def compute_density(evecs, occupations, gvector, fft_shape, volume, 
                    k_weights=1.0):
    """
    Compute electron density from wavefunctions.
    
    Args:
        evecs: Wavefunctions in G-space, shape (npw, nbands) or 
               (nk, npw, nbands) for multiple k-points
        occupations: Occupation numbers, shape (nbands,) or (nk, nbands)
        gvector: GVector object
        fft_shape: FFT grid shape (n1, n2, n3)
        volume: Cell volume
        k_weights: K-point weights, scalar or array of shape (nk,)
    
    Notes:
        Occupations are expected in the "spin-paired" convention used elsewhere
        in this codebase (0..2 per spatial orbital).

    Returns:
        rho_r: Electron density in real space, shape fft_shape
    """
    evecs = np.asarray(evecs)
    occupations = np.asarray(occupations)
    if evecs.ndim not in (2, 3):
        raise ValueError(
            f"evecs must have ndim 2 or 3, got shape {evecs.shape}"
        )

    rho_r = np.zeros(fft_shape, dtype=float)
    
    # Handle single k-point case
    if evecs.ndim == 2:
        if occupations.ndim != 1:
            raise ValueError(
                "For single-k input, occupations must have shape (nbands,)."
            )
        nbands = evecs.shape[1]
        if occupations.shape[0] != nbands:
            raise ValueError(
                f"Occupation size mismatch: {occupations.shape[0]} vs nbands={nbands}."
            )
        
        for iband in range(nbands):
            if occupations[iband] < 1e-10:
                continue
            
            # psi(G) -> psi(r), then accumulate f_n |psi_n(r)|^2.
            psi_r = g_to_r(evecs[:, iband], gvector, fft_shape, volume)
            
            # Add to density
            rho_r += occupations[iband] * np.abs(psi_r)**2
    
    else:
        # Multiple k-points
        nk = evecs.shape[0]
        nbands = evecs.shape[2]
        if occupations.ndim == 2 and occupations.shape != (nk, nbands):
            raise ValueError(
                "For multi-k input, occupations must be (nk, nbands) "
                f"or (nbands,); got {occupations.shape}."
            )
        if occupations.ndim == 1 and occupations.shape[0] != nbands:
            raise ValueError(
                f"Occupation size mismatch: {occupations.shape[0]} vs nbands={nbands}."
            )
        
        if np.isscalar(k_weights):
            # Scalar weights are interpreted as "equal total weight".
            k_weights = np.ones(nk) * k_weights / nk
        else:
            k_weights = np.asarray(k_weights)
            if k_weights.shape != (nk,):
                raise ValueError(
                    f"k_weights must have shape ({nk},), got {k_weights.shape}."
                )
        
        for ik in range(nk):
            for iband in range(nbands):
                occ = occupations[ik, iband] if occupations.ndim > 1 else occupations[iband]
                
                if occ < 1e-10:
                    continue
                
                # Weighted k-point contribution: w_k f_nk |psi_nk(r)|^2.
                psi_r = g_to_r(evecs[ik, :, iband], gvector, fft_shape, volume)
                
                # Add to density with k-weight
                rho_r += k_weights[ik] * occ * np.abs(psi_r)**2
    
    return rho_r


def compute_density_g(rho_r, gvector, fft_shape):
    """
    Transform density from real space to G-space.
    
    Args:
        rho_r: Density in real space
        gvector: GVector object
        fft_shape: FFT grid shape
    
    Returns:
        rho_g: Density in G-space (complex array)
    """
    n_fft = np.prod(fft_shape)
    
    # FFT
    rho_fft = np.fft.fftn(rho_r) / n_fft
    
    # Map to G-vector list
    rho_g = gvector.map_from_fft_grid(rho_fft)
    
    return rho_g


def initial_density_uniform(n_electrons, volume, fft_shape):
    """
    Create uniform initial density.
    
    Args:
        n_electrons: Number of electrons
        volume: Cell volume
        fft_shape: FFT grid shape
    
    Returns:
        rho_r: Uniform density in real space
    """
    rho_0 = n_electrons / volume
    return np.full(fft_shape, rho_0, dtype=float)


def initial_density_atomic(atom_species, atom_positions, gvector, fft_shape, volume):
    """
    Create initial density from atomic superposition.
    
    This is a simplified version using Gaussian atom densities.
    
    Args:
        atom_species: List of AtomicSpecies
        atom_positions: Fractional coordinates
        gvector: GVector object
        fft_shape: FFT grid shape
        volume: Cell volume
    
    Returns:
        rho_r: Atomic superposition density in real space
    """
    from .lattice import Lattice
    
    n1, n2, n3 = fft_shape
    rho_r = np.zeros(fft_shape, dtype=float)
    
    # Real-space grid points
    x = np.linspace(0, 1, n1, endpoint=False)
    y = np.linspace(0, 1, n2, endpoint=False)
    z = np.linspace(0, 1, n3, endpoint=False)
    
    # For each atom, add Gaussian charge distribution
    for sp, pos in zip(atom_species, atom_positions):
        zion = sp.zion
        
        # Gaussian width (rough estimate from atomic radius)
        sigma = 1.0  # Bohr
        
        for i, xi in enumerate(x):
            for j, yj in enumerate(y):
                for k, zk in enumerate(z):
                    # Fractional displacement (with periodic images)
                    dx = xi - pos[0]
                    dy = yj - pos[1]
                    dz = zk - pos[2]
                    
                    # Wrap to nearest image
                    dx = dx - np.round(dx)
                    dy = dy - np.round(dy)
                    dz = dz - np.round(dz)
                    
                    # This is a simplification - actual atomic density
                    # would need proper unit cell handling
                    r2 = (dx**2 + dy**2 + dz**2) * 100  # Rough scaling
                    
                    rho_r[i,j,k] += zion * np.exp(-r2 / (2*sigma**2)) / (2*np.pi*sigma**2)**1.5
    
    # Normalize to total electrons
    total = np.sum(rho_r) * volume / np.prod(fft_shape)
    n_electrons = sum(sp.zion for sp in atom_species)
    
    if total > 1e-10:
        rho_r *= n_electrons / total
    
    return rho_r


def compute_density_gradient(rho_r, gvector, fft_shape, lattice):
    """
    Compute gradient of density (for GGA functionals).
    
    Args:
        rho_r: Density in real space
        gvector: GVector object
        fft_shape: FFT grid shape
        lattice: Lattice object
    
    Returns:
        grad_rho: Gradient components (3, n1, n2, n3)
        grad_rho_mag: |grad(rho)|, shape (n1, n2, n3)
    """
    n_fft = np.prod(fft_shape)
    
    # FFT to G-space
    rho_fft = np.fft.fftn(rho_r)
    
    # Gradient in G-space: i*G * rho(G)
    grad_rho = np.zeros((3,) + fft_shape, dtype=float)
    
    for ig in range(gvector.npw):
        m = gvector.miller[ig]
        G = gvector.gvecs[ig]
        
        # Get FFT indices
        i1 = m[0] % fft_shape[0]
        i2 = m[1] % fft_shape[1]
        i3 = m[2] % fft_shape[2]
        
        # i * G * rho(G)
        for d in range(3):
            grad_rho[d] += np.real(1j * G[d] * rho_fft[i1, i2, i3] * 
                                   np.exp(2j * np.pi * (m[0]*np.arange(fft_shape[0])[:, None, None]/fft_shape[0] +
                                                        m[1]*np.arange(fft_shape[1])[None, :, None]/fft_shape[1] +
                                                        m[2]*np.arange(fft_shape[2])[None, None, :]/fft_shape[2])))
    
    # Magnitude
    grad_rho_mag = np.sqrt(np.sum(grad_rho**2, axis=0))
    
    return grad_rho, grad_rho_mag


def integrate_density(rho_r, volume, fft_shape):
    """
    Integrate density to get total electrons.
    
    Args:
        rho_r: Density in real space
        volume: Cell volume
        fft_shape: FFT grid shape
    
    Returns:
        Total number of electrons
    """
    dr = volume / np.prod(fft_shape)
    return np.sum(rho_r) * dr


def check_charge_neutrality(rho_r, n_electrons, volume, fft_shape, tol=0.01):
    """
    Check if integrated density matches expected electron count.
    
    Args:
        rho_r: Density in real space
        n_electrons: Expected number of electrons
        volume: Cell volume
        fft_shape: FFT grid shape
        tol: Tolerance for charge mismatch
    
    Returns:
        is_neutral: True if charge matches
        n_computed: Computed number of electrons
    """
    n_computed = integrate_density(rho_r, volume, fft_shape)
    is_neutral = abs(n_computed - n_electrons) < tol
    
    return is_neutral, n_computed
