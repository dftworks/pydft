"""
Local pseudopotential helpers for plane-wave DFT.

This module focuses on local ionic potentials used in educational and
toy-system workflows:
- radial local potential transforms V_loc(r) -> V_loc(G)
- simple analytic model potentials
- structure-factor assembly over atoms in the unit cell

References:
    - Kleinman, L. & Bylander, D. M. Phys. Rev. Lett. 48, 1425 (1982)
    - Hamann, D. R. et al. Phys. Rev. Lett. 43, 1494 (1979)
"""

import numpy as np
from scipy.special import erf
from scipy.integrate import simpson
from .constants import TWOPI, FOURPI


class LocalPotential:
    """
    Local pseudopotential in G-space.
    
    The local potential is computed as:
    
    V_loc(G) = (4*pi/V) * int_0^inf [r*V_loc(r) + Z*erf(r)] * sin(Gr)/G * dr
    
    for G != 0. For G = 0, we subtract the divergent Coulomb term which
    cancels with the Hartree G=0 term.
    """
    
    def __init__(self, rad, vloc, zion, volume):
        """
        Initialize local potential.
        
        Args:
            rad: Radial grid points (Bohr)
            vloc: V_loc(r) values on radial grid (Hartree)
            zion: Ionic charge
            volume: Cell volume (Bohr^3)
        """
        self.rad = np.array(rad)
        self.vloc = np.array(vloc)
        self.zion = zion
        self.volume = volume
    
    def compute_vloc_g(self, gshells):
        """
        Compute V_loc(G) on G-shells.
        
        Args:
            gshells: Array of unique |G| values
        
        Returns:
            vloc_g: V_loc(G) values for each shell
        """
        nshells = len(gshells)
        vloc_g = np.zeros(nshells)
        
        r = self.rad
        vloc = self.vloc
        zion = self.zion
        
        # G = 0: integrate r * (r*V_loc + Z)
        work = r * (r * vloc + zion)
        vloc_g[0] = self._integrate_radial(work)
        
        # G > 0
        for ig in range(1, nshells):
            g = gshells[ig]
            g2 = g * g
            
            # Integrand: (r*V_loc + Z*erf(r)) * sin(G*r) / G
            work = (r * vloc + zion * erf(r)) * np.sin(g * r) / g
            
            # Subtract divergent hydrogen-like term
            vh = zion * np.exp(-g2 / 4.0) / g2
            
            vloc_g[ig] = self._integrate_radial(work) - vh
        
        # Apply prefactor
        vloc_g *= FOURPI / self.volume
        
        return vloc_g
    
    def _integrate_radial(self, work):
        """Integrate function on radial grid using Simpson's rule."""
        return simpson(work, x=self.rad)


class SimplePseudopotential:
    """
    Simple pseudopotential for educational purposes.
    
    Uses an analytical local potential form suitable for tests and
    examples that do not rely on UPF files.
    """
    
    def __init__(self, zion, r_loc=1.0, alpha=1.0):
        """
        Create simple local pseudopotential.
        
        V_loc(r) = -Z/r * erf(r/r_loc)
        
        This is a smoothed Coulomb potential that is finite at r=0.
        
        Args:
            zion: Ionic charge
            r_loc: Localization radius
            alpha: Decay parameter
        """
        self.zion = zion
        self.r_loc = r_loc
        self.alpha = alpha
        
        # Generate radial grid
        self.rad = np.linspace(0.01, 15.0, 1000)
        
        # Compute V_loc on grid
        self.vloc = -zion / self.rad * erf(self.rad / r_loc)
    
    def get_vloc_g(self, gshells, volume):
        """Get V_loc(G) for G-shells."""
        loc_pot = LocalPotential(self.rad, self.vloc, self.zion, volume)
        return loc_pot.compute_vloc_g(gshells)


def create_periodic_gaussian_model_potential(
    lattice, atom_positions, fft_shape, depth=1.0, r_c=1.0
):
    """Build a smooth periodic sum of short-range Gaussian model wells.

    Each atom contributes ``-depth * exp(-|r-R|^2/r_c^2)`` over all lattice
    translations. Its Fourier coefficient is

        -depth * pi^(3/2) * r_c^3 / Omega
        * exp(-|G|^2*r_c^2/4) * sum_a exp(-i G.R_a).

    Args:
        lattice: Cell lattice in Bohr
        atom_positions: Fractional atomic positions, shape (natoms, 3)
        fft_shape: Three-dimensional FFT grid shape
        depth: Positive well depth in Hartree
        r_c: Gaussian radius in Bohr

    Returns:
        Real-space periodic potential sampled on ``fft_shape``.
    """
    if not np.isfinite(depth) or depth < 0:
        raise ValueError("depth must be finite and nonnegative.")
    if not np.isfinite(r_c) or r_c <= 0:
        raise ValueError("r_c must be finite and positive.")
    fft_shape = tuple(fft_shape)
    if (len(fft_shape) != 3 or any(
            not isinstance(n, (int, np.integer)) or n < 1 for n in fft_shape)):
        raise ValueError("fft_shape must contain three positive integer dimensions.")
    atom_positions = np.asarray(atom_positions, dtype=float)
    if atom_positions.ndim != 2 or atom_positions.shape[1] != 3:
        raise ValueError("atom_positions must have shape (natoms, 3).")
    if not np.isfinite(atom_positions).all():
        raise ValueError("atom_positions must contain only finite values.")

    miller_axes = [np.fft.fftfreq(n) * n for n in fft_shape]
    miller = np.stack(np.meshgrid(*miller_axes, indexing="ij"), axis=-1)
    g_cart = miller @ lattice.reciprocal_vectors
    g2 = np.sum(g_cart**2, axis=-1)

    structure_factor = np.zeros(fft_shape, dtype=complex)
    for atom_position in atom_positions:
        phase = TWOPI * np.einsum("...i,i->...", miller, atom_position)
        structure_factor += np.exp(-1j * phase)
    v_g = (-depth * np.pi**1.5 * r_c**3 / lattice.volume
           * np.exp(-g2 * r_c**2 / 4.0) * structure_factor)

    return np.fft.ifftn(v_g * np.prod(fft_shape)).real


class AtomicSpecies:
    """
    Represents an atomic species with its pseudopotential.
    
    Attributes:
        symbol: Element symbol (e.g., 'Si')
        zion: Ionic charge
        mass: Atomic mass (optional)
        psp: Local potential model object
        has_nonlocal: Whether a non-local model is attached
    """
    
    def __init__(self, symbol, zion, mass=None):
        """
        Initialize atomic species.
        
        Args:
            symbol: Element symbol
            zion: Ionic charge
            mass: Atomic mass in atomic units (optional)
        """
        self.symbol = symbol
        self.zion = zion
        self.mass = mass
        
        # Create simple pseudopotential
        self.psp = SimplePseudopotential(zion)
        
        # No non-local potential in simple case
        self.has_nonlocal = False
        self.vnl = None
    
    def get_vloc_g(self, gshells, volume):
        """Get local potential in G-space."""
        return self.psp.get_vloc_g(gshells, volume)


def compute_local_potential_g(atom_species, atom_positions, gvector, volume):
    """
    Compute total local pseudopotential in G-space.
    
    V_loc^tot(G) = sum_atoms V_loc(G) * S(G)
    
    where S(G) = exp(-i*G*r_atom) is the structure factor.
    
    Args:
        atom_species: List of AtomicSpecies for each atom
        atom_positions: Fractional coordinates, shape (natoms, 3)
        gvector: GVector object
        volume: Cell volume
    
    Returns:
        vloc_g: Complex array, shape (npw,)
    """
    npw = gvector.npw
    vloc_g = np.zeros(npw, dtype=complex)
    
    # Get G-shell information
    gshells = np.unique(gvector.norms)
    gshell_map = {g: i for i, g in enumerate(gshells)}
    
    # Compute V_loc(G) for each species type
    species_vloc = {}
    for sp in atom_species:
        if sp.symbol not in species_vloc:
            species_vloc[sp.symbol] = sp.get_vloc_g(gshells, volume)
    
    # Sum over atoms with structure factor
    miller = gvector.miller
    
    for iat, (sp, pos) in enumerate(zip(atom_species, atom_positions)):
        vloc_species = species_vloc[sp.symbol]
        
        for ig in range(npw):
            m = miller[ig]
            g_norm = gvector.norms[ig]
            
            # Find shell index
            shell_idx = gshell_map.get(g_norm, 0)
            
            # Structure factor phase
            phase = TWOPI * (m[0]*pos[0] + m[1]*pos[1] + m[2]*pos[2])
            sfact = np.exp(-1j * phase)
            
            vloc_g[ig] += vloc_species[shell_idx] * sfact
    
    return vloc_g


def create_jellium_potential(volume):
    """
    Create a jellium (uniform background) potential.
    
    In jellium, V_ext = 0 (uniform positive background cancels
    the G=0 Hartree term).
    
    Returns:
        Callable that returns zero potential in G-space
    """
    def vloc_g(gvector):
        return np.zeros(gvector.npw, dtype=complex)
    
    return vloc_g


def create_harmonic_potential(lattice, omega=0.1):
    """
    Create a harmonic confining potential in real space.
    
    V(r) = 0.5 * omega^2 * |r - r_center|^2
    
    Args:
        lattice: Lattice object
        omega: Harmonic frequency
    
    Returns:
        Function that generates V(r) on FFT grid
    """
    def v_harmonic(fft_shape):
        n1, n2, n3 = fft_shape
        
        # Fractional coordinates
        x = np.linspace(0, 1, n1, endpoint=False)
        y = np.linspace(0, 1, n2, endpoint=False)
        z = np.linspace(0, 1, n3, endpoint=False)
        
        X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
        
        # Center at (0.5, 0.5, 0.5)
        dx = X - 0.5
        dy = Y - 0.5
        dz = Z - 0.5
        
        # Convert to Cartesian
        a, b, c = lattice.vectors
        
        v = np.zeros_like(X)
        for i in range(n1):
            for j in range(n2):
                for k in range(n3):
                    r = dx[i,j,k]*a + dy[i,j,k]*b + dz[i,j,k]*c
                    v[i,j,k] = 0.5 * omega**2 * np.sum(r**2)
        
        return v
    
    return v_harmonic


def sum_ionic_charges(atom_species):
    """Sum ionic charges of all atoms."""
    return sum(sp.zion for sp in atom_species)
