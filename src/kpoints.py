"""
K-Point Sampling for Brillouin Zone Integration.

Implements Monkhorst-Pack mesh generation and related utilities
for sampling the Brillouin zone in periodic DFT calculations.

Reference:
    Monkhorst, H. J. & Pack, J. D. "Special points for Brillouin-zone 
    integrations" Phys. Rev. B 13, 5188 (1976)

CROSS-REFERENCE: 
See `pydft-book`, Chapter "DFT in Plane Wave", Section "Brillouin zone sampling" 
(`ksampling.tex`) for theoretical foundations and mesh illustrations.
"""

import numpy as np
from .constants import TWOPI


def _monkhorst_pack_fractional(mesh, shift=(0.0, 0.0, 0.0)):
    """Generate a Gamma-centered mesh, optionally shifted by half a grid step."""
    mesh = np.asarray(mesh, dtype=int)
    shift = np.asarray(shift, dtype=float)
    axes = [np.arange(n, dtype=float) for n in mesh]
    indices = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, 3)
    k_frac = (indices + 0.5 * shift) / mesh
    weights = np.full(len(k_frac), 1.0 / len(k_frac))
    return k_frac, weights


class KPoints:
    """
    K-point mesh for Brillouin zone sampling.
    
    Generates a Monkhorst-Pack mesh of k-points in the first Brillouin zone.
    
    Attributes:
        k_frac: K-points in fractional (crystal) coordinates
        k_cart: K-points in Cartesian coordinates
        weights: Integration weights (sum to 1)
        nk: Number of k-points
    """
    
    def __init__(self, lattice, mesh, shift=None):
        """
        Generate Monkhorst-Pack k-point mesh.
        
        Args:
            lattice: Lattice object
            mesh: Tuple (n1, n2, n3) specifying mesh density
            shift: Optional tuple (s1, s2, s3) of shift flags
                   (0 = Gamma-centered, 1 = half-grid shift). This matches
                   generate_monkhorst_pack in kpoint_scf.py.
        """
        self.lattice = lattice
        self.mesh = mesh
        self.shift = shift if shift is not None else (0, 0, 0)
        
        # Generate k-points
        self.k_frac, self.weights = self._generate_mp_mesh()
        self.nk = len(self.k_frac)
        
        # Convert to Cartesian
        self.k_cart = self._frac_to_cart_all()
    
    def _generate_mp_mesh(self):
        """
        Generate Monkhorst-Pack mesh.

        k_i = (i + 0.5*s_i) / N, for i = 0, ..., N-1.  The shift flag
        s_i = 0 gives a Gamma-centered mesh; s_i = 1 shifts it by half a step.
        """
        return _monkhorst_pack_fractional(self.mesh, self.shift)
    
    def _frac_to_cart_all(self):
        """Convert all k-points from fractional to Cartesian."""
        b = self.lattice.reciprocal_vectors
        
        k_cart = np.zeros_like(self.k_frac)
        for i, k in enumerate(self.k_frac):
            k_cart[i] = k[0] * b[0] + k[1] * b[1] + k[2] * b[2]
        
        return k_cart
    
    def get_k_frac(self, index):
        """Get k-point in fractional coordinates."""
        return self.k_frac[index].copy()
    
    def get_k_cart(self, index):
        """Get k-point in Cartesian coordinates."""
        return self.k_cart[index].copy()
    
    def get_weight(self, index):
        """Get integration weight for k-point."""
        return self.weights[index]
    
    def get_nk(self):
        """Return number of k-points."""
        return self.nk
    
    def display(self):
        """Print k-point information."""
        print(f"\nK-point mesh: {self.mesh[0]} x {self.mesh[1]} x {self.mesh[2]}")
        print(f"Shift: {self.shift}")
        print(f"Total k-points: {self.nk}")
        print()
        print(f"{'Index':>6} {'k1':>10} {'k2':>10} {'k3':>10} {'Weight':>10}")
        print("-" * 50)
        
        for i in range(min(self.nk, 20)):
            k = self.k_frac[i]
            w = self.weights[i]
            print(f"{i+1:6d} {k[0]:10.6f} {k[1]:10.6f} {k[2]:10.6f} {w:10.6f}")
        
        if self.nk > 20:
            print(f"... ({self.nk - 20} more k-points)")


class GammaPoint:
    """
    Single Gamma point (k = 0) for simple calculations.
    
    This is a simplified version for educational purposes where
    we only consider the Gamma point.
    """
    
    def __init__(self, lattice):
        """Initialize with just the Gamma point."""
        self.lattice = lattice
        self.k_frac = np.array([[0.0, 0.0, 0.0]])
        self.k_cart = np.array([[0.0, 0.0, 0.0]])
        self.weights = np.array([1.0])
        self.nk = 1
        self.mesh = (1, 1, 1)
    
    def get_k_frac(self, index=0):
        return np.array([0.0, 0.0, 0.0])
    
    def get_k_cart(self, index=0):
        return np.array([0.0, 0.0, 0.0])
    
    def get_weight(self, index=0):
        return 1.0
    
    def get_nk(self):
        return 1
    
    def display(self):
        print("\nK-point sampling: Gamma point only")
        print("k = (0, 0, 0), weight = 1.0")


def gamma_only(lattice):
    """Create Gamma-point-only k-point sampling."""
    return GammaPoint(lattice)


def monkhorst_pack(lattice, n1, n2, n3, shift=False):
    """
    Create Monkhorst-Pack k-point mesh.
    
    Args:
        lattice: Lattice object
        n1, n2, n3: Mesh density in each direction
        shift: If True, apply a half-grid shift (flags (1, 1, 1))

    Returns:
        KPoints object
    """
    s = (1, 1, 1) if shift else (0, 0, 0)
    return KPoints(lattice, (n1, n2, n3), s)


def automatic_kpoints(lattice, kspacing=0.1):
    """
    Generate k-point mesh based on target k-spacing.
    
    Args:
        lattice: Lattice object
        kspacing: Maximum distance between k-points (Bohr^-1)
    
    Returns:
        KPoints object
    """
    if not np.isfinite(kspacing) or kspacing <= 0:
        raise ValueError("kspacing must be finite and positive.")

    b = lattice.reciprocal_vectors

    # Start from the floor, then compare the represented spacing directly.
    # The loop handles ratios that lie just above an integer after rounding.
    mesh = []
    for reciprocal_vector in b:
        length = np.linalg.norm(reciprocal_vector)
        count = max(1, int(length / kspacing))
        while length / count > kspacing:
            count += 1
        mesh.append(count)
    
    return KPoints(lattice, tuple(mesh))


def high_symmetry_path(lattice, path_string, npoints_per_segment=20):
    """
    Generate k-points along high-symmetry path for band structure.
    
    This is a simplified version supporting common paths.
    
    Args:
        lattice: Lattice object
        path_string: String like "GXMGR" for high-symmetry points
        npoints_per_segment: Number of points between each pair
    
    Returns:
        k_frac: K-points in fractional coordinates
        k_dist: Cumulative distance along path (for plotting)
        labels: Labels for high-symmetry points
    
    Note: Only supports cubic lattices in this simplified version.
    """
    # High-symmetry points for cubic lattices
    special_points = {
        'G': np.array([0.0, 0.0, 0.0]),      # Gamma
        'X': np.array([0.5, 0.0, 0.0]),
        'M': np.array([0.5, 0.5, 0.0]),
        'R': np.array([0.5, 0.5, 0.5]),
        'L': np.array([0.5, 0.5, 0.5]),      # Same as R for simple cubic
    }
    
    # Parse path
    points = [special_points[c] for c in path_string if c in special_points]
    
    if len(points) < 2:
        raise ValueError(f"Path must have at least 2 points, got: {path_string}")
    
    # Generate k-points along path
    k_frac = []
    k_dist = []
    labels = []
    label_positions = []
    
    cumulative_dist = 0.0
    b = lattice.reciprocal_vectors
    
    for i in range(len(points) - 1):
        k1, k2 = points[i], points[i+1]
        
        # Record label position
        if i == 0:
            labels.append(path_string[0])
            label_positions.append(0.0)
        
        # Linear interpolation
        for j in range(npoints_per_segment):
            t = j / npoints_per_segment
            k = (1 - t) * k1 + t * k2
            k_frac.append(k)
            
            # Distance in Cartesian reciprocal space
            if len(k_frac) > 1:
                dk = k_frac[-1] - k_frac[-2]
                dk_cart = dk[0] * b[0] + dk[1] * b[1] + dk[2] * b[2]
                cumulative_dist += np.linalg.norm(dk_cart)
            
            k_dist.append(cumulative_dist)
        
        # Add label for end of segment
        labels.append(path_string[i+1])
        label_positions.append(cumulative_dist + np.linalg.norm(
            (k2 - k1)[0] * b[0] + (k2 - k1)[1] * b[1] + (k2 - k1)[2] * b[2]
        ) / npoints_per_segment * (npoints_per_segment - 1))
    
    # Add final point
    k_frac.append(points[-1])
    dk = k_frac[-1] - k_frac[-2]
    dk_cart = dk[0] * b[0] + dk[1] * b[1] + dk[2] * b[2]
    cumulative_dist += np.linalg.norm(dk_cart)
    k_dist.append(cumulative_dist)
    label_positions[-1] = cumulative_dist
    
    return np.array(k_frac), np.array(k_dist), labels, label_positions


def get_kpg_norms(k_cart, gvector):
    """
    Compute |k + G|^2 for all G-vectors at a given k-point.
    
    Args:
        k_cart: K-point in Cartesian coordinates (1D array of length 3)
        gvector: GVector object
    
    Returns:
        kpg_squared: Array of |k + G|^2 values, shape (npw,)
    """
    k_cart = np.asarray(k_cart, dtype=float)
    return np.sum((gvector.cart + k_cart)**2, axis=1)
