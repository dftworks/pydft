"""
Electronic Smearing Functions for Occupation Numbers.

Implements various smearing schemes for computing fractional occupation
numbers in metallic systems, which is essential for smooth convergence
of the SCF loop.

Reference:
    - Fermi-Dirac: Standard statistical mechanics
    - Gaussian: Fu, C.-L. & Ho, K.-M. Phys. Rev. B 28, 5480 (1983)
    - Methfessel-Paxton: Methfessel, M. & Paxton, A. T. Phys. Rev. B 40, 3616 (1989)
"""

import numpy as np
from scipy.special import erfc, xlogy
from .constants import BOLTZMANN_CONSTANT, HA_TO_EV


class FermiDirac:
    """
    Fermi-Dirac smearing.
    
    f(E) = 1 / (exp((E - mu) / kT) + 1)
    
    This is the physically correct distribution for electrons at
    finite temperature, but has slow convergence with respect to
    the number of k-points.
    """
    
    def __init__(self, temperature=300.0):
        """
        Initialize Fermi-Dirac smearing.
        
        Args:
            temperature: Electronic temperature in Kelvin
        """
        if not np.isfinite(temperature) or temperature <= 0:
            raise ValueError("Temperature must be finite and positive; use fixed for T=0.")
        self.temperature = temperature
        self.kbt = max(BOLTZMANN_CONSTANT * temperature, 1e-10)
    
    def occupation(self, energy, fermi_level):
        """
        Compute occupation number.
        
        Args:
            energy: Eigenvalue (Hartree)
            fermi_level: Fermi level (Hartree)
        
        Returns:
            Occupation number between 0 and 1
        """
        x = (energy - fermi_level) / self.kbt
        
        # Avoid overflow
        if x > 100:
            return 0.0
        elif x < -100:
            return 1.0
        else:
            return 1.0 / (np.exp(x) + 1.0)
    
    def occupation_array(self, energies, fermi_level):
        """Compute occupations for array of energies."""
        x = (energies - fermi_level) / self.kbt
        x = np.clip(x, -100, 100)
        return 1.0 / (np.exp(x) + 1.0)
    
    def entropy(self, occupations):
        """
        Compute electronic entropy.
        
        S = -kT * sum_i [f*ln(f) + (1-f)*ln(1-f)]
        
        Returns:
            Positive T*S in Hartree; subtract this from internal energy
            to obtain F=E-T*S. Input occupations are per spin in [0, 1].
        """
        f = np.asarray(occupations, dtype=float)
        if not np.all(np.isfinite(f)) or np.any((f < 0) | (f > 1)):
            raise ValueError("Entropy expects finite per-spin occupations in [0, 1].")
        return -self.kbt * np.sum(xlogy(f, f) + xlogy(1-f, 1-f))

    @property
    def name(self):
        return "Fermi-Dirac"


class Gaussian:
    """
    Gaussian smearing (distinct from Marzari-Vanderbilt cold smearing).
    
    f(E) = 0.5 * erfc((E - mu) / sigma)
    
    This smearing has better convergence properties than Fermi-Dirac
    but the entropy term is not physical.
    """
    
    def __init__(self, sigma=0.01):
        """
        Initialize Gaussian smearing.
        
        Args:
            sigma: Smearing width in Hartree (typical: 0.01-0.1 Ha)
        """
        if not np.isfinite(sigma) or sigma <= 0:
            raise ValueError("Smearing width must be finite and positive.")
        self.sigma = sigma
    
    def occupation(self, energy, fermi_level):
        """Compute occupation number."""
        x = (energy - fermi_level) / self.sigma
        return 0.5 * erfc(x)
    
    def occupation_array(self, energies, fermi_level):
        """Compute occupations for array of energies."""
        x = (energies - fermi_level) / self.sigma
        return 0.5 * erfc(x)
    
    def entropy(self, occupations):
        """
        Compute entropy correction for Gaussian smearing.
        
        Note: This is not a physical entropy, but a correction term.
        """
        # Simplified approximation
        return 0.0
    
    @property
    def name(self):
        return "Gaussian"


class MethfesselPaxton:
    """
    Methfessel-Paxton smearing (orders 1 and 2).
    
    Provides better integration accuracy than Gaussian smearing
    while still having fast k-point convergence.
    
    The first-order MP smearing function is:
    f(x) = 0.5*erfc(x) - exp(-x^2) * x / (2*sqrt(pi))
    """
    
    def __init__(self, sigma=0.01, order=1):
        """
        Initialize Methfessel-Paxton smearing.
        
        Args:
            sigma: Smearing width in Hartree
            order: Order of the method (1 or 2)
        """
        if not np.isfinite(sigma) or sigma <= 0:
            raise ValueError("Smearing width must be finite and positive.")
        self.sigma = sigma
        if order not in (1, 2):
            raise ValueError("Methfessel-Paxton order must be 1 or 2.")
        self.order = order
    
    def occupation(self, energy, fermi_level):
        """Compute occupation number."""
        x = (energy - fermi_level) / self.sigma
        return self._mp_function(x)
    
    def occupation_array(self, energies, fermi_level):
        """Compute occupations for array of energies."""
        x = (energies - fermi_level) / self.sigma
        return self._mp_function(x)
    
    def _mp_function(self, x):
        """Methfessel-Paxton occupation function."""
        # 0th order = Gaussian
        f = 0.5 * erfc(x)
        
        if self.order >= 1:
            # 1st order correction
            A1 = -0.5 / np.sqrt(np.pi)
            f += A1 * x * np.exp(-x**2)
        
        if self.order >= 2:
            # 2nd order correction
            A2 = 1.0 / (32 * np.sqrt(np.pi))
            H3 = 8.0 * x**3 - 12.0 * x  # Odd Hermite polynomial
            f += A2 * H3 * np.exp(-x**2)
        
        # Signed overshoot cancels integration-error moments; do not clip.
        return f
    
    def entropy(self, occupations):
        """Entropy correction term."""
        return 0.0
    
    @property
    def name(self):
        return f"Methfessel-Paxton (order {self.order})"


class FixedOccupation:
    """
    Fixed occupation numbers (no smearing).
    
    For insulators and semiconductors with a gap, fixed occupation
    is often sufficient and provides exact results.
    """
    
    def __init__(self):
        pass
    
    def occupation(self, energy, fermi_level):
        """Occupation is 1 below Fermi level, 0 above."""
        return 1.0 if energy < fermi_level else 0.0
    
    def occupation_array(self, energies, fermi_level):
        """Compute occupations for array of energies."""
        return np.where(energies < fermi_level, 1.0, 0.0)
    
    def entropy(self, occupations):
        """No entropy for fixed occupations."""
        return 0.0
    
    @property
    def name(self):
        return "Fixed"


def create_smearing(scheme, **kwargs):
    """
    Factory function to create smearing object.
    
    Args:
        scheme: 'fd', 'gaussian', 'mp', 'mp1', 'mp2', or 'fixed'
        **kwargs: Additional parameters for smearing (temperature, sigma, etc.)
    
    Returns:
        Smearing object
    """
    scheme = scheme.lower()
    
    if scheme == 'fd' or scheme == 'fermi-dirac':
        temperature = kwargs.get('temperature', 300.0)
        return FermiDirac(temperature=temperature)
    
    elif scheme in ('gs', 'gaussian'):
        sigma = kwargs.get('sigma', 0.01)
        return Gaussian(sigma=sigma)
    
    elif scheme in ('mp', 'mp1'):
        sigma = kwargs.get('sigma', 0.01)
        return MethfesselPaxton(sigma=sigma, order=1)
    
    elif scheme == 'mp2':
        sigma = kwargs.get('sigma', 0.01)
        return MethfesselPaxton(sigma=sigma, order=2)
    
    elif scheme == 'fixed':
        return FixedOccupation()
    
    else:
        raise ValueError(f"Unknown smearing scheme: {scheme}")


def find_fermi_level(eigenvalues, weights, n_electrons, smearing, 
                     spin_factor=2.0, tol=1e-10, max_iter=100):
    """
    Find Fermi level for given eigenvalues and number of electrons.
    
    Uses bisection method to find the Fermi level that gives the
    correct number of electrons.
    
    Args:
        eigenvalues: Array of shape (nk, n_bands) or (n_bands,)
        weights: K-point weights, shape (nk,) or scalar 1.0
        n_electrons: Target number of electrons
        smearing: Smearing object
        spin_factor: 2.0 for spin-paired (default), 1.0 for spin-polarized
        tol: Tolerance for electron count
        max_iter: Maximum iterations
    
    Returns:
        fermi_level: Fermi level in Hartree
        occupations: Occupation numbers for each eigenvalue
    """
    eigenvalues = np.asarray(eigenvalues, dtype=float)
    if (eigenvalues.ndim not in (1, 2) or eigenvalues.size == 0
            or not np.all(np.isfinite(eigenvalues))):
        raise ValueError("Eigenvalues must be a finite nonempty 1D or 2D array.")
    if (not np.isfinite(spin_factor) or spin_factor <= 0
            or not np.isfinite(tol) or tol <= 0
            or not isinstance(max_iter, (int, np.integer)) or max_iter < 1):
        raise ValueError("Positive spin_factor, tol and integer max_iter are required.")
    nk = 1 if eigenvalues.ndim == 1 else eigenvalues.shape[0]
    if np.isscalar(weights):
        weights = np.full(nk, float(weights) / nk)
    weights = np.asarray(weights, dtype=float)
    if (weights.shape != (nk,) or not np.all(np.isfinite(weights))
            or np.any(weights < 0) or weights.sum() <= 0):
        raise ValueError("K-point weights must be finite, nonnegative and shaped (nk,).")
    eigs = eigenvalues.ravel()
    wts = np.repeat(weights, eigenvalues.shape[-1])
    capacity = spin_factor * wts.sum()
    if not np.isfinite(n_electrons) or not 0 <= n_electrons <= capacity:
        raise ValueError(f"Electron count must lie in [0, {capacity:g}] for these bands.")

    width = max(1.0, 10 * getattr(smearing, 'sigma', 0.0),
                10 * getattr(smearing, 'kbt', 0.0))
    lo, hi = eigs.min() - width, eigs.max() + width
    if n_electrons == 0 or n_electrons == capacity:
        filled = n_electrons == capacity
        # Return occupations consistent with the reported finite mu. At
        # finite T, exact empty/full filling is an asymptote; use charge tol.
        for _ in range(max_iter):
            mu = eigs.max()+width if filled else eigs.min()-width
            occupations = spin_factor * smearing.occupation_array(eigs, mu)
            if abs(np.dot(wts, occupations)-n_electrons) < tol:
                return mu, occupations.reshape(eigenvalues.shape)
            width *= 2
        raise RuntimeError("Fermi-level search did not converge at band-capacity endpoint.")

    if isinstance(smearing, FixedOccupation):
        # Share partial filling across a degenerate manifold rather than
        # arbitrarily favoring a band/k-point. Weights determine its capacity.
        occupations = np.zeros_like(eigs)
        order = np.argsort(eigs)
        remaining = float(n_electrons)
        start = 0
        while start < len(order):
            stop = start + 1
            while stop < len(order) and abs(eigs[order[stop]]-eigs[order[start]]) <= 1e-12:
                stop += 1
            group = order[start:stop]
            group_capacity = spin_factor * wts[group].sum()
            if group_capacity > 0:
                fill = min(remaining, group_capacity)
                occupations[group] = spin_factor * fill / group_capacity
                remaining -= fill
                if remaining <= tol:
                    mu = eigs[order[start]]
                    return mu, occupations.reshape(eigenvalues.shape)
            start = stop
        raise RuntimeError("Fixed occupation filling failed to conserve charge.")

    def residual(mu):
        return spin_factor * np.dot(wts, smearing.occupation_array(eigs, mu)) - n_electrons

    # A verified sign bracket works even for nonmonotone MP occupations,
    # although MP may have more than one root. Do not assume a unique mu.
    for _ in range(100):
        if residual(lo) <= 0 <= residual(hi):
            break
        width *= 2
        lo, hi = eigs.min()-width, eigs.max()+width
    else:
        raise RuntimeError("Could not bracket the Fermi level.")
    for _ in range(max_iter):
        mu = (lo + hi) / 2
        error = residual(mu)
        if abs(error) < tol:
            occupations = spin_factor * smearing.occupation_array(eigs, mu)
            return mu, occupations.reshape(eigenvalues.shape)
        if error < 0:
            lo = mu
        else:
            hi = mu
    raise RuntimeError(f"Fermi-level search did not converge: charge error {error:.3g}.")


def compute_band_energy(eigenvalues, occupations, weights=1.0):
    """
    Compute band structure energy.
    
    E_band = sum_{n,k} w_k * f_{n,k} * eps_{n,k}
    
    Args:
        eigenvalues: Eigenvalues (nk, nbands) or (nbands,)
        occupations: Occupation numbers (same shape)
        weights: K-point weights
    
    Returns:
        Band energy in Hartree
    """
    if eigenvalues.ndim == 1:
        return np.sum(occupations * eigenvalues)
    else:
        if np.isscalar(weights):
            weights = np.ones(eigenvalues.shape[0]) * weights
        return np.sum(weights[:, np.newaxis] * occupations * eigenvalues)
