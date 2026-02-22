"""
Hartree potential computation.

The Hartree potential describes electron-electron Coulomb repulsion.
In reciprocal space: V_H(G) = 4*pi*rho(G) / |G|^2

=============================================================================
PEDAGOGICAL NOTES
=============================================================================

THE HARTREE POTENTIAL
---------------------
The Hartree (or "classical Coulomb") potential is the electrostatic
potential created by the electron charge distribution:

    V_H(r) = ∫ ρ(r') / |r - r'| dr'

This is CLASSICAL electrostatics - it ignores exchange and correlation.
(Those are handled separately in E_xc.)

SOLVING POISSON'S EQUATION
--------------------------
V_H satisfies Poisson's equation:
    ∇²V_H(r) = -4πρ(r)

In real space, this requires solving a PDE (expensive!).

THE G-SPACE TRICK
-----------------
In Fourier space, ∇² becomes -|G|²:
    -|G|² V_H(G) = -4π ρ(G)

So:
    V_H(G) = 4π ρ(G) / |G|²

This is just ALGEBRA, not a PDE! The Laplacian becomes trivial.

THE G=0 PROBLEM
---------------
V_H(G=0) = 4π ρ(G=0) / 0² = ∞  (!)

This divergence corresponds to the infinite self-energy of a uniform
charge distribution.

SOLUTION: Set V_H(G=0) = 0. This is equivalent to adding a uniform
neutralizing background charge (the "jellium" model). For neutral
systems (like atoms in molecules or crystals), this is physically correct:
the total charge integrates to zero.

THE HARTREE ENERGY
------------------
    E_H = (1/2) ∫ ρ(r) V_H(r) dr
        = (Ω/2) Σ_{G≠0} 4π|ρ(G)|²/|G|²

The factor of 1/2 avoids double-counting (each pair of electrons
contributes once, not twice).
=============================================================================
"""

import numpy as np
from .constants import FOURPI


def compute_hartree_potential(rhog, g_norms):
    """
    Compute Hartree potential in G-space.

    V_H(G) = 4*pi * rho(G) / |G|^2  for G != 0
    V_H(0) = 0  (charge neutrality / jellium background)

    Args:
        rhog: Density in G-space (complex array, G=0 should be first)
        g_norms: |G| for each G-vector (G=0 should be first)

    Returns:
        vhg: Hartree potential in G-space

    PEDAGOGICAL NOTE: Derivation
    -----------------------------
    Starting from Poisson's equation: ∇²V_H = -4πρ

    Take Fourier transform of both sides:
        F[∇²V_H] = -4π F[ρ]

    Using F[∇²f] = -|G|² F[f]:
        -|G|² V_H(G) = -4π ρ(G)

    Solve for V_H(G):
        V_H(G) = 4π ρ(G) / |G|²

    The G=0 term is set to zero because:
    1. 4π ρ(G=0) / 0² would be infinite
    2. Physically, this corresponds to a neutralizing background
    3. For neutral systems, the total energy is well-defined
    """
    vhg = np.zeros_like(rhog, dtype=complex)

    # G=0 term: set to zero (neutralizing background)
    # This avoids the 1/0² divergence and ensures a well-defined energy
    vhg[0] = 0.0

    # G ≠ 0 terms: apply the Poisson equation solution
    # V_H(G) = 4π ρ(G) / |G|²
    g2 = g_norms[1:]**2
    vhg[1:] = FOURPI * rhog[1:] / g2

    return vhg


def compute_hartree_energy(rhog, g_norms, volume):
    """
    Compute Hartree energy.

    E_H = (1/2) * integral(V_H * rho) dr
        = (Omega/2) * sum_{G!=0} 4*pi * |rho(G)|^2 / G^2

    Args:
        rhog: Density in G-space
        g_norms: |G| for each G-vector
        volume: Cell volume

    Returns:
        Hartree energy

    PEDAGOGICAL NOTE: Why the Factor of 1/2?
    -----------------------------------------
    The Hartree energy represents electron-electron Coulomb repulsion:
        E_H = (1/2) ∫∫ ρ(r)ρ(r')/|r-r'| dr dr'

    The factor of 1/2 prevents DOUBLE-COUNTING: without it, each pair
    of electrons would be counted twice (once for r, once for r').

    DERIVATION IN G-SPACE
    Using Parseval's theorem and V_H(G) = 4πρ(G)/|G|²:
        E_H = (1/2) ∫ ρ(r) V_H(r) dr
            = (1/2) Ω Σ_G ρ*(G) V_H(G)       [Parseval]
            = (1/2) Ω Σ_G ρ*(G) × 4πρ(G)/|G|²
            = (Ω/2) Σ_{G≠0} 4π|ρ(G)|²/|G|²

    The G=0 term is excluded (consistent with V_H(G=0) = 0).
    """
    # Skip G=0 (it's zero anyway due to V_H(0) = 0)
    g2 = g_norms[1:]**2
    rhog_no_g0 = rhog[1:]

    # E_H = (Ω/2) × Σ_{G≠0} 4π|ρ(G)|²/|G|²
    # The factor of 1/2 prevents double-counting of electron pairs
    e_hartree = 0.5 * volume * FOURPI * np.sum(np.abs(rhog_no_g0)**2 / g2)

    return e_hartree


def compute_hartree_stress(rhog, g_cart, g_norms, volume):
    """
    Compute Hartree contribution to stress tensor.
    
    sigma_ij = -(1/Omega) * sum_{G!=0} 4*pi * |rho(G)|^2 / G^4 * 
               (delta_ij * G^2 - 2 * G_i * G_j)
    
    Args:
        rhog: Density in G-space
        g_cart: Cartesian G-vectors
        g_norms: |G| for each G-vector
        volume: Cell volume
    
    Returns:
        3x3 stress tensor
    """
    stress = np.zeros((3, 3))
    
    # Skip G=0
    for ig in range(1, len(g_norms)):
        g = g_cart[ig]
        g2 = g_norms[ig]**2
        rho_g2 = np.abs(rhog[ig])**2
        
        prefactor = FOURPI * rho_g2 / (g2 * g2)
        
        for i in range(3):
            for j in range(3):
                delta_ij = 1.0 if i == j else 0.0
                stress[i, j] += prefactor * (delta_ij * g2 - 2.0 * g[i] * g[j])
    
    stress *= -1.0 / volume
    
    return stress
