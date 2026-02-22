"""
Exchange-correlation functionals.

Implements LDA (Local Density Approximation) with:
- Slater exchange
- Perdew-Zunger correlation

=============================================================================
PEDAGOGICAL NOTES
=============================================================================

THE HEART OF DFT: Exchange-Correlation

The exact ground state energy is a functional of the density:
    E[ρ] = T[ρ] + E_ext[ρ] + E_H[ρ] + E_xc[ρ]

where:
    T[ρ]     = kinetic energy of interacting electrons (UNKNOWN!)
    E_ext[ρ] = interaction with external potential
    E_H[ρ]   = classical Coulomb (Hartree) energy
    E_xc[ρ]  = exchange-correlation energy (ALL the quantum many-body physics!)

THE KOHN-SHAM TRICK
-------------------
Replace the interacting electrons with non-interacting electrons that have
the SAME density ρ(r). The kinetic energy of non-interacting electrons is:
    T_s[ρ] = Σ_i f_i ⟨ψ_i|-½∇²|ψ_i⟩  (easy to compute!)

The difference is absorbed into E_xc:
    E_xc = (T - T_s) + (E_ee - E_H)

So E_xc contains:
    1. Kinetic correlation (T - T_s)
    2. Exchange energy (Pauli exclusion)
    3. Correlation energy (electron-electron correlation beyond exchange)

THE CHALLENGE: E_xc[ρ] IS UNKNOWN!
----------------------------------
We must approximate it. The simplest approximation is the Local Density
Approximation (LDA):

    E_xc[ρ] ≈ ∫ ρ(r) ε_xc(ρ(r)) dr

where ε_xc(ρ) is the XC energy per electron of a UNIFORM electron gas
at density ρ. This is known exactly from Quantum Monte Carlo!

The XC potential (for the Kohn-Sham equations) is:
    V_xc(r) = δE_xc/δρ(r) = d(ρε_xc)/dρ

LDA is surprisingly accurate for solids, but fails for molecules with
rapidly varying density. GGA (e.g., PBE) improves on this.

References:
    Kohn & Sham, Phys. Rev. 140, A1133 (1965) - The original KS paper
    Perdew & Zunger, Phys. Rev. B 23, 5048 (1981) - LDA parametrization used here
=============================================================================
"""

import numpy as np
from .constants import PI


def lda_xc(rho):
    """
    Compute LDA exchange-correlation potential and energy density.
    
    Uses Slater exchange and Perdew-Zunger (1981) correlation.
    
    Args:
        rho: Electron density array (real, positive)
    
    Returns:
        vxc: Exchange-correlation potential (same shape as rho)
        exc: Exchange-correlation energy density (same shape as rho)
    
    Reference:
        Perdew & Zunger, Phys. Rev. B 23, 5048 (1981)
    """
    rho = np.asarray(rho, dtype=float)
    
    # Avoid division by zero for very small densities
    rho_safe = np.maximum(np.abs(rho), 1e-30)
    
    # Exchange
    vx, ex = slater_exchange(rho_safe)
    
    # Correlation
    vc, ec = pz_correlation(rho_safe)
    
    return vx + vc, ex + ec


def slater_exchange(rho):
    """
    Slater (Dirac) exchange functional.

    epsilon_x = -3/4 * (3*rho/pi)^(1/3)
    V_x = -4/3 * epsilon_x / rho = -(3*rho/pi)^(1/3)

    Args:
        rho: Electron density (positive)

    Returns:
        vx: Exchange potential
        ex: Exchange energy density

    PEDAGOGICAL NOTE: What is Exchange?
    ------------------------------------
    Exchange arises from the Pauli exclusion principle: two electrons
    with the same spin cannot be at the same position. This creates an
    "exchange hole" around each electron.

    For a uniform electron gas, Dirac (1930) derived the exact exchange:
        ε_x = -3/4 × (3/π)^{1/3} × ρ^{1/3}

    Note:
    - ε_x < 0 (exchange always lowers the energy)
    - ε_x ~ ρ^{1/3} (depends only weakly on density)

    The exchange potential is V_x = d(ρε_x)/dρ:
        V_x = ε_x + ρ × dε_x/dρ = ε_x × (1 + 1/3) = (4/3)ε_x

    So V_x = -(3ρ/π)^{1/3}  (Slater's Xα potential with α=1)
    """
    # Exchange coefficient: c_x = -(3/π)^{1/3}
    cx = -(3.0 / PI) ** (1.0 / 3.0)

    # Exchange potential: V_x = c_x × ρ^{1/3}
    rho_third = rho ** (1.0 / 3.0)
    vx = cx * rho_third

    # Exchange energy density: ε_x = (3/4) × V_x
    # (because V_x = (4/3)ε_x, so ε_x = (3/4)V_x)
    ex = 0.75 * vx

    return vx, ex


def pz_correlation(rho):
    """
    Perdew-Zunger parameterization of the correlation energy.

    Different formulas for high density (rs < 1) and low density (rs >= 1),
    where rs = (3/(4*pi*rho))^(1/3) is the Wigner-Seitz radius.

    Args:
        rho: Electron density (positive)

    Returns:
        vc: Correlation potential
        ec: Correlation energy density

    PEDAGOGICAL NOTE: What is Correlation?
    ---------------------------------------
    Correlation is the part of electron-electron interaction NOT captured
    by the Hartree (classical Coulomb) and exchange energies.

    Physical origin: electrons avoid each other beyond what Pauli exclusion
    requires. They are "correlated" - knowing one electron's position tells
    us something about where others are likely to be.

    THE WIGNER-SEITZ RADIUS
    The Wigner-Seitz radius r_s is defined by:
        (4π/3)r_s³ = 1/ρ   (volume per electron)
    So:
        r_s = (3/(4πρ))^{1/3}

    Physical meaning:
    - r_s ~ 2-6 Bohr for metals (Na: ~4, Al: ~2)
    - r_s < 1: high density (electrons close together)
    - r_s > 1: low density (electrons far apart)

    WHY TWO REGIMES?
    - High density (r_s < 1): perturbation theory works → logarithmic form
    - Low density (r_s ≥ 1): strong correlation → fit to QMC data

    The PZ parameterization interpolates Ceperley-Alder QMC data.
    Reference: Perdew & Zunger, Phys. Rev. B 23, 5048 (1981)
    """
    # Wigner-Seitz radius: r_s = (3/(4πρ))^{1/3}
    rs = (3.0 / (4.0 * PI * rho)) ** (1.0 / 3.0)

    # Initialize outputs
    vc = np.zeros_like(rho)
    ec = np.zeros_like(rho)

    # Low density regime: r_s >= 1 (strong correlation, fit to QMC)
    mask_low = rs >= 1.0
    if np.any(mask_low):
        vc[mask_low], ec[mask_low] = _pz_low_density(rs[mask_low])

    # High density regime: r_s < 1 (weak correlation, perturbative form)
    mask_high = ~mask_low
    if np.any(mask_high):
        vc[mask_high], ec[mask_high] = _pz_high_density(rs[mask_high])

    return vc, ec


def _pz_low_density(rs):
    """
    PZ correlation for rs >= 1 (low density).
    
    ec = gamma / (1 + beta1*sqrt(rs) + beta2*rs)
    """
    # Parameters for unpolarized electron gas
    gamma = -0.1423
    beta1 = 1.0529
    beta2 = 0.3334
    
    rs_sqrt = np.sqrt(rs)
    denom = 1.0 + beta1 * rs_sqrt + beta2 * rs
    
    # Correlation energy density
    ec = gamma / denom
    
    # Correlation potential: vc = d(rho*ec)/d(rho)
    # Using chain rule with rs
    numer = 1.0 + (7.0 / 6.0) * beta1 * rs_sqrt + (4.0 / 3.0) * beta2 * rs
    vc = ec * numer / denom
    
    return vc, ec


def _pz_high_density(rs):
    """
    PZ correlation for rs < 1 (high density).
    
    ec = A*ln(rs) + B + C*rs*ln(rs) + D*rs
    """
    # Parameters
    A = 0.0311
    B = -0.048
    C = 0.0020
    D = -0.0116
    
    rs_ln = np.log(rs)
    
    # Correlation energy density
    ec = A * rs_ln + B + C * rs * rs_ln + D * rs
    
    # Correlation potential
    vc = (A * rs_ln + (B - A / 3.0) + 
          (2.0 / 3.0) * C * rs * rs_ln + 
          (1.0 / 3.0) * (2.0 * D - C) * rs)
    
    return vc, ec


def compute_xc_energy(rho, exc, volume, n_grid):
    """
    Compute total XC energy from energy density.
    
    E_xc = integral(rho * epsilon_xc) dr
         = (volume / n_grid) * sum(rho * epsilon_xc)
    
    Args:
        rho: Electron density in real space
        exc: XC energy density
        volume: Cell volume
        n_grid: Total number of grid points
    
    Returns:
        Total XC energy
    """
    return (volume / n_grid) * np.sum(np.real(rho * exc))


def compute_xc_potential_energy(rho, vxc, volume, n_grid):
    """
    Compute XC potential energy (for double-counting correction).
    
    E_vxc = integral(rho * V_xc) dr
    
    Args:
        rho: Electron density in real space
        vxc: XC potential
        volume: Cell volume
        n_grid: Total number of grid points
    
    Returns:
        XC potential energy
    """
    return (volume / n_grid) * np.sum(np.real(rho * vxc))
