"""
Ewald Summation for Ion-Ion Coulomb Energy.

Computes the electrostatic energy of point charges in a periodic system
by splitting the slowly converging 1/r sum into real-space and 
reciprocal-space parts.

Reference:
    Ewald, P. P. "Die Berechnung optischer und elektrostatischer 
    Gitterpotentiale" Ann. Phys. 369, 253 (1921)
"""

import numpy as np
from scipy.special import erfc
from .constants import PI, TWOPI, FOURPI


class Ewald:
    """
    Ewald summation for periodic systems.
    
    Computes:
    - Ion-ion Coulomb energy
    - Forces on ions
    - Stress tensor contribution
    
    The Ewald method splits the Coulomb sum into three parts:
    
    E_ewald = E_real + E_recip + E_self + E_G0
    
    Attributes:
        energy: Total Ewald energy (Hartree)
        forces: Forces on each ion (Hartree/Bohr)
        stress: Stress tensor (3x3, Hartree/Bohr^3)
    """
    
    def __init__(self, lattice, atom_positions, charges, gvector=None, eta=None):
        """
        Compute Ewald summation.
        
        Args:
            lattice: Lattice object with vectors and volume
            atom_positions: Fractional coordinates of atoms, shape (natoms, 3)
            charges: Ion charges (positive), shape (natoms,)
            gvector: Optional GVector object for reciprocal space part
            eta: Ewald parameter (computed automatically if None)
        """
        self.lattice = lattice
        self.positions = np.asarray(atom_positions, dtype=float) % 1.0
        self.charges = np.asarray(charges, dtype=float)
        self.natoms = len(charges)
        self.volume = lattice.volume
        if (self.positions.shape != (self.natoms, 3) or self.natoms == 0
                or not np.all(np.isfinite(self.positions))
                or self.charges.shape != (self.natoms,)
                or not np.all(np.isfinite(self.charges))):
            raise ValueError("Supply finite positions (natoms, 3) and charges (natoms,).")
        
        # Lattice vectors
        self.a = lattice.vectors[0]
        self.b = lattice.vectors[1]
        self.c = lattice.vectors[2]

        for i in range(self.natoms):
            for j in range(i):
                separation = self.positions[i] - self.positions[j]
                separation -= np.round(separation)
                if np.linalg.norm(separation @ lattice.vectors) < 1e-10:
                    raise ValueError("Distinct ions cannot occupy the same periodic position.")
        
        # Compute optimal eta if not provided
        if eta is None:
            self.eta = self._compute_optimal_eta()
        else:
            self.eta = eta
        
        if not np.isfinite(self.eta) or self.eta <= 0:
            raise ValueError("Ewald eta must be finite and positive.")

        # Get real-space cutoff
        self.rmax = self._compute_rmax()
        
        # Generate neighbor cells for real-space sum
        self.nn_cells = self._make_near_cells()
        
        # Compute energy
        e_real = self._compute_energy_real_space()
        e_recip = self._compute_energy_reciprocal_space(gvector)
        e_self, e_g0 = self._compute_energy_self_and_g0()
        
        self.energy = e_real + e_recip + e_self + e_g0
        
        # Compute forces
        f_real = self._compute_force_real_space()
        f_recip = self._compute_force_reciprocal_space(gvector)
        
        self.forces = f_real + f_recip
        
        # Compute stress
        self.stress = self._compute_stress(gvector)
    
    def _compute_optimal_eta(self):
        """
        Compute optimal Ewald parameter.
        
        Balances real and reciprocal space convergence.
        eta ~ (natoms / volume)^(2/3)
        """
        rho = self.natoms / self.volume
        eta = (rho ** (2.0/3.0)) * PI
        # Ensure reasonable value
        return max(eta, 0.1)
    
    def _compute_rmax(self, eps=1e-12):
        """Compute real-space cutoff where erfc(sqrt(eta)*r) < eps."""
        rmax = 1.0
        sqrt_eta = np.sqrt(self.eta)
        while erfc(sqrt_eta * rmax) > eps * rmax:
            rmax += 0.5
            if rmax > 100.0:  # Safety limit
                raise ValueError("Ewald real-space cutoff exceeds 100 Bohr; increase eta.")
        return rmax
    
    def _make_near_cells(self):
        """Generate list of nearby cells for real-space sum."""
        # A pair can be short even when its lattice translation is long.
        # Fractional pair offsets lie in (-1,1); dual-basis bounds are safe
        # for skew cells as well as orthogonal cells.
        bounds = np.ceil(self.rmax * np.linalg.norm(
            np.linalg.inv(self.lattice.vectors), axis=0) + 1).astype(int)
        from itertools import product
        return [np.array(c) for c in product(
            *(range(-n, n + 1) for n in bounds))]

    def _frac_to_cart(self, frac):
        """Convert fractional to Cartesian coordinates."""
        return frac[0] * self.a + frac[1] * self.b + frac[2] * self.c
    
    def _compute_energy_real_space(self):
        """
        Compute real-space part of Ewald energy.
        
        E_real = (1/2) * sum_{R} sum_{i!=j} Z_i * Z_j * erfc(sqrt(eta)*r) / r
        """
        sqrt_eta = np.sqrt(self.eta)
        energy = 0.0
        
        for cell in self.nn_cells:
            for i in range(self.natoms):
                for j in range(self.natoms):
                    if i == j and np.all(cell == 0):
                        continue
                    
                    # r_i - r_j - R in fractional
                    dfrac = self.positions[i] - self.positions[j] - cell
                    
                    # Convert to Cartesian
                    dr = self._frac_to_cart(dfrac)
                    r = np.linalg.norm(dr)
                    
                    if 1e-10 < r <= self.rmax:
                        energy += 0.5 * self.charges[i] * self.charges[j] * \
                                  erfc(sqrt_eta * r) / r
        
        return energy
    
    def _reciprocal_terms(self, gvector):
        """Return nonzero G, structure factors and screened Coulomb weights.

        A supplied GVector controls reciprocal truncation; its convergence
        must be checked separately. Otherwise use a 1e-14 Gaussian tail.
        """
        if gvector is None:
            from itertools import product
            gmax = np.sqrt(-4 * self.eta * np.log(1e-14))
            bounds = np.ceil(gmax * np.linalg.norm(self.lattice.vectors, axis=1)
                             / TWOPI).astype(int)
            miller = np.array(list(product(*(range(-n, n + 1) for n in bounds))))
            cart = miller @ self.lattice.reciprocal_vectors
            g2 = np.sum(cart**2, axis=1)
            keep = (g2 > 1e-20) & (g2 <= gmax**2)
            miller, cart, g2 = miller[keep], cart[keep], g2[keep]
        else:
            keep = gvector.norms > 1e-10
            miller = gvector.miller[keep]
            cart = gvector.cart[keep]
            g2 = gvector.norms[keep]**2
        phases = np.exp(1j * TWOPI * (miller @ self.positions.T))
        structure = phases @ self.charges
        weight = np.exp(-g2 / (4 * self.eta)) / g2
        return cart, g2, phases, structure, weight

    def _compute_energy_reciprocal_space(self, gvector):
        """E_G = (2*pi/V) sum_G exp(-G²/(4*eta)) |S(G)|²/G²."""
        _, _, _, structure, weight = self._reciprocal_terms(gvector)
        return 2 * PI / self.volume * np.sum(weight * np.abs(structure)**2)

    def _compute_energy_recip_simple(self):
        return self._compute_energy_reciprocal_space(None)

    def _compute_energy_self_and_g0(self):
        """
        Compute self-energy and G=0 corrections.
        
        E_self = -sqrt(eta/pi) * sum_i Z_i^2
        E_G0 = -pi / (2*V*eta) * (sum_i Z_i)^2
        """
        z_sum = np.sum(self.charges)
        z2_sum = np.sum(self.charges**2)
        
        e_self = -np.sqrt(self.eta / PI) * z2_sum
        e_g0 = -PI / (2.0 * self.volume * self.eta) * z_sum**2
        
        return e_self, e_g0
    
    def _compute_force_real_space(self):
        """
        Compute real-space force contribution.
        
        F_i = -dE/dr_i
        """
        sqrt_eta = np.sqrt(self.eta)
        forces = np.zeros((self.natoms, 3))
        
        for i in range(self.natoms):
            f = np.zeros(3)
            
            for cell in self.nn_cells:
                for j in range(self.natoms):
                    if i == j and np.all(cell == 0):
                        continue
                    
                    dfrac = self.positions[i] - self.positions[j] - cell
                    dr = self._frac_to_cart(dfrac)
                    r = np.linalg.norm(dr)
                    
                    if 1e-10 < r <= self.rmax:
                        # Derivative of erfc(sqrt(eta)*r)/r
                        coeff = self.charges[i] * self.charges[j] / r**2
                        coeff *= (2.0 * sqrt_eta / np.sqrt(PI) * 
                                 np.exp(-self.eta * r**2) + 
                                 erfc(sqrt_eta * r) / r)
                        
                        f += coeff * dr
            
            forces[i] = f
        
        return forces
    
    def _compute_force_reciprocal_space(self, gvector):
        """Differentiate S(G) at fixed cell; F=-dE/dR."""
        cart, _, phases, structure, weight = self._reciprocal_terms(gvector)
        coeff = weight[:, None] * structure[:, None] * phases.conj()
        coeff *= self.charges[None, :]
        return 4 * PI / self.volume * np.real(1j * coeff.T @ cart)

    def _compute_force_recip_simple(self):
        return self._compute_force_reciprocal_space(None)

    def _compute_stress(self, gvector):
        """Stress=-dE/dstrain/V, at fixed fractional positions and eta.

        R transforms with the cell while G transforms contragrediently.
        Hence S(G) is strain-independent. Self energy has zero derivative;
        the neutralizing-background term contributes E_G0/V to the diagonal.
        See pydft-book/ewald.tex for the explicit strain derivatives.
        """
        stress = np.zeros((3, 3))
        sqrt_eta = np.sqrt(self.eta)
        for cell in self.nn_cells:
            for i in range(self.natoms):
                for j in range(self.natoms):
                    if i == j and np.all(cell == 0):
                        continue
                    dr = self._frac_to_cart(self.positions[i] - self.positions[j] - cell)
                    r = np.linalg.norm(dr)
                    if 1e-10 < r <= self.rmax:
                        coeff = self.charges[i] * self.charges[j] * (
                            erfc(sqrt_eta * r) / r**3
                            + 2 * sqrt_eta / np.sqrt(PI) * np.exp(-self.eta*r*r) / r**2)
                        stress += 0.5 * coeff * np.outer(dr, dr) / self.volume

        cart, g2, _, structure, weight = self._reciprocal_terms(gvector)
        energy_g = 2 * PI / self.volume * weight * np.abs(structure)**2
        stress += np.eye(3) * np.sum(energy_g) / self.volume
        stress -= np.einsum('g,gi,gj->ij',
                            2 * energy_g * (1/g2 + 1/(4*self.eta)) / self.volume,
                            cart, cart)
        _, e_g0 = self._compute_energy_self_and_g0()
        stress += np.eye(3) * e_g0 / self.volume
        return stress

    def get_energy(self):
        """Return Ewald energy in Hartree."""
        return self.energy
    
    def get_forces(self):
        """Return forces on ions in Hartree/Bohr."""
        return self.forces.copy()
    
    def get_stress(self):
        """Return stress tensor in Hartree/Bohr^3."""
        return self.stress.copy()


def compute_ewald_energy(lattice, atom_positions, charges, gvector=None):
    """
    Convenience function to compute just the Ewald energy.
    
    Args:
        lattice: Lattice object
        atom_positions: Fractional coordinates, shape (natoms, 3)
        charges: Ion charges, shape (natoms,)
        gvector: Optional GVector object
    
    Returns:
        Ewald energy in Hartree
    """
    ewald = Ewald(lattice, atom_positions, charges, gvector)
    return ewald.get_energy()
