#!/usr/bin/env python3
"""
Non-local Pseudopotential Handling for Plane-Wave DFT.

This module provides classes to compute local and non-local pseudopotential
contributions in G-space from UPF files.

The Kleinman-Bylander separable form is used for the non-local potential:
    V_nl |psi> = sum_i D_ii |beta_i> <beta_i|psi>
"""

import numpy as np
from .constants import TWOPI
from .upf_reader import UPFPseudopotential


class NonlocalPotential:
    """
    Handles non-local pseudopotential calculations for a periodic system.
    
    This class computes:
    - Local potential V_loc(G) from UPF data
    - Non-local projectors beta(|k+G|) in reciprocal space
    - Application of V_nl to wavefunctions
    
    Attributes:
        psp: UPFPseudopotential object with raw pseudopotential data
        atom_positions: Array of fractional atomic positions (natoms, 3)
        gvec: GVector object with reciprocal lattice information
        vloc_g: Local potential in G-space (complex array)
        beta_g: List of beta projectors at Gamma point
    """
    
    def __init__(self, psp, atom_positions, gvec, volume):
        """
        Initialize the non-local potential handler.
        
        Args:
            psp: UPFPseudopotential object
            atom_positions: Fractional atomic positions (natoms, 3)
            gvec: GVector object
            volume: Cell volume in Bohr^3
        """
        self.psp = psp
        self.atom_positions = np.asarray(atom_positions)
        self.gvec = gvec
        self.volume = volume
        self.npw = gvec.npw
        
        # Compute V_loc(G) with structure factors
        self.vloc_g = self._compute_vloc_g()
        
        # Compute beta projectors at Gamma point
        self.beta_g = self._compute_beta_g(gvec.norms)
        
    def _compute_vloc_g(self):
        """
        Compute local potential in G-space from UPF.
        
        Returns:
            Complex array of V_loc(G) values with structure factors
        """
        # Get unique G-shell values
        g_unique = np.unique(np.round(self.gvec.norms, decimals=8))
        
        # Compute V_loc for each shell
        vloc_shells = self.psp.get_vloc_g(g_unique, self.volume)
        
        # Map shells to G-vectors
        vloc_g = np.zeros(self.npw, dtype=complex)
        
        # Create shell index map
        shell_map = {round(g, 8): i for i, g in enumerate(g_unique)}
        
        # Structure factor for each atom
        for atom_pos in self.atom_positions:
            for ig in range(self.npw):
                m = self.gvec.miller[ig]
                g_norm = round(self.gvec.norms[ig], 8)
                shell_idx = shell_map.get(g_norm, 0)
                
                # Phase factor: exp(i G . tau)
                phase = TWOPI * (m[0]*atom_pos[0] + m[1]*atom_pos[1] + m[2]*atom_pos[2])
                sfact = np.exp(1j * phase)
                
                vloc_g[ig] += vloc_shells[shell_idx] * sfact
        
        return vloc_g
    
    def _compute_beta_g(self, kg_norms):
        """
        Compute beta projectors in reciprocal space.
        
        Args:
            kg_norms: |k+G| values (array of length npw)
            
        Returns:
            List of beta projector arrays
        """
        return self.psp.get_beta_kg(kg_norms, self.volume)
    
    def get_vloc_g(self):
        """Get the local potential in G-space."""
        return self.vloc_g
    
    def compute_structure_factor_phases(self, atom_pos):
        """
        Compute structure factor phases for a single atom.
        
        Args:
            atom_pos: Fractional position of atom (3,)
            
        Returns:
            Complex array of phases exp(i G . tau)
        """
        phases = np.zeros(self.npw, dtype=complex)
        for ig in range(self.npw):
            m = self.gvec.miller[ig]
            phase = TWOPI * (m[0]*atom_pos[0] + m[1]*atom_pos[1] + m[2]*atom_pos[2])
            phases[ig] = np.exp(1j * phase)
        return phases
    
    def apply_vnl(self, psi_g, beta_kg=None):
        """
        Apply non-local potential to wavefunction.
        
        V_nl |psi> = sum_{atoms} sum_i D_ii |beta_i> <beta_i|psi>
        
        Args:
            psi_g: Wavefunction in G-space (npw,)
            beta_kg: Optional k-dependent beta projectors. If None, uses Gamma-point values.
            
        Returns:
            V_nl |psi> in G-space
        """
        if beta_kg is None:
            beta_kg = self.beta_g
            
        vnl_psi = np.zeros_like(psi_g)
        
        for atom_pos in self.atom_positions:
            phases = self.compute_structure_factor_phases(atom_pos)
            
            for ibeta in range(self.psp.nbeta):
                beta = beta_kg[ibeta]
                d_ii = self.psp.get_dfact(ibeta)
                
                # <beta|psi> with structure factor
                proj = np.sum(np.conj(beta * phases) * psi_g)
                
                # |beta> D <beta|psi>
                vnl_psi += d_ii * proj * beta * phases
        
        return vnl_psi
    
    def get_beta_kg(self, k_cart):
        """
        Compute beta projectors for a specific k-point.
        
        Args:
            k_cart: k-vector in Cartesian coordinates (3,)
            
        Returns:
            List of beta projector arrays for this k-point
        """
        # Compute |k+G| for all G-vectors
        kg_norms = np.zeros(self.npw)
        for ig in range(self.npw):
            kpg = k_cart + self.gvec.cart[ig]
            kg_norms[ig] = np.linalg.norm(kpg)
        
        return self._compute_beta_g(kg_norms)
    
    def apply_vnl_k(self, psi_g, k_cart):
        """
        Apply non-local potential at a specific k-point.
        
        Args:
            psi_g: Wavefunction in G-space
            k_cart: k-vector in Cartesian coordinates
            
        Returns:
            V_nl |psi> in G-space
        """
        beta_kg = self.get_beta_kg(k_cart)
        return self.apply_vnl(psi_g, beta_kg)
    
    @property
    def zion(self):
        """Ionic charge (valence electrons per atom)."""
        return self.psp.zion
    
    @property
    def nbeta(self):
        """Number of beta projectors."""
        return self.psp.nbeta
    
    def display(self):
        """Display pseudopotential information."""
        self.psp.display()


class PseudopotentialSet:
    """
    Manages pseudopotentials for multiple atomic species.
    
    This class handles systems with multiple elements, each having
    its own pseudopotential file.
    """
    
    def __init__(self):
        """Initialize empty pseudopotential set."""
        self.pseudopotentials = {}  # element -> UPFPseudopotential
        self.nonlocal_pots = {}     # element -> NonlocalPotential
    
    def add_species(self, element, upf_path):
        """
        Add a pseudopotential for an element.
        
        Args:
            element: Element symbol (e.g., 'Si', 'O')
            upf_path: Path to UPF file
        """
        self.pseudopotentials[element] = UPFPseudopotential(upf_path)
    
    def setup_for_crystal(self, crystal, gvec, volume):
        """
        Set up non-local potentials for all species in a crystal.
        
        Args:
            crystal: Crystal object with atom information
            gvec: GVector object
            volume: Cell volume
        """
        # Group atoms by element
        atoms_by_element = {}
        for atom in crystal.atoms:
            if atom.symbol not in atoms_by_element:
                atoms_by_element[atom.symbol] = []
            atoms_by_element[atom.symbol].append(atom.position)
        
        # Create NonlocalPotential for each element
        for element, positions in atoms_by_element.items():
            if element not in self.pseudopotentials:
                raise ValueError(f"No pseudopotential loaded for element {element}")
            
            self.nonlocal_pots[element] = NonlocalPotential(
                psp=self.pseudopotentials[element],
                atom_positions=np.array(positions),
                gvec=gvec,
                volume=volume
            )
    
    def get_total_vloc_g(self, gvec):
        """
        Get total local potential from all atoms.
        
        Returns:
            Complex array of V_loc(G)
        """
        vloc_total = np.zeros(gvec.npw, dtype=complex)
        for nlpot in self.nonlocal_pots.values():
            vloc_total += nlpot.get_vloc_g()
        return vloc_total
    
    def apply_vnl_all(self, psi_g, k_cart=None):
        """
        Apply non-local potential from all atoms.
        
        Args:
            psi_g: Wavefunction in G-space
            k_cart: Optional k-vector (uses Gamma if None)
            
        Returns:
            V_nl |psi> in G-space
        """
        vnl_psi = np.zeros_like(psi_g)
        for nlpot in self.nonlocal_pots.values():
            if k_cart is None:
                vnl_psi += nlpot.apply_vnl(psi_g)
            else:
                vnl_psi += nlpot.apply_vnl_k(psi_g, k_cart)
        return vnl_psi
    
    def get_total_valence_electrons(self):
        """Get total number of valence electrons."""
        total = 0
        for element, nlpot in self.nonlocal_pots.items():
            natoms = len(nlpot.atom_positions)
            total += nlpot.zion * natoms
        return int(total)
