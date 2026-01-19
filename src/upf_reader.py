"""
UPF Pseudopotential File Reader.

Reads Unified Pseudopotential Format (UPF) files used by Quantum ESPRESSO
and other DFT codes. Supports UPF version 2.0 XML format.

This implementation is based on the Rust dftworks UPF parser.
"""

import numpy as np
import xml.etree.ElementTree as ET
import re
from scipy.integrate import simpson
from scipy.special import spherical_jn, erf
from .constants import RY_TO_HA, FOURPI


class UPFPseudopotential:
    """
    Norm-conserving pseudopotential from UPF file.
    
    Attributes:
        element: Element symbol
        zion: Number of valence electrons
        lmax: Maximum angular momentum
        lloc: Local channel (-1 if separate local potential)
        nbeta: Number of projectors
        mmax: Radial mesh size
        rad: Radial grid points
        rab: dr for integration (r * d_log_r for log grids)
        vloc: Local potential V_loc(r) in Hartree
        beta: List of projector functions beta_l(r)
        lbeta: Angular momentum for each projector
        dij: D_ij matrix elements (Hartree)
        rhoatom: Atomic charge density
        rhocore: Core charge density (for NLCC)
        has_nlcc: Whether non-linear core correction is present
    """
    
    def __init__(self, filename=None):
        """Initialize and optionally read UPF file."""
        self.element = ""
        self.zion = 0.0
        self.zatom = 0.0
        self.lmax = 0
        self.lloc = -1
        self.nbeta = 0
        self.mmax = 0
        
        self.rad = np.array([])
        self.rab = np.array([])
        self.vloc = np.array([])
        self.beta = []
        self.lbeta = []
        self.dij = np.array([])
        self.rhoatom = np.array([])
        self.rhocore = np.array([])
        self.has_nlcc = False
        
        if filename is not None:
            self.read_file(filename)
    
    def read_file(self, filename):
        """
        Read and parse UPF file.
        
        Args:
            filename: Path to UPF file
        """
        print(f"Reading pseudopotential: {filename}")
        
        # Parse XML
        tree = ET.parse(filename)
        root = tree.getroot()
        
        # Parse header
        header = root.find('.//PP_HEADER')
        if header is not None:
            self._parse_header(header)
        
        # Parse mesh
        mesh = root.find('.//PP_MESH')
        if mesh is not None:
            self._parse_mesh(mesh)
        
        # Parse local potential
        local = root.find('.//PP_LOCAL')
        if local is not None:
            self._parse_local(local)
        
        # Parse nonlocal (projectors)
        nonlocal_elem = root.find('.//PP_NONLOCAL')
        if nonlocal_elem is not None:
            self._parse_nonlocal(nonlocal_elem)
        
        # Parse atomic density
        rhoatom = root.find('.//PP_RHOATOM')
        if rhoatom is not None:
            self._parse_rhoatom(rhoatom)
        
        # Parse NLCC
        nlcc = root.find('.//PP_NLCC')
        if nlcc is not None:
            self._parse_nlcc(nlcc)
        
        print(f"  Element: {self.element}")
        print(f"  Z_valence: {self.zion}")
        print(f"  L_max: {self.lmax}")
        print(f"  N_projectors: {self.nbeta}")
        print(f"  Mesh points: {self.mmax}")
    
    def _parse_header(self, header):
        """Parse PP_HEADER element."""
        self.element = header.get('element', '').strip()
        
        z_val = header.get('z_valence', '0')
        self.zion = float(z_val)
        
        l_max = header.get('l_max', '0')
        self.lmax = int(l_max)
        
        l_local = header.get('l_local', '-1')
        self.lloc = int(l_local)
        
        mesh_size = header.get('mesh_size', '0')
        self.mmax = int(mesh_size)
        
        n_proj = header.get('number_of_proj', '0')
        self.nbeta = int(n_proj)
        
        core_corr = header.get('core_correction', 'F')
        self.has_nlcc = core_corr.strip().upper() in ('T', 'TRUE', '.TRUE.')
    
    def _parse_mesh(self, mesh):
        """Parse PP_MESH element."""
        r_elem = mesh.find('PP_R')
        if r_elem is not None and r_elem.text:
            self.rad = self._parse_array(r_elem.text)
        
        rab_elem = mesh.find('PP_RAB')
        if rab_elem is not None and rab_elem.text:
            self.rab = self._parse_array(rab_elem.text)
        
        # Update mmax from actual array length
        if len(self.rad) > 0:
            self.mmax = len(self.rad)
    
    def _parse_local(self, local):
        """Parse PP_LOCAL element (local potential)."""
        if local.text:
            # UPF stores in Rydberg, convert to Hartree
            self.vloc = self._parse_array(local.text) * RY_TO_HA
    
    def _parse_nonlocal(self, nonlocal_elem):
        """Parse PP_NONLOCAL element (projectors and D_ij)."""
        self.beta = []
        self.lbeta = []
        
        # Find all beta projectors
        for child in nonlocal_elem:
            if 'PP_BETA' in child.tag:
                l = int(child.get('angular_momentum', '0'))
                self.lbeta.append(l)
                
                if child.text:
                    # UPF stores beta in Ry units, multiply by Ry_to_Ha
                    beta = self._parse_array(child.text) * RY_TO_HA
                    self.beta.append(beta)
        
        # Parse D_ij matrix
        dij_elem = nonlocal_elem.find('PP_DIJ')
        if dij_elem is not None and dij_elem.text:
            dij_flat = self._parse_array(dij_elem.text) / RY_TO_HA  # Divide for D matrix
            n = self.nbeta
            if len(dij_flat) >= n * n:
                self.dij = dij_flat[:n*n].reshape((n, n))
            else:
                self.dij = np.zeros((n, n))
                for i, val in enumerate(dij_flat):
                    self.dij[i // n, i % n] = val
    
    def _parse_rhoatom(self, rhoatom):
        """Parse PP_RHOATOM element."""
        if rhoatom.text:
            self.rhoatom = self._parse_array(rhoatom.text)
    
    def _parse_nlcc(self, nlcc):
        """Parse PP_NLCC element (core charge)."""
        if nlcc.text:
            self.rhocore = self._parse_array(nlcc.text)
            self.has_nlcc = True
    
    def _parse_array(self, text):
        """Parse whitespace-separated numbers from text."""
        values = text.split()
        return np.array([float(v) for v in values])
    
    def get_vloc_g(self, gshells, volume):
        """
        Compute local potential in G-space.
        
        V_loc(G) = (4*pi/V) * int [r*V_loc(r) + Z*erf(r)] * sin(Gr)/G dr
        
        Args:
            gshells: Array of unique |G| values
            volume: Cell volume
        
        Returns:
            vloc_g: V_loc(G) for each shell
        """
        nshells = len(gshells)
        vloc_g = np.zeros(nshells)
        
        r = self.rad
        vloc = self.vloc
        zion = self.zion
        rab = self.rab
        
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
        vloc_g *= FOURPI / volume
        
        return vloc_g
    
    def get_beta_kg(self, kg, volume):
        """
        Compute projector functions in G-space.
        
        beta_l(|k+G|) = (4*pi/sqrt(V)) * int beta_l(r) * r * j_l(|k+G|*r) dr
        
        Args:
            kg: Array of |k+G| values
            volume: Cell volume
        
        Returns:
            beta_kg: List of arrays, one per projector
        """
        npw = len(kg)
        prefactor = FOURPI / np.sqrt(volume)
        
        beta_kg = []
        
        for ibeta in range(self.nbeta):
            l = self.lbeta[ibeta]
            beta_r = self.beta[ibeta]
            
            beta_g = np.zeros(npw)
            
            for ipw in range(npw):
                if kg[ipw] < 1e-10:
                    # Special case for |k+G| = 0
                    if l == 0:
                        work = beta_r * self.rad
                        beta_g[ipw] = prefactor * self._integrate_radial(work)
                    else:
                        beta_g[ipw] = 0.0
                else:
                    # j_l(k*r)
                    jl = spherical_jn(l, kg[ipw] * self.rad)
                    work = beta_r * self.rad * jl
                    beta_g[ipw] = prefactor * self._integrate_radial(work)
            
            beta_kg.append(beta_g)
        
        return beta_kg
    
    def _integrate_radial(self, work):
        """Integrate on radial grid using Simpson's rule with rab weights."""
        if len(self.rab) == len(work):
            return simpson(work, x=self.rad)
        else:
            return simpson(work, x=self.rad)
    
    def get_dfact(self, ibeta):
        """Get D_ij diagonal element for projector ibeta."""
        return self.dij[ibeta, ibeta]
    
    def get_atomic_density(self):
        """Return atomic charge density on radial grid."""
        return self.rhoatom.copy()
    
    def display(self):
        """Print pseudopotential information."""
        print("\n" + "=" * 50)
        print("Pseudopotential Information")
        print("=" * 50)
        print(f"Element: {self.element}")
        print(f"Z_valence: {self.zion}")
        print(f"L_max: {self.lmax}")
        print(f"L_local: {self.lloc}")
        print(f"N_projectors: {self.nbeta}")
        print(f"Mesh points: {self.mmax}")
        print(f"Has NLCC: {self.has_nlcc}")
        
        if self.nbeta > 0:
            print("\nProjectors:")
            for i in range(self.nbeta):
                print(f"  Beta {i}: l={self.lbeta[i]}, D_ii={self.dij[i,i]:.6f}")
        
        print("=" * 50)


def read_upf(filename):
    """
    Convenience function to read UPF file.
    
    Args:
        filename: Path to UPF file
    
    Returns:
        UPFPseudopotential object
    """
    return UPFPseudopotential(filename)


def compute_structure_factor(miller, gindex, atom_positions):
    """
    Compute structure factor S(G) for atomic positions.
    
    S(G) = sum_atoms exp(i * G . r_atom)
    
    Args:
        miller: Miller indices for G-vectors
        gindex: Indices into miller array
        atom_positions: Fractional coordinates (natoms, 3)
    
    Returns:
        sfact: Complex structure factor for each G
    """
    from .constants import TWOPI
    
    npw = len(gindex)
    natoms = len(atom_positions)
    
    sfact = np.zeros(npw, dtype=complex)
    
    for ig in range(npw):
        m = miller[gindex[ig]] if gindex is not None else miller[ig]
        
        for pos in atom_positions:
            phase = TWOPI * (m[0]*pos[0] + m[1]*pos[1] + m[2]*pos[2])
            sfact[ig] += np.exp(1j * phase)
    
    return sfact
