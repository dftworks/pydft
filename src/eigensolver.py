"""
Preconditioned Conjugate Gradient (PCG) eigensolver.

Finds the lowest eigenvalues and eigenvectors of the Kohn-Sham Hamiltonian
using a band-by-band optimization approach.

=============================================================================
PEDAGOGICAL NOTES
=============================================================================

WHY NOT JUST DIAGONALIZE THE MATRIX?
-------------------------------------
The Hamiltonian matrix in a plane wave basis has size N_pw × N_pw, where
N_pw can be 10,000 to 1,000,000 for real calculations. Direct diagonalization
is O(N³), which is prohibitively expensive.

THE PCG ALTERNATIVE
-------------------
Iterative eigensolvers find the lowest few eigenvalues without building
the full matrix! We only need to apply H|ψ⟩, which is O(N log N) via FFT.

The key insight: minimize the Rayleigh quotient
    ε = ⟨ψ|H|ψ⟩ / ⟨ψ|ψ⟩

The minimum over all |ψ⟩ is the ground state energy ε₀ with ψ = ψ₀.

BAND-BY-BAND APPROACH
---------------------
1. Find the lowest eigenvalue ε₀ and eigenvector ψ₀
2. Find ε₁ by minimizing in the subspace orthogonal to ψ₀
3. Continue for ε₂, ε₃, ...

Each minimization uses Conjugate Gradient (CG), accelerated by:
- PRECONDITIONING: The Teter-Payne-Allan preconditioner makes CG converge
  faster by approximately inverting the kinetic energy operator

THE 2×2 SUBSPACE OPTIMIZATION
-----------------------------
Instead of a line search, we solve a 2×2 eigenvalue problem in the space
{|ψ⟩, |d⟩} where |d⟩ is the search direction. This gives the optimal
linear combination exactly!

Reference:
    Teter, Payne, Allan. Phys. Rev. B 40, 12255 (1989)
=============================================================================
"""

import numpy as np


class PCGEigensolver:
    """
    Preconditioned Conjugate Gradient eigensolver for Kohn-Sham equations.
    
    Optimizes one band at a time, orthogonalizing to previously converged bands.
    Uses the Teter-Payne-Allan preconditioner.
    
    Reference:
        Teter, Payne, Allan. Phys. Rev. B 40, 12255 (1989)
    """
    
    def __init__(self, npw, n_bands):
        """
        Initialize PCG solver.
        
        Args:
            npw: Number of plane waves
            n_bands: Number of bands to compute
        """
        self.npw = npw
        self.n_bands = n_bands
        
        # Workspace vectors
        self._x0 = np.zeros(npw, dtype=complex)
        self._h_x0 = np.zeros(npw, dtype=complex)
        self._d0 = np.zeros(npw, dtype=complex)
        self._h_d0 = np.zeros(npw, dtype=complex)
        self._g0 = np.zeros(npw, dtype=complex)
        self._g1 = np.zeros(npw, dtype=complex)
        self._pg0 = np.zeros(npw, dtype=complex)
        self._pg1 = np.zeros(npw, dtype=complex)
        self._precond = np.zeros(npw, dtype=complex)
    
    def solve(self, ham_apply, ham_diag, evecs, evals,
              tol=1e-6, max_iter=100):
        """
        Solve for eigenvalues and eigenvectors.

        Args:
            ham_apply: Function to apply Hamiltonian: h_psi = ham_apply(psi)
            ham_diag: Diagonal of Hamiltonian (kinetic energies)
            evecs: (npw, n_bands) initial guess, overwritten with eigenvectors
            evals: (n_bands,) array, overwritten with eigenvalues
            tol: Convergence tolerance for eigenvalues
            max_iter: Maximum CG iterations per band

        Returns:
            n_converged: Number of bands converged
            n_hpsi: Total number of H|psi> applications

        PEDAGOGICAL NOTE: The Band-by-Band Algorithm
        ---------------------------------------------
        For each band i:
            1. Orthogonalize ψ_i to all lower bands ψ_0, ..., ψ_{i-1}
            2. Minimize ⟨ψ_i|H|ψ_i⟩ subject to orthogonality constraints
            3. The minimum is ε_i (the i-th eigenvalue)

        The orthogonality constraint ensures we find excited states,
        not just the ground state over and over!

        WHY THIS WORKS: Variational Principle
        If ψ is orthogonal to the true eigenstates ψ_0, ..., ψ_{i-1},
        then ⟨ψ|H|ψ⟩ ≥ ε_i, with equality when ψ = ψ_i.
        """
        n_hpsi = 0
        n_converged = 0

        for iband in range(self.n_bands):
            # ===== Orthogonalize to lower bands =====
            # This ensures we find the i-th eigenvalue, not a lower one
            if iband > 0:
                self._gram_schmidt(evecs, iband)

            # Copy current band to workspace
            self._x0[:] = evecs[:, iband]
            self._normalize(self._x0)

            # Apply Hamiltonian: |H·x₀⟩
            self._h_x0[:] = ham_apply(self._x0)
            n_hpsi += 1

            # Initial eigenvalue estimate: ω = ⟨x₀|H|x₀⟩
            # This is the Rayleigh quotient (since ⟨x₀|x₀⟩ = 1)
            omega0 = np.real(np.vdot(self._x0, self._h_x0))
            
            # ===== CG iterations for this band =====
            # We minimize ⟨x|H|x⟩ using conjugate gradient
            converged = False
            self._d0.fill(0)  # Search direction (starts at zero)

            for cg_iter in range(max_iter):
                # ===== STEP 1: Compute preconditioner =====
                # P ≈ T⁻¹ to accelerate convergence (see _compute_preconditioner)
                self._compute_preconditioner(self._x0, ham_diag)

                # ===== STEP 2: Compute gradient of Rayleigh quotient =====
                # The gradient of ε = ⟨x|H|x⟩/⟨x|x⟩ is:
                #   g = 2(H|x⟩ - ε|x⟩) / ⟨x|x⟩ = 2(H - ε)|x⟩  (for normalized x)
                # We drop the factor of 2 since it doesn't affect the direction
                self._g1[:] = self._h_x0 - omega0 * self._x0

                # Must orthogonalize gradient to lower bands to stay in subspace
                self._orthogonalize_to_bands(evecs, iband, self._g1)

                # ===== STEP 3: Apply preconditioner =====
                # pg = P·g  (preconditioned gradient)
                # Good preconditioner makes P·H ≈ I, improving convergence
                self._pg1[:] = self._g1 * self._precond

                # Must re-orthogonalize after preconditioning
                self._orthogonalize_to_bands(evecs, iband, self._pg1)
                self._normalize(self._pg1)

                # ===== STEP 4: Compute CG beta (Polak-Ribiere formula) =====
                # β determines how much of the old direction to keep
                # Polak-Ribiere: β = ⟨g_new|pg_new - pg_old⟩ / ⟨g_old|pg_old⟩
                if cg_iter == 0:
                    beta = 0.0  # First iteration: steepest descent
                else:
                    num = np.vdot(self._g1, self._pg1 - self._pg0)
                    denom = np.vdot(self._pg0, self._g0)
                    if abs(denom) > 1e-20:
                        beta = max(0.0, np.real(num / denom))  # Ensure β ≥ 0
                    else:
                        beta = 0.0

                # ===== STEP 5: Update search direction =====
                # d = -pg + β·d_old  (negative because we minimize)
                self._d0[:] = -self._pg1 + beta * self._d0

                # Orthogonalize d to current x (stay on unit sphere)
                proj = np.vdot(self._x0, self._d0)
                self._d0 -= proj * self._x0
                self._normalize(self._d0)

                # ===== STEP 6: Apply H to search direction =====
                self._h_d0[:] = ham_apply(self._d0)
                n_hpsi += 1

                # ===== STEP 7: Find optimal step via 2×2 subspace =====
                # Instead of line search, we diagonalize H in {x, d} subspace
                # This gives the exact minimum in this 2D space!
                alpha = self._get_optimal_step()

                # ===== STEP 8: Update wavefunction =====
                # x_new = cos(θ)·x + sin(θ)·d  where tan(θ) = α
                # This parameterization keeps x_new normalized
                t = np.sqrt(1.0 + np.abs(alpha)**2)
                cs = 1.0 / t  # cos(θ)
                sn = alpha / t  # sin(θ)

                self._x0[:] = cs * self._x0 + sn * self._d0
                self._h_x0[:] = cs * self._h_x0 + sn * self._h_d0  # Update H|x⟩ too

                # ===== STEP 9: New eigenvalue estimate =====
                omega = np.real(np.vdot(self._x0, self._h_x0))

                # ===== STEP 10: Check convergence =====
                if abs(omega - omega0) < tol:
                    converged = True
                    n_converged += 1
                    break

                # Save for next iteration (needed for Polak-Ribiere β)
                self._pg0[:] = self._pg1
                self._g0[:] = self._g1
                omega0 = omega
            
            # Store results
            evals[iband] = omega0 if not converged else omega
            evecs[:, iband] = self._x0
        
        return n_converged, n_hpsi
    
    def _gram_schmidt(self, evecs, iband):
        """Orthogonalize band iband to all lower bands."""
        v = evecs[:, iband].copy()
        
        for j in range(iband):
            proj = np.vdot(evecs[:, j], v)
            v -= proj * evecs[:, j]
        
        v /= np.linalg.norm(v)
        evecs[:, iband] = v
    
    def _orthogonalize_to_bands(self, evecs, n_bands, v):
        """Orthogonalize vector v to first n_bands bands."""
        for j in range(n_bands):
            proj = np.vdot(evecs[:, j], v)
            v -= proj * evecs[:, j]
    
    def _normalize(self, v):
        """Normalize vector in place."""
        norm = np.linalg.norm(v)
        if norm > 1e-20:
            v /= norm
    
    def _compute_preconditioner(self, psi, kin):
        """
        Compute Teter-Payne-Allan preconditioner.

        P(G) = (27 + 18x + 12x^2 + 8x^3) / (27 + 18x + 12x^2 + 8x^3 + 16x^4)
        where x = E_kin(G) / (1.5 * E_avg)

        PEDAGOGICAL NOTE: Why Preconditioning?
        ---------------------------------------
        The condition number of H (ratio of largest to smallest eigenvalue)
        determines CG convergence. For plane waves:
            - Smallest eigenvalue: ~ε_lowest (ground state energy)
            - Largest eigenvalue: ~½|G_max|² (highest kinetic energy)

        This ratio can be 10,000:1 or worse, making CG very slow!

        SOLUTION: Preconditioner P such that P·H has better conditioning.

        THE IDEAL PRECONDITIONER
        If P = H⁻¹, then P·H = I (perfect conditioning, 1 iteration!)
        But H⁻¹ is expensive to compute.

        THE TPA APPROXIMATION
        Teter, Payne, Allan observed:
            - H ≈ T + V, where T (kinetic) dominates for large |G|
            - So P ≈ T⁻¹ = 2/|G|² for large |G|
            - But must be smooth near G=0 to avoid instability

        The TPA formula smoothly interpolates:
            - P(G) → 2/(1.5·ε_avg) as |G| → 0  (bounded)
            - P(G) → 2/|G|² as |G| → ∞  (like T⁻¹)

        Reference: Phys. Rev. B 40, 12255 (1989)
        """
        # Average kinetic energy: ⟨ψ|T|ψ⟩ = Σ_G |c(G)|² · ½|G|²
        ek = np.sum(np.abs(psi)**2 * kin)
        ek = max(ek, 1e-10)  # Avoid division by zero

        # Dimensionless variable: x = T(G) / (1.5 × ε_avg)
        x = kin / (1.5 * ek)
        x2 = x * x
        x3 = x * x2
        x4 = x * x3

        # TPA formula: smooth transition from constant (x→0) to 1/x² (x→∞)
        y = 27.0 + 18.0 * x + 12.0 * x2 + 8.0 * x3
        self._precond[:] = y / (y + 16.0 * x4)

        # Scale factor to match dimensions (energy⁻¹)
        self._precond *= 2.0 / (1.5 * ek)
    
    def _get_optimal_step(self):
        """
        Find optimal step by solving 2x2 generalized eigenvalue problem.
        
        Returns the optimal mixing coefficient alpha.
        """
        # Build 2x2 Hamiltonian and overlap matrices
        h00 = np.vdot(self._x0, self._h_x0)
        h01 = np.vdot(self._x0, self._h_d0)
        h10 = np.vdot(self._d0, self._h_x0)
        h11 = np.vdot(self._d0, self._h_d0)
        
        s00 = np.vdot(self._x0, self._x0)
        s01 = np.vdot(self._x0, self._d0)
        s10 = np.vdot(self._d0, self._x0)
        s11 = np.vdot(self._d0, self._d0)
        
        # Form matrices
        H = np.array([[h00, h01], [h10, h11]])
        S = np.array([[s00, s01], [s10, s11]])
        
        # Solve generalized eigenvalue problem
        try:
            # S^(-1) * H
            S_inv = np.linalg.inv(S)
            M = S_inv @ H
            
            # Get eigenvector for lowest eigenvalue
            evals, evecs = np.linalg.eig(M)
            idx = np.argmin(np.real(evals))
            ev = evecs[:, idx]
            
            # alpha = ev[1] / ev[0]
            if abs(ev[0]) > 1e-20:
                alpha = ev[1] / ev[0]
            else:
                alpha = 0.0
        except:
            alpha = 0.0
        
        return alpha


def random_initial_guess(npw, n_bands):
    """
    Generate random initial guess for eigenvectors.
    
    Args:
        npw: Number of plane waves
        n_bands: Number of bands
    
    Returns:
        (npw, n_bands) array of orthonormalized random vectors
    """
    evecs = np.random.randn(npw, n_bands) + 1j * np.random.randn(npw, n_bands)
    
    # Orthonormalize using Gram-Schmidt
    for i in range(n_bands):
        for j in range(i):
            proj = np.vdot(evecs[:, j], evecs[:, i])
            evecs[:, i] -= proj * evecs[:, j]
        evecs[:, i] /= np.linalg.norm(evecs[:, i])
    
    return evecs
