"""
Density mixing schemes for SCF convergence acceleration.

Implements:
- Linear mixing (simple)
- Broyden mixing (modified Broyden II method)

=============================================================================
PEDAGOGICAL NOTES
=============================================================================

THE PROBLEM: SCF INSTABILITY
----------------------------
The naive SCF iteration:
    ρ_in → V_eff → solve H → ψ → ρ_out → repeat with ρ_out

often DIVERGES! The density oscillates wildly instead of converging.

WHY? The map ρ_in → ρ_out is a fixed-point iteration. If the "Jacobian"
(derivative of ρ_out with respect to ρ_in) has eigenvalues > 1, small
changes get amplified and the iteration is unstable.

THE SOLUTION: DENSITY MIXING
----------------------------
Instead of using ρ_out directly, we mix it with ρ_in:
    ρ_next = ρ_in + α × (ρ_out - ρ_in)

where α < 1 is the mixing parameter. This damps the oscillations.

LINEAR MIXING (Simple)
----------------------
    ρ_next = ρ_in + α × residual
    where residual = ρ_out - ρ_in

Pros: Simple, always stable for small enough α
Cons: Slow! Typical α ~ 0.1-0.3 means many iterations

BROYDEN MIXING (Accelerated)
----------------------------
Uses history of (ρ_in, ρ_out) pairs to estimate the inverse Jacobian.
This predicts how the density should change to reach the fixed point.

Key idea: The inverse Jacobian J⁻¹ tells us:
    Δρ_in ≈ J⁻¹ × Δresidual

By accumulating history, we build up a low-rank approximation to J⁻¹.
This dramatically accelerates convergence (typical: 10-20 iterations).

PULAY MIXING (DIIS)
-------------------
Direct Inversion in the Iterative Subspace. Finds the linear combination
of previous densities that minimizes the residual norm:
    ρ_next = Σ_i c_i × ρ_i  where c_i minimize ||residual||²

This is a least-squares problem with a constraint Σc_i = 1.

References:
    Johnson, D.D. Phys. Rev. B 38, 12807 (1988) - Modified Broyden
    Pulay, P. Chem. Phys. Lett. 73, 393 (1980) - DIIS
=============================================================================
"""

import numpy as np


class LinearMixer:
    """
    Simple linear density mixing.
    
    rho_next = rho_in + alpha * (rho_out - rho_in)
    
    Simple but requires small alpha (0.1-0.3) for stability.
    """
    
    def __init__(self, alpha=0.3):
        """
        Initialize linear mixer.
        
        Args:
            alpha: Mixing parameter (0 < alpha <= 1)
        """
        self.alpha = alpha
    
    def mix(self, rho_in, rho_out):
        """
        Perform linear mixing.
        
        Args:
            rho_in: Input density (from previous iteration)
            rho_out: Output density (computed from wavefunctions)
        
        Returns:
            rho_next: Mixed density for next iteration
        """
        return rho_in + self.alpha * (rho_out - rho_in)
    
    def reset(self):
        """Reset mixer state (no state for linear mixing)."""
        pass


class BroydenMixer:
    """
    Modified Broyden mixing (Broyden II method).

    Uses history of input/output densities to estimate the inverse
    Jacobian and accelerate convergence.

    Reference:
        Johnson, D.D. Phys. Rev. B 38, 12807 (1988)

    PEDAGOGICAL NOTE: How Broyden Works
    ------------------------------------
    We want to solve: ρ = F(ρ)  (fixed point problem)
    Equivalently: R(ρ) = F(ρ) - ρ = 0  (root finding)

    Newton's method: ρ_next = ρ - J⁻¹ × R(ρ)
    where J = ∂R/∂ρ is the Jacobian.

    But J is huge (N×N for N grid points)! We can't compute or store it.

    BROYDEN'S TRICK: Build a low-rank approximation to J⁻¹ from history.

    Given pairs (Δρ^i, ΔR^i) where:
        Δρ^i = ρ^{i+1} - ρ^i
        ΔR^i = R^{i+1} - R^i

    We want J⁻¹ such that: J⁻¹ × ΔR^i ≈ Δρ^i (secant condition)

    The update formula modifies ρ_next based on how residuals changed
    in previous iterations.
    """

    def __init__(self, alpha=0.7, n_history=8, omega0=0.01):
        """
        Initialize Broyden mixer.

        Args:
            alpha: Mixing parameter for initial linear mixing step
                   (larger α → more aggressive mixing, but less stable)
            n_history: Maximum number of iterations to keep in history
                       (more history → better J⁻¹ approximation, but more memory)
            omega0: Regularization parameter for matrix inversion
                    (prevents numerical instability when history is nearly dependent)
        """
        self.alpha = alpha
        self.n_history = n_history
        self.omega0 = omega0

        # History storage: lists of (ρ_in, ρ_out) from previous iterations
        self.history_in = []
        self.history_out = []
        self.n_iter = 0
    
    def mix(self, rho_in, rho_out):
        """
        Perform Broyden mixing.
        
        Args:
            rho_in: Input density (flattened complex array)
            rho_out: Output density (flattened complex array)
        
        Returns:
            rho_next: Mixed density for next iteration
        """
        # Ensure arrays are 1D complex
        rho_in = np.asarray(rho_in).flatten()
        rho_out = np.asarray(rho_out).flatten()
        
        # Store in history
        self.history_in.append(rho_in.copy())
        self.history_out.append(rho_out.copy())
        
        # Start with linear mixing step
        rho_next = rho_in + self.alpha * (rho_out - rho_in)
        
        # Apply Broyden correction if we have history
        if len(self.history_in) > 1:
            m = len(self.history_in) - 1
            m = min(m, self.n_history)
            
            # Weights (can be made adaptive)
            omega = np.ones(len(self.history_in))
            
            # Compute overlap matrix A
            A = self._compute_a_matrix(m, omega)
            
            # Compute coefficient matrix C
            C = self._compute_c_matrix(m, omega)
            
            # Compute beta = (A + omega0^2 * I)^(-1)
            beta = self._compute_beta(A)
            
            # Compute gamma = C @ beta
            gamma = C @ beta
            
            # Apply correction
            ng = len(rho_in)
            for n in range(m):
                # Residual differences: ΔR = R_{n+1} - R_n
                # where R_i = rho_out_i - rho_in_i (the SCF residual)
                res_new = self.history_out[n + 1] - self.history_in[n + 1]
                res_old = self.history_out[n] - self.history_in[n]
                dres = res_new - res_old
                drho = self.history_in[n + 1] - self.history_in[n]
                
                # Normalize
                norm = np.sqrt(np.sum(np.abs(dres)**2))
                if norm > 1e-20:
                    dres = dres / norm
                    drho = drho / norm
                
                # Apply correction
                rho_next -= omega[n] * gamma[m, n] * (self.alpha * dres + drho)
        
        # Trim history to maximum size
        if len(self.history_in) > self.n_history:
            self.history_in.pop(0)
            self.history_out.pop(0)
        
        self.n_iter += 1
        return rho_next
    
    def _compute_a_matrix(self, m, omega):
        """
        Compute overlap matrix A.
        
        A_ij = <dR_i | dR_j> * omega_i * omega_j
        where dR_i = R_{i+1} - R_i and R_i = rho_out_i - rho_in_i
        """
        A = np.zeros((m, m), dtype=complex)
        
        for i in range(m):
            # ΔR_i = R_{i+1} - R_i where R = rho_out - rho_in
            res_new_i = self.history_out[i + 1] - self.history_in[i + 1]
            res_old_i = self.history_out[i] - self.history_in[i]
            dres_i = res_new_i - res_old_i
            norm_i = np.sqrt(np.sum(np.abs(dres_i)**2))
            if norm_i > 1e-20:
                dres_i = dres_i / norm_i
            
            for j in range(m):
                res_new_j = self.history_out[j + 1] - self.history_in[j + 1]
                res_old_j = self.history_out[j] - self.history_in[j]
                dres_j = res_new_j - res_old_j
                norm_j = np.sqrt(np.sum(np.abs(dres_j)**2))
                if norm_j > 1e-20:
                    dres_j = dres_j / norm_j
                
                A[i, j] = np.vdot(dres_j, dres_i) * omega[i] * omega[j]
        
        return A
    
    def _compute_c_matrix(self, m, omega):
        """
        Compute coefficient matrix C.
        
        C_ik = <R_i | dR_k> * omega_k
        Dimension: (m+1) x m
        """
        C = np.zeros((m + 1, m), dtype=complex)
        
        for i in range(m + 1):
            res_i = self.history_out[i] - self.history_in[i]
            
            for k in range(m):
                # ΔR_k = R_{k+1} - R_k
                res_new_k = self.history_out[k + 1] - self.history_in[k + 1]
                res_old_k = self.history_out[k] - self.history_in[k]
                dres_k = res_new_k - res_old_k
                norm_k = np.sqrt(np.sum(np.abs(dres_k)**2))
                if norm_k > 1e-20:
                    dres_k = dres_k / norm_k
                
                C[i, k] = np.vdot(dres_k, res_i) * omega[k]
        
        return C
    
    def _compute_beta(self, A):
        """
        Compute beta = (A + omega0^2 * I)^(-1).
        
        The regularization omega0^2 * I prevents ill-conditioning.
        """
        m = A.shape[0]
        A_reg = A + self.omega0**2 * np.eye(m)
        
        try:
            beta = np.linalg.inv(A_reg)
        except np.linalg.LinAlgError:
            # Fallback to pseudo-inverse
            beta = np.linalg.pinv(A_reg)
        
        return beta
    
    def reset(self):
        """Reset mixer state (clear history)."""
        self.history_in = []
        self.history_out = []
        self.n_iter = 0


class PulayMixer:
    """
    Pulay mixing (DIIS - Direct Inversion in the Iterative Subspace).
    
    Minimizes the residual in the space spanned by previous iterations.
    """
    
    def __init__(self, alpha=0.5, n_history=8):
        """
        Initialize Pulay mixer.
        
        Args:
            alpha: Mixing parameter for linear mixing step
            n_history: Maximum history size
        """
        self.alpha = alpha
        self.n_history = n_history
        
        self.history_in = []
        self.history_res = []  # Residuals: rho_out - rho_in
    
    def mix(self, rho_in, rho_out):
        """
        Perform Pulay mixing.
        
        Args:
            rho_in: Input density
            rho_out: Output density
        
        Returns:
            rho_next: Mixed density
        """
        rho_in = np.asarray(rho_in).flatten()
        rho_out = np.asarray(rho_out).flatten()
        
        # Residual
        residual = rho_out - rho_in
        
        # Store in history (after linear mixing)
        rho_mixed = rho_in + self.alpha * residual
        self.history_in.append(rho_mixed.copy())
        self.history_res.append(residual.copy())
        
        # If not enough history, return linear mixing result
        if len(self.history_in) < 2:
            return rho_mixed
        
        m = len(self.history_in)
        
        # Build overlap matrix of residuals
        B = np.zeros((m + 1, m + 1), dtype=complex)
        
        for i in range(m):
            for j in range(m):
                B[i, j] = np.vdot(self.history_res[i], self.history_res[j])
        
        # Lagrange constraint row/column
        B[m, :m] = 1.0
        B[:m, m] = 1.0
        B[m, m] = 0.0
        
        # Right-hand side
        rhs = np.zeros(m + 1, dtype=complex)
        rhs[m] = 1.0
        
        # Solve for coefficients
        try:
            coeffs = np.linalg.solve(B, rhs)
        except np.linalg.LinAlgError:
            coeffs = np.linalg.lstsq(B, rhs, rcond=None)[0]
        
        # Build optimal density
        rho_next = np.zeros_like(rho_in)
        for i in range(m):
            rho_next += coeffs[i] * self.history_in[i]
        
        # Trim history
        if len(self.history_in) > self.n_history:
            self.history_in.pop(0)
            self.history_res.pop(0)
        
        return rho_next
    
    def reset(self):
        """Reset mixer state."""
        self.history_in = []
        self.history_res = []
