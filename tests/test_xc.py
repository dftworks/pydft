import numpy as np
import pytest
from src.xc import lda_xc, slater_exchange, pz_correlation

def test_slater_exchange():
    """Verify Slater exchange relationship V_x = 4/3 * epsilon_x."""
    rho = np.linspace(0.01, 1.0, 10)
    vx, ex = slater_exchange(rho)
    
    # Relationship: V_x = 4/3 * eps_x
    np.testing.assert_allclose(vx, (4.0/3.0) * ex, rtol=1e-10)
    
    # Check physical values (eps_x should be negative)
    assert np.all(ex < 0)
    assert np.all(vx < 0)

def test_pz_correlation():
    """Verify Perdew-Zunger correlation properties."""
    # Test both high and low density regimes
    rho = np.array([1.0, 0.001]) # rs < 1 and rs > 1
    vc, ec = pz_correlation(rho)
    
    # Correlation energy should be negative
    assert np.all(ec < 0)
    assert np.all(vc < 0)
    
    # ec should be more negative at higher density (rs smaller)
    assert ec[0] < ec[1]

def test_lda_xc_interface():
    """Test the combined LDA XC interface."""
    rho = np.random.rand(10, 10, 10)
    vxc, exc = lda_xc(rho)
    
    assert vxc.shape == rho.shape
    assert exc.shape == rho.shape
    
    # Overall XC energy density should be negative
    assert np.all(exc < 0)
