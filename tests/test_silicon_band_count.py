"""Silicon examples size occupations and gap indices from electron count."""
import numpy as np
import pytest

from src.silicon_example import SiliconSCF
from src.silicon_bandstructure import SiliconBandStructure


@pytest.mark.parametrize("n_bands", [6, 10])
def test_silicon_scf_runs_with_nondefault_band_count(n_bands):
    solver = SiliconSCF(ecut=2.0, n_bands=n_bands)

    with pytest.warns(RuntimeWarning, match="did not converge"):
        energy = solver.run(max_iter=1, verbose=False)

    assert np.isfinite(energy)
    assert solver.occupations.shape == (n_bands,)
    np.testing.assert_array_equal(
        solver.occupations,
        np.clip(solver.n_electrons - 2 * np.arange(n_bands), 0, 2))


@pytest.mark.parametrize("n_bands", [6, 10])
def test_silicon_bandstructure_runs_with_nondefault_band_count(n_bands):
    solver = SiliconBandStructure(ecut=2.0, n_bands=n_bands)

    with pytest.warns(RuntimeWarning, match="did not converge"):
        evals = solver.run_scf(max_iter=1)

    assert evals.shape == (n_bands,)


@pytest.mark.parametrize("cls", [SiliconSCF, SiliconBandStructure])
def test_silicon_examples_reject_too_few_bands(cls):
    with pytest.raises(ValueError, match="cannot hold 8 electrons"):
        cls(ecut=2.0, n_bands=3)
