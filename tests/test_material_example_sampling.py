import numpy as np
import pytest

from src.cnt_example import CarbonNanotubeSCF
from src.graphene_example import GrapheneSCF
from src.silicon_example import SiliconSCF


@pytest.mark.parametrize(
    "solver_type,kwargs",
    [
        (SiliconSCF, {}),
        (GrapheneSCF, {}),
        (CarbonNanotubeSCF, {"n": 1, "n_extra_bands": 1}),
    ],
)
def test_material_examples_report_gamma_only_and_do_not_accept_ignored_mesh(
    solver_type, kwargs, capsys
):
    solver = solver_type(ecut=0.5, **kwargs)
    output = capsys.readouterr().out

    assert "Sampling: Gamma-only (1 k-point)" in output
    assert "Ewald" not in output
    assert not hasattr(solver, "ewald")
    assert not hasattr(solver, "kpts")
    assert not hasattr(solver, "nk")

    solver.evals = np.zeros(solver.n_bands)
    solver.occupations = np.zeros(solver.n_bands)
    solver._print_results(0.0)
    result_output = capsys.readouterr().out
    assert "Electronic energy in model potential:" in result_output
    assert "Ewald energy" not in result_output

    solver.rho_g = np.zeros(solver.density_gvec.npw, dtype=complex)
    solver.rho_r = np.ones(solver.fft_shape)
    solver._exc_r = np.zeros(solver.fft_shape)
    solver._v_xc_r = np.zeros(solver.fft_shape)
    assert solver._compute_total_energy() == pytest.approx(0.0)

    with pytest.raises(TypeError, match="kpts_mesh"):
        solver_type(ecut=0.5, kpts_mesh=(2, 2, 2), **kwargs)
