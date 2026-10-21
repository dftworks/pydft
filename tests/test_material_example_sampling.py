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
    assert not hasattr(solver, "kpts")
    assert not hasattr(solver, "nk")
    with pytest.raises(TypeError, match="kpts_mesh"):
        solver_type(ecut=0.5, kpts_mesh=(2, 2, 2), **kwargs)
