"""Unsupported UPF physics must fail before constructing an operator."""
import pytest

from src.nonlocal_potential import NonlocalPotential
from src.upf_reader import UPFPseudopotential


def write_upf(path, *, dij, core_correction="F", nlcc=""):
    path.write_text(f"""<UPF version="2.0.1">
  <PP_HEADER element="X" z_valence="1" l_max="0" l_local="0"
             mesh_size="3" number_of_proj="2"
             core_correction="{core_correction}" />
  <PP_MESH><PP_R>0.1 0.5 1.0</PP_R><PP_RAB>0.1 0.4 0.5</PP_RAB></PP_MESH>
  <PP_LOCAL>0 0 0</PP_LOCAL>
  <PP_NONLOCAL>
    <PP_BETA.1 index="1" angular_momentum="0">1 0 0</PP_BETA.1>
    <PP_BETA.2 index="2" angular_momentum="0">0 1 0</PP_BETA.2>
    <PP_DIJ>{dij}</PP_DIJ>
  </PP_NONLOCAL>
  <PP_RHOATOM>0 0 0</PP_RHOATOM>
  {nlcc}
</UPF>""")


def test_nonlocal_potential_rejects_offdiagonal_dij(tmp_path):
    path = tmp_path / "offdiagonal.upf"
    write_upf(path, dij="2.0 0.4 0.4 3.0")
    psp = UPFPseudopotential(path)

    assert psp.dij[0, 1] != 0
    with pytest.raises(NotImplementedError, match="off-diagonal PP_DIJ"):
        NonlocalPotential(psp, [], None, 1.0)


def test_nonlocal_potential_rejects_nlcc(tmp_path):
    path = tmp_path / "nlcc.upf"
    write_upf(
        path, dij="2.0 0.0 0.0 3.0", core_correction="T",
        nlcc="<PP_NLCC>0.1 0.05 0</PP_NLCC>")
    psp = UPFPseudopotential(path)

    assert psp.has_nlcc
    assert psp.rhocore.size == 3
    with pytest.raises(NotImplementedError, match="PP_NLCC"):
        NonlocalPotential(psp, [], None, 1.0)
