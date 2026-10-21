"""Custom example SCF drivers must reject energy-only convergence."""
from types import SimpleNamespace

import numpy as np
import pytest

from src import cnt_example, graphene_example, silicon_bandstructure
from src import silicon_example, silicon_upf
from src import hamiltonian


class HoldingMixer:
    """Keep the old density so each fake output has a nonzero residual."""

    @staticmethod
    def mix(old, new):
        return old.copy()


class FakeEigensolver:
    def solve(self, **kwargs):
        pass


class CyclingEigensolver:
    def __init__(self, *args):
        self.iteration = 0

    def solve(self, *, evals, **kwargs):
        energy = (1.0, 2.0)[self.iteration % 2]
        evals[:4] = energy / 8.0
        self.iteration += 1


def _fake_energy_sequence():
    return iter([1.0, 2.0] * 6)


@pytest.mark.parametrize(
    "module,cls,uses_instance_solver",
    [
        (silicon_example, silicon_example.SiliconSCF, True),
        (graphene_example, graphene_example.GrapheneSCF, True),
        (cnt_example, cnt_example.CarbonNanotubeSCF, False),
    ],
)
def test_model_examples_require_density_convergence_and_return_last_energy(
    monkeypatch, module, cls, uses_instance_solver
):
    driver = cls.__new__(cls)
    driver.n_electrons = 0
    driver.volume = 1.0
    driver.fft_shape = (2,)
    driver.npw = 8
    driver.n_bands = 8
    driver._r_to_g_density = lambda rho: np.asarray(rho)
    driver._g_to_r_density = lambda rho: np.asarray(rho)
    driver._build_potential = lambda: None
    driver._compute_density = lambda: np.ones(2)
    energies = _fake_energy_sequence()
    driver._compute_total_energy = lambda: next(energies)
    driver._print_results = lambda energy: None
    driver.mixer = HoldingMixer()
    driver.hamiltonian = SimpleNamespace(
        apply=lambda psi: psi, get_diagonal=lambda: np.zeros(8))
    if uses_instance_solver:
        driver.eigensolver = FakeEigensolver()
    else:
        monkeypatch.setattr(module, "PCGEigensolver", lambda *args: FakeEigensolver())
    monkeypatch.setattr(
        module, "random_initial_guess", lambda npw, bands: np.zeros((npw, bands)))

    with pytest.warns(RuntimeWarning, match="did not converge"):
        energy = driver.run(max_iter=12, tol=10.0, density_tol=0.01, verbose=False)

    assert energy == 2.0
    assert not driver.converged
    assert driver.iterations == 12
    assert driver.density_residual == pytest.approx(1.0)


def test_bandstructure_scf_oscillation_is_not_success(monkeypatch):
    driver = silicon_bandstructure.SiliconBandStructure.__new__(
        silicon_bandstructure.SiliconBandStructure)
    driver.n_electrons = 1
    driver.volume = 1.0
    driver.fft_shape = (2,)
    driver.npw = driver.n_bands = 8
    driver.gvec = SimpleNamespace(npw=8)
    driver.density_gvec = SimpleNamespace(norms=np.ones(8))
    driver.v_ext = np.zeros(2)
    driver._r_to_g = lambda rho: np.asarray(rho)
    driver._g_to_r = lambda rho: np.asarray(rho)
    driver._build_potential = lambda: None
    driver.hamiltonian = SimpleNamespace(
        apply=lambda psi: psi, get_diagonal=lambda: np.zeros(8),
        set_local_potential=lambda potential: None)
    monkeypatch.setattr(
        silicon_bandstructure, "PCGEigensolver", CyclingEigensolver)
    monkeypatch.setattr(
        silicon_bandstructure, "random_initial_guess",
        lambda npw, bands: np.zeros((npw, bands)))
    monkeypatch.setattr(
        hamiltonian, "g_to_r",
        lambda *args: np.full(2, np.sqrt(0.5)))
    monkeypatch.setattr(
        silicon_bandstructure, "LinearMixer", lambda **kwargs: HoldingMixer())

    with pytest.warns(RuntimeWarning, match="did not converge"):
        evals = driver.run_scf(max_iter=12, tol=10.0, density_tol=0.01)

    assert evals.shape == (8,)
    assert not driver.converged
    assert driver.iterations == 12
    assert driver.density_residual > 0.01


def test_upf_gamma_scf_checks_total_energy_and_density(monkeypatch):
    driver = silicon_upf.SiliconUPF.__new__(silicon_upf.SiliconUPF)
    driver.n_electrons = 2
    driver.volume = 1.0
    driver.fft_shape = (2,)
    driver.npw = driver.n_bands = 8
    driver.gvec = SimpleNamespace(npw=8)
    driver.density_gvec = SimpleNamespace(norms=np.ones(8))
    driver.vloc_g_atomic = np.zeros(8, dtype=complex)
    driver._r_to_g = lambda rho: np.asarray(rho)
    driver._g_to_r = lambda rho: np.asarray(rho)
    driver._build_potential = lambda: None
    driver._print_results = lambda energy, occupations: None
    driver._compute_total_energy = lambda occupations: next(energies)
    driver.hamiltonian = SimpleNamespace(
        apply=lambda psi: psi, get_diagonal=lambda: np.zeros(8),
        set_local_potential=lambda potential: None)
    driver.nlpot = SimpleNamespace(get_full_projectors=lambda k: None)
    driver.mixer = HoldingMixer()
    energies = _fake_energy_sequence()
    monkeypatch.setattr(silicon_upf, "PCGEigensolver", CyclingEigensolver)
    monkeypatch.setattr(
        silicon_upf, "random_initial_guess",
        lambda npw, bands: np.zeros((npw, bands)))
    monkeypatch.setattr(
        silicon_upf, "g_to_r", lambda *args: np.full(2, np.sqrt(0.5)))
    monkeypatch.setattr(
        silicon_upf, "LinearMixer", lambda **kwargs: HoldingMixer())

    with pytest.warns(RuntimeWarning, match="did not converge"):
        energy = driver._run_scf_gamma(max_iter=12, tol=10.0, density_tol=0.01)

    assert energy == 2.0
    assert not driver.converged
    assert driver.iterations == 12
    assert driver.density_residual > 0.01


def test_upf_gamma_scf_uses_total_energy_not_band_sum(monkeypatch):
    driver = silicon_upf.SiliconUPF.__new__(silicon_upf.SiliconUPF)
    driver.n_electrons = 2
    driver.volume = 1.0
    driver.fft_shape = (2,)
    driver.npw = driver.n_bands = 8
    driver.gvec = SimpleNamespace(npw=8)
    driver.density_gvec = SimpleNamespace(norms=np.ones(8))
    driver.vloc_g_atomic = np.zeros(8, dtype=complex)
    driver._r_to_g = lambda rho: np.asarray(rho)
    driver._g_to_r = lambda rho: np.asarray(rho)
    driver._build_potential = lambda: None
    driver._print_results = lambda energy, occupations: None
    energies = iter([0.0, 1.0, 0.0, 1.0, 0.0])
    driver._compute_total_energy = lambda occupations: next(energies)
    driver.hamiltonian = SimpleNamespace(
        apply=lambda psi: psi, get_diagonal=lambda: np.zeros(8),
        set_local_potential=lambda potential: None)
    driver.nlpot = SimpleNamespace(get_full_projectors=lambda k: None)
    monkeypatch.setattr(silicon_upf, "PCGEigensolver", lambda *args: FakeEigensolver())
    monkeypatch.setattr(
        silicon_upf, "random_initial_guess",
        lambda npw, bands: np.zeros((npw, bands)))
    monkeypatch.setattr(silicon_upf, "g_to_r", lambda *args: np.ones(2))

    with pytest.warns(RuntimeWarning, match="did not converge"):
        energy = driver._run_scf_gamma(max_iter=5, tol=0.1, density_tol=1e-7)

    assert energy == 0.0
    assert not driver.converged
    assert driver.iterations == 5
    assert driver.density_residual == pytest.approx(0.0)
