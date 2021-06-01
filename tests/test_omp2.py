import pytest

from quantum_systems import construct_pyscf_system_rhf

from optimized_mp2.omp2.omp2 import OMP2


@pytest.fixture
def bh_groundstate_omp2():
    return -25.18727358121961


def test_omp2_groundstate_pyscf(bh_groundstate_omp2):

    molecule = "b 0.0 0.0 0.0;h 0.0 0.0 2.4"
    basis = "cc-pvdz"

    system = construct_pyscf_system_rhf(molecule, basis=basis)

    omp2 = OMP2(system, verbose=True)
    omp2.compute_ground_state(
        max_iterations=100,
        num_vecs=10,
        tol=1e-10,
        termination_tol=1e-10,
        tol_factor=1e-1,
    )

    energy_tol = 1e-10

    assert abs((omp2.compute_energy().real) - bh_groundstate_omp2) < energy_tol
