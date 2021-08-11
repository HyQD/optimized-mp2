import pytest

from quantum_systems import construct_pyscf_system_rhf

from optimized_mp2.romp2.romp2 import ROMP2

def make_psi4_system(geometry, basis, add_spin=True, anti_symmetrize=True):

    import psi4
    import numpy as np

    from quantum_systems import (
        BasisSet,
        SpatialOrbitalSystem,
        GeneralOrbitalSystem,
        QuantumSystem,
    )

    # Psi4 setup
    psi4.set_memory("2 GB")
    psi4.core.set_output_file("output.dat", False)
    mol = psi4.geometry(geometry)

    # roots per irrep must be set to do the eom calculation with psi4
    psi4.set_options(
        {
            "basis": basis,
            "scf_type": "pk",
            "reference": "rhf",
            "mp2_type": "conv",
            "e_convergence": 1e-8,
            "d_convergence": 1e-8,
            "r_convergence": 1e-8,
        }
    )
    rhf_e, rhf_wfn = psi4.energy("SCF", return_wfn=True)
    ccsd_e, ccsd_wfn = psi4.energy("ccsd", return_wfn=True)
    omp2_e, omp2_wfn = psi4.energy("omp2", return_wfn=True)
    print(f"SCF: {rhf_e}")
    print(f"CCSD: {ccsd_e}")
    print(f"OMP2: {omp2_e}")

    wfn = rhf_wfn
    ndocc = wfn.doccpi()[0]
    n_electrons = 2 * ndocc
    nmo = wfn.nmo()
    C = wfn.Ca()
    npC = np.asarray(C)

    mints = psi4.core.MintsHelper(wfn.basisset())
    H = np.asarray(mints.ao_kinetic()) + np.asarray(mints.ao_potential())
    nmo = H.shape[0]

    # Update H, transform to MO basis
    H = np.einsum("uj,vi,uv", npC, npC, H)
    # Integral generation from Psi4's MintsHelper
    MO = np.asarray(mints.mo_eri(C, C, C, C))
    # Physicist notation
    MO = MO.swapaxes(1, 2)

    dipole_integrals = np.zeros((3, nmo, nmo))
    ints = mints.ao_dipole()
    for n in range(3):
        dipole_integrals[n] = np.einsum(
            "ui,uv,vj->ij", npC, np.asarray(ints[n]), npC, optimize=True
        )

    bs = BasisSet(nmo, dim=3, np=np)
    bs.h = np.complex128(H)
    bs.s = np.complex128(np.eye(nmo))
    bs.u = np.complex128(MO)
    bs.nuclear_repulsion_energy = mol.nuclear_repulsion_energy()
    bs.position = np.complex128(dipole_integrals)
    bs.change_module(np=np)
    system = SpatialOrbitalSystem(n_electrons, bs)

    return (
        (
            omp2_e,
            ccsd_e,
            system.construct_general_orbital_system(
                anti_symmetrize=anti_symmetrize
            ),
        )
        if add_spin
        else (omp2_e, ccsd_e, system)
    )

def test_romp2_groundstate_pyscf():

    geometry = """
    units au
    li 0.0 0.0 0.0
    h  0.0 0.0 3.08
    symmetry c1
    """

    basis = "cc-pvtz"

    omp2_e, ccsd_e, system = make_psi4_system(geometry, basis, add_spin=False, anti_symmetrize=False)

    romp2 = ROMP2(system, verbose=True)
    romp2.compute_ground_state(
        max_iterations=100,
        num_vecs=10,
        tol=1e-10,
        termination_tol=1e-10,
        tol_factor=1e-1,
    )
    
    romp2_e = romp2.compute_energy().real

    print(f"EOMP2-EOMP2_Psi4: {romp2.compute_energy().real-omp2_e}")
    print(f"EOMP2_Psi4: {omp2_e}")
    print(f"EOMP2: {romp2_e}")
    assert(abs(romp2_e-omp2_e) < 1e-8)
    print(f"dip_z OMP2: {romp2.compute_one_body_expectation_value(system.dipole_moment[2]).real}")


if __name__ == "__main__":
    test_romp2_groundstate_pyscf()
