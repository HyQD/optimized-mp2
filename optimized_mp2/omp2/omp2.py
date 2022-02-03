from scipy.linalg import expm

from optimized_mp2.omp2_helper import (
    construct_d_t_1_matrix,
    construct_d_t_2_matrix,
    OACCVector,
)

from optimized_mp2.omp2.rhs_t import (
    compute_t_2_amplitudes,
    compute_l_2_amplitudes,
    compute_t_2_amplitudes_MO_driven,
)

# from coupled_cluster.mix import DIIS

from optimized_mp2.omp2.density_matrices import (
    compute_one_body_density_matrix,
    compute_two_body_density_matrix,
)

from optimized_mp2.omp2_helper import OACCVector, AmplitudeContainer

from optimized_mp2.omp2.p_space_equations import (
    compute_R_tilde_ai,
    compute_R_tilde_ai_MO_driven,
)

from opt_einsum import contract

import time


class OMP2:
    """Orbital-optimized second-order Møller-Plesset perturbation theory (OMP2)

    Parameters
    ----------
    system : QuantumSystem
        QuantumSystem class instance description of system

    References
    ----------
    .. [1] U. Bozkaya, J. M. Turney, Y. Yamaguchi, H. F. Schaefer, C. D. Sherrill
          "Quadratically convergent algorithm for orbital optimization in the orbital-optimized coupled-cluster doubles method
          and in orbital-optimized second-order Møller-Plesset perturbation theory", J. Chem. Phys. 135, 104103, 2011.

    """

    def __init__(self, system, verbose=False, MO_driven=False, **kwargs):

        self.np = system.np
        self.system = system
        self.verbose = verbose
        self.MO_driven = MO_driven

        self.n, self.m, self.l = system.n, system.m, system.l
        self.o, self.v = self.system.o, self.system.v

        self.h = self.system.h
        if self.MO_driven == False:
            self.u = self.system.u

        self.f = self.system.construct_fock_matrix(self.system.h, self.system.u)

        o, v = self.o, self.v

        np = self.np
        n, m, l = self.n, self.m, self.l

        self.rhs_t_2 = np.zeros((m, m, n, n), dtype=self.system.u.dtype)
        self.rhs_l_2 = np.zeros((n, n, m, m), dtype=self.system.u.dtype)

        self.t_2 = np.zeros_like(self.rhs_t_2)
        self.l_2 = np.zeros_like(self.rhs_l_2)

        self.d_t_2 = construct_d_t_2_matrix(self.f, self.o, self.v, np)
        self.d_l_2 = self.d_t_2.transpose(2, 3, 0, 1).copy()

        self.l_2_mixer = None
        self.t_2_mixer = None

        # self.compute_initial_guess()

        self.kappa = np.zeros((l, l), dtype=self.t_2.dtype)

        self.kappa_up = np.zeros((m, n), dtype=self.t_2.dtype)

        self.C = expm(self.kappa - self.kappa.T)
        self.C_tilde = self.C.T

    def compute_initial_guess(self):
        np = self.np
        o, v = self.o, self.v

        np.copyto(self.rhs_t_2, self.system.u[v, v, o, o])
        np.divide(self.rhs_t_2, self.d_t_2, out=self.t_2)

        np.copyto(self.rhs_l_2, self.system.u[o, o, v, v])
        np.divide(self.rhs_l_2, self.d_l_2, out=self.l_2)

    def get_amplitudes(self, get_t_0=False):
        """Getter for amplitudes, overwrites CC.get_amplitudes to also include
        coefficients.

        Parameters
        ----------
        get_t_0 : bool
            Returns amplitude at t=0 if True

        Returns
        -------
        OACCVector
            Amplitudes and coefficients in OACCVector object
        """

        if get_t_0:
            amps = AmplitudeContainer(
                t=[
                    self.np.array([0], dtype=self.np.complex128),
                    *self._get_t_copy(),
                ],
                l=self._get_l_copy(),
                np=self.np,
            )

        else:
            amps = AmplitudeContainer(
                t=self._get_t_copy(), l=self._get_l_copy(), np=self.np
            )

        return OACCVector(*amps, C=self.C, C_tilde=self.C_tilde, np=self.np)

    def _get_t_copy(self):
        return [self.t_2.copy()]

    def _get_l_copy(self):
        return [self.l_2.copy()]

    def setup_kappa_mixer(self, **kwargs):
        self.kappa_up_mixer = self.mixer(**kwargs)
        self.kappa_down_mixer = self.mixer(**kwargs)

    def compute_energy(self):

        rho_qp = self.compute_one_body_density_matrix()
        rho_qspr = self.compute_two_body_density_matrix()
        o, v = self.o, self.v
        if self.MO_driven:

            energy = contract("pq,qp->", self.h, rho_qp)

            u_oooo = contract(
                "ip,jq,pqrs,rk,sl->ijkl",
                self.C_tilde[o, :],
                self.C_tilde[o, :],
                self.system.u,
                self.C[:, o],
                self.C[:, o],
            )

            u_oovv = contract(
                "ip,jq,pqrs,ra,sb->ijab",
                self.C_tilde[o, :],
                self.C_tilde[o, :],
                self.system.u,
                self.C[:, v],
                self.C[:, v],
            )

            u_vvoo = contract(
                "ap,bq,pqrs,ri,sj->abij",
                self.C_tilde[v, :],
                self.C_tilde[v, :],
                self.system.u,
                self.C[:, o],
                self.C[:, o],
            )

            u_vovo = contract(
                "ap,iq,pqrs,rb,sj->aibj",
                self.C_tilde[v, :],
                self.C_tilde[o, :],
                self.system.u,
                self.C[:, v],
                self.C[:, o],
            )

            u_ovvo = contract(
                "ip,aq,pqrs,rb,sj->iabj",
                self.C_tilde[o, :],
                self.C_tilde[v, :],
                self.system.u,
                self.C[:, v],
                self.C[:, o],
            )

            u_ovov = contract(
                "ip,aq,pqrs,rj,sb->iajb",
                self.C_tilde[o, :],
                self.C_tilde[v, :],
                self.system.u,
                self.C[:, o],
                self.C[:, v],
            )

            u_voov = contract(
                "ap,iq,pqrs,rj,sb->aijb",
                self.C_tilde[v, :],
                self.C_tilde[o, :],
                self.system.u,
                self.C[:, o],
                self.C[:, v],
            )

            energy_tb = contract("ijkl, klij->", u_oooo, rho_qspr[o, o, o, o])
            energy_tb += contract("ijab, abij->", u_oovv, rho_qspr[v, v, o, o])
            energy_tb += contract("abij, ijab->", u_vvoo, rho_qspr[o, o, v, v])
            energy_tb += contract("aibj, bjai->", u_vovo, rho_qspr[v, o, v, o])
            energy_tb += contract("iabj, bjia->", u_ovvo, rho_qspr[v, o, o, v])
            energy_tb += contract("iajb, jbia->", u_ovov, rho_qspr[o, v, o, v])
            energy_tb += contract("aijb, jbai->", u_voov, rho_qspr[o, v, v, o])

            energy += 0.25 * energy_tb + self.system.nuclear_repulsion_energy

            return energy

        else:
            return (
                contract("pq,qp->", self.h, rho_qp)
                + 0.25 * contract("pqrs,rspq->", self.u, rho_qspr)
                + self.system.nuclear_repulsion_energy
            )

    def compute_t_amplitudes(self):
        np = self.np

        self.rhs_t_2.fill(0)
        self.rhs_t_2 = compute_t_2_amplitudes(
            self.f, self.u, self.t_2, self.o, self.v, out=self.rhs_t_2, np=np
        )

        trial_vector = self.t_2
        direction_vector = np.divide(self.rhs_t_2, self.d_t_2)
        error_vector = self.rhs_t_2.copy()

        self.t_2 = self.t_2_mixer.compute_new_vector(
            trial_vector, direction_vector, error_vector
        )

    def compute_l_amplitudes(self):
        np = self.np

        self.rhs_l_2.fill(0)
        compute_l_2_amplitudes(
            self.f,
            self.u,
            self.t_2,
            self.l_2,
            self.o,
            self.v,
            out=self.rhs_l_2,
            np=np,
        )

        trial_vector = self.l_2
        direction_vector = np.divide(self.rhs_l_2, self.d_l_2)
        error_vector = self.rhs_l_2.copy()

        self.l_2 = self.l_2_mixer.compute_new_vector(
            trial_vector, direction_vector, error_vector
        )

    def compute_one_body_density_matrix(self):
        return compute_one_body_density_matrix(
            self.t_2, self.l_2, self.o, self.v, np=self.np
        )

    def compute_two_body_density_matrix(self):
        return compute_two_body_density_matrix(
            self.t_2, self.l_2, self.o, self.v, np=self.np
        )

    def compute_ground_state(
        self,
        max_iterations=100,
        tol=1e-4,
        termination_tol=1e-4,
        tol_factor=0.1,
        change_system_basis=True,
        **mixer_kwargs,
    ):
        """Compute ground state

        Parameters
        ----------
        max_iterations : int
            Maximum number of iterations
        tol : float
            Tolerance parameter, e.g. 1e-4
        tol_factor : float
            Tolerance factor
        change_system_basis : bool
            Whether or not to change the basis when the ground state is
            reached. Default is ``True``.
        """
        np = self.np

        self.C = expm(self.kappa - self.kappa.T)
        self.C_tilde = self.C.T

        v, o = self.v, self.o

        for i in range(max_iterations):

            tic = time.time()
            self.f = contract(
                "pa,ab,bq->pq", self.C_tilde, self.system.h, self.C
            )
            self.f += contract(
                "pa,jb,abgd,gq,dj",
                self.C_tilde,
                self.C_tilde[o, :],
                self.system.u,
                self.C,
                self.C[:, o],
            )
            toc = time.time()
            print(f"Fock matrix: {toc-tic}")

            self.d_t_1 = construct_d_t_1_matrix(self.f, self.o, self.v, np)
            self.d_t_2 = construct_d_t_2_matrix(self.f, self.o, self.v, np)

            if self.MO_driven:
                self.t_2 += (
                    compute_t_2_amplitudes_MO_driven(
                        self.f,
                        self.system.u,
                        self.t_2,
                        self.C,
                        self.o,
                        self.v,
                        np,
                    )
                    / self.d_t_2
                )

            else:
                self.t_2 += (
                    compute_t_2_amplitudes(
                        self.f, self.u, self.t_2, self.o, self.v, np
                    )
                    / self.d_t_2
                )

            if self.MO_driven == False:
                rho_qp = self.compute_one_body_density_matrix()
                rho_qspr = self.compute_two_body_density_matrix()

            ############################################################
            # This part of the code is common to most (if not all)
            # orbital-optimized methods.
            tic_1 = time.time()
            if self.MO_driven:
                w_ai = compute_R_tilde_ai_MO_driven(
                    self.h, self.system.u, self.t_2, self.C, o, v, np
                )
            else:
                w_ai = compute_R_tilde_ai(
                    self.f, self.u, rho_qp, self.t_2, o, v, np
                )
            toc_1 = time.time()
            print(f"w_ai: {toc_1-tic_1}")

            residual_w_ai = np.linalg.norm(w_ai)

            self.kappa[self.v, self.o] -= w_ai / self.d_t_1

            self.C = expm(self.kappa - self.kappa.T.conj())
            self.C_tilde = self.C.T.conj()

            self.h = self.system.transform_one_body_elements(
                self.system.h, self.C, self.C_tilde
            )

            if self.MO_driven == False:
                tic_2 = time.time()
                self.u = self.system.transform_two_body_elements(
                    self.system.u, self.C, self.C_tilde
                )
                toc_2 = time.time()
                print(f"u_transform: {toc_2-tic_2}")
            ############################################################

            if self.verbose:
                print(f"\nIteration: {i}")
                print(f"Residual norms: |w_ai| = {residual_w_ai}")

            if np.abs(residual_w_ai) < tol:
                break

        energy = self.compute_energy()
        print(f"Energy: {energy}")
        # self.C = C
        # self.C_tilde = C.T.conj()

        if change_system_basis:
            if self.verbose:
                print("Changing system basis...")

            self.system.change_basis(C=self.C, C_tilde=self.C_tilde)
            self.C = np.eye(self.system.l)
            self.C_tilde = np.eye(self.system.l)

        if self.verbose:
            print(
                f"Final {self.__class__.__name__} energy: "
                + f"{self.compute_energy()}"
            )

    def compute_one_body_expectation_value(self, mat, make_hermitian=True):

        rho_qp = self.compute_one_body_density_matrix()

        return self.np.trace(self.np.dot(rho_qp, self.C_tilde @ mat @ self.C))
