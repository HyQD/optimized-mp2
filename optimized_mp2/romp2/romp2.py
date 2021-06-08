from scipy.linalg import expm

from optimized_mp2.omp2_helper import (
    construct_d_t_1_matrix,
    construct_d_t_2_matrix,
    OACCVector,
)

from optimized_mp2.romp2.rhs_t import (
    compute_t_2_amplitudes,
    compute_l_2_amplitudes,
)

from optimized_mp2.romp2.density_matrices import (
    compute_one_body_density_matrix,
    compute_two_body_density_matrix,
)

from optimized_mp2.omp2_helper import OACCVector, AmplitudeContainer

from optimized_mp2.romp2.p_space_equations import compute_R_tilde_ai

from opt_einsum import contract

import time


class ROMP2:
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

    def __init__(self, system, verbose=False, **kwargs):

        self.np = system.np
        self.system = system
        self.verbose = verbose

        self.n, self.m, self.l = system.n, system.m, system.l

        self.h = self.system.h
        self.u = self.system.u
        self.f = self.system.construct_fock_matrix(self.h, self.u)

        self.o, self.v = self.system.o, self.system.v

        np = self.np
        n, m, l = self.n, self.m, self.l

        self.rhs_t_2 = np.zeros((m, m, n, n), dtype=self.u.dtype)
        self.rhs_l_2 = np.zeros((n, n, m, m), dtype=self.u.dtype)

        self.t_2 = np.zeros_like(self.rhs_t_2)
        self.l_2 = np.zeros_like(self.rhs_l_2)

        self.d_t_2 = construct_d_t_2_matrix(self.f, self.o, self.v, np)
        self.d_l_2 = self.d_t_2.transpose(2, 3, 0, 1).copy()

        self.l_2_mixer = None
        self.t_2_mixer = None

        self.compute_initial_guess()

        self.kappa = np.zeros((l, l), dtype=self.t_2.dtype)

        self.kappa_up = np.zeros((m, n), dtype=self.t_2.dtype)

    def compute_initial_guess(self):
        np = self.np
        o, v = self.o, self.v

        np.copyto(self.rhs_t_2, self.u[v, v, o, o])
        np.divide(self.rhs_t_2, self.d_t_2, out=self.t_2)

        np.copyto(
            self.rhs_l_2,
            compute_l_2_amplitudes(
                self.f, self.u, self.rhs_t_2, self.l_2, self.o, self.v, np
            ),
        )
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

        u = self.u
        o, v = self.o, self.v

        e_tb = contract("klij, ijkl->", rho_qspr[o, o, o, o], u[o, o, o, o])
        e_tb += contract("abij, ijab->", rho_qspr[v, v, o, o], u[o, o, v, v])
        e_tb += contract("ijab, abij->", rho_qspr[o, o, v, v], u[v, v, o, o])

        e_tb += contract("iajb, jbia->", rho_qspr[o, v, o, v], u[o, v, o, v])
        e_tb += contract("aibj, bjai->", rho_qspr[v, o, v, o], u[v, o, v, o])

        e_tb += contract("aijb, jbai->", rho_qspr[v, o, o, v], u[o, v, v, o])
        e_tb += contract("iabj, bjia->", rho_qspr[o, v, v, o], u[v, o, o, v])

        return (
            contract("pq,qp->", self.h, rho_qp)
            + 0.5 * e_tb
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

        for i in range(max_iterations):

            self.f = self.system.construct_fock_matrix(self.h, self.u)

            self.d_t_1 = construct_d_t_1_matrix(self.f, self.o, self.v, np)
            self.d_t_2 = construct_d_t_2_matrix(self.f, self.o, self.v, np)

            tic = time.time()
            rhs_t2 = compute_t_2_amplitudes(
                self.f, self.u, self.t_2, self.o, self.v, np
            )
            toc = time.time()
            print(f"Compute t2: {toc-tic}")

            rhs_l2 = compute_l_2_amplitudes(
                self.f, self.u, rhs_t2, self.l_2, self.o, self.v, np
            )

            self.t_2 += rhs_t2 / self.d_t_2
            self.l_2 += rhs_l2 / self.d_t_2.transpose(2, 3, 0, 1)

            tic = time.time()
            rho_qp = self.compute_one_body_density_matrix()
            rho_qspr = self.compute_two_body_density_matrix()
            toc = time.time()
            print(f"Compute density matrices: {toc-tic}")

            ############################################################
            # This part of the code is common to most (if not all)
            # orbital-optimized methods.
            v, o = self.v, self.o
            tic = time.time()
            w_ai = compute_R_tilde_ai(
                self.h, self.u, rho_qp, rho_qspr, o, v, np
            )
            toc = time.time()
            print(f"Compute kappa derivatives: {toc-tic}")
            residual_w_ai = np.linalg.norm(w_ai)

            self.kappa[self.v, self.o] -= 0.5 * w_ai / self.d_t_1

            C = expm(self.kappa - self.kappa.T)
            Ctilde = C.T

            tic = time.time()
            self.h = self.system.transform_one_body_elements(
                self.system.h, C, Ctilde
            )

            self.u = self.system.transform_two_body_elements(
                self.system.u, C, Ctilde
            )
            toc = time.time()
            print(f"Transform integrals: {toc-tic}")
            ############################################################

            if self.verbose:
                print(f"\nIteration: {i}")
                print(f"Residual norms: |w_ai| = {residual_w_ai}")

            if np.abs(residual_w_ai) < tol:
                break

        tic = time.time()
        energy = self.compute_energy()
        toc = time.time()
        print(f"Compute energy: {toc-tic}")
        print(f"Energy: {energy}")

        self.C = C
        self.C_tilde = C.T.conj()

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

    def compute_one_body_expectation_value(self, mat):

        rho_qp = self.compute_one_body_density_matrix()

        return self.np.trace(self.np.dot(rho_qp, self.C_tilde @ mat @ self.C))
