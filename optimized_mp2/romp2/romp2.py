from scipy.linalg import expm

from optimized_mp2.omp2_helper import (
    construct_d_t_1_matrix,
    construct_d_t_2_matrix,
    OACCVector,
)

from optimized_mp2.romp2.rhs_t import (
    compute_t_2_amplitudes,
    compute_l_2_amplitudes,
    compute_t_2_amplitudes_v2,
)

from optimized_mp2.romp2.density_matrices import (
    compute_one_body_density_matrix,
    compute_two_body_density_matrix,
)

from optimized_mp2.omp2_helper import OACCVector, AmplitudeContainer

from optimized_mp2.romp2.p_space_equations import (
    compute_R_tilde_ai,
)

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
        self.o, self.v = self.system.o, self.system.v

        np = self.np
        n, m, l, o, v = self.n, self.m, self.l, self.o, self.v

        self.h = self.system.h
        self.kappa = np.zeros((l, l), dtype=self.h.dtype)

        self.C = self.C = expm(self.kappa - self.kappa.T)
        self.C_tilde = self.C.T.conj()

        self.u = self.system.u
        self.f = self.system.construct_fock_matrix(self.h, self.u)

        self.rhs_t_2 = np.zeros((m, m, n, n), dtype=self.h.dtype)
        self.rhs_l_2 = np.zeros((n, n, m, m), dtype=self.h.dtype)

        self.t_2 = np.zeros_like(self.rhs_t_2)
        self.l_2 = np.zeros_like(self.rhs_l_2)

        self.d_t_2 = construct_d_t_2_matrix(self.f, self.o, self.v, np)
        self.d_l_2 = self.d_t_2.transpose(2, 3, 0, 1).copy()

        self.l_2_mixer = None
        self.t_2_mixer = None

        self.compute_initial_guess()

    def compute_initial_guess(self):
        np = self.np
        o, v = self.o, self.v

        np.copyto(self.rhs_t_2, self.system.u[v, v, o, o])
        np.divide(self.rhs_t_2, self.d_t_2, out=self.t_2)

        t2_tt = 2 * self.t_2 - self.t_2.transpose(0, 1, 3, 2)
        self.l_2 = 2 * t2_tt.conj().transpose(2, 3, 0, 1)

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

        e_ref = self.compute_reference_energy(self.h, self.u, self.o)
        e_corr = self.compute_Lagrangian(
            self.f, self.u, self.l_2, self.t_2, self.o, self.v
        )

        return e_ref + e_corr

    def compute_reference_energy(self, h, u, o):
        e_ref = (
            2 * self.np.trace(h[o, o])
            + 2 * self.np.trace(self.np.trace(u[o, o, o, o], axis1=1, axis2=3))
            - self.np.trace(self.np.trace(u[o, o, o, o], axis1=1, axis2=2))
            + self.system.nuclear_repulsion_energy
        )

        return e_ref

    def compute_Lagrangian(self, f, u, l2, t2, o, v):

        lag = 2 * contract("abij, ijab->", t2, u[o, o, v, v])
        lag -= contract("abij, ijba->", t2, u[o, o, v, v])
        lag += 0.5 * contract("ijab, abij->", l2, u[v, v, o, o])
        lag += contract("ab, ijac, bcij->", f[v, v], l2, t2)

        # f^i_j * (gamma_corr)^j_i
        lag -= contract("ij, jkab, abik->", f[o, o], l2, t2)

        return lag

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

        v, o = self.v, self.o

        tic = time.time()
        for i in range(max_iterations):

            # self.f = self.transform_f(self.system.h, self.C, self.C_tilde, o, v)

            self.f = self.system.construct_fock_matrix(self.h, self.u)

            self.d_t_1 = construct_d_t_1_matrix(self.f, self.o, self.v, np)
            self.d_t_2 = construct_d_t_2_matrix(self.f, self.o, self.v, np)

            rhs_t2 = compute_t_2_amplitudes(
                self.f, self.u, self.t_2, self.o, self.v, np
            )

            self.t_2 += rhs_t2 / self.d_t_2
            t2_tt = 2 * self.t_2 - self.t_2.transpose(0, 1, 3, 2)
            self.l_2 = 2 * t2_tt.conj().transpose(2, 3, 0, 1)

            rho_qp = self.compute_one_body_density_matrix()

            ############################################################
            # This part of the code is common to most (if not all)
            # orbital-optimized methods.

            w_ai = compute_R_tilde_ai(
                self.f, self.u, rho_qp, self.t_2, o, v, np
            )

            residual_w_ai = np.linalg.norm(w_ai)

            self.kappa[self.v, self.o] -= 0.5 * w_ai / self.d_t_1

            self.C = expm(self.kappa - self.kappa.T)
            self.C_tilde = self.C.T

            self.h = self.system.transform_one_body_elements(
                self.system.h, self.C, self.C_tilde
            )
            self.u = self.system.transform_two_body_elements(
                self.system.u, self.C, self.C_tilde
            )

            energy = self.compute_energy()
            ############################################################

            if self.verbose:
                print(f"\nIteration: {i}")
                print(f"Residual norms: |w_ai| = {residual_w_ai}")

            if np.abs(residual_w_ai) < tol:
                break

        toc = time.time()
        if self.verbose:
            print(f"Time computing tau and C: {toc-tic}")

        if self.verbose:
            print(
                f"Final {self.__class__.__name__} energy: "
                + f"{self.compute_energy()}"
            )

        if change_system_basis:
            if self.verbose:
                print("Changing system basis...")

            self.system.change_basis(C=self.C, C_tilde=self.C_tilde)
            self.C = np.eye(self.system.l)
            self.C_tilde = np.eye(self.system.l)

    def compute_one_body_expectation_value(self, mat):

        rho_qp = self.compute_one_body_density_matrix()

        return self.np.trace(self.np.dot(rho_qp, self.C_tilde @ mat @ self.C))

    def transform_f(self, h, C, C_tilde, o, v):
        f = contract("pa,ab,bq->pq", C_tilde, h, C)

        f += 2 * contract(
            "pA,jB,ABGD,Gq,Dj->pq",
            C_tilde,
            C_tilde[o, :],
            self.system.u,
            C,
            C[:, o],
        )
        f -= contract(
            "pA,jB,ABGD,Gj,Dq->pq",
            C_tilde,
            C_tilde[o, :],
            self.system.u,
            C[:, o],
            C,
        )

        return f
