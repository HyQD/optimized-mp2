from optimized_mp2.romp2.rhs_t import (
    compute_t_2_amplitudes,
    compute_l_2_amplitudes,
)


from optimized_mp2.romp2.density_matrices import (
    compute_one_body_density_matrix,
    compute_two_body_density_matrix,
)

from optimized_mp2.romp2.p_space_equations import (
    compute_eta,
)

from optimized_mp2.omp2_helper import OACCVector
from optimized_mp2.omp2_helper import AmplitudeContainer

from opt_einsum import contract


class TDROMP2:
    """Time-dependent orbital-optimized second-order Møller-Plesset perturbation theory (TDOMP2)

    Parameters
    ----------
    system : QuantumSystem
        QuantumSystem class instance description of system

    References
    ----------
    .. [1] H. Pathak, T. Sato, K. Ishikawa
          "Time-dependent optimized coupled-cluster method for multielectron dynamics. III.
          A second-order many-body perturbation approximation", J. Chem. Phys. 153, 034110, 2020.

    """

    def __init__(self, system, C=None, C_tilde=None):

        self.np = system.np
        self.truncation = "CCD"
        self.system = system

        # these lines is copy paste from super().__init__, and would be nice to
        # remove.
        # See https://github.com/Schoyen/coupled-cluster/issues/36
        self.h = self.system.h

        self.o = self.system.o
        self.v = self.system.v
        o, v = self.o, self.v

        if C is None:
            C = self.np.eye(system.l)
        if C_tilde is None:
            C_tilde = C.T

        assert C.shape == C_tilde.T.shape

        self.u = self.system.u
        self.f = self.system.construct_fock_matrix(self.h, self.u)

        n_prime = self.system.n
        l_prime = C.shape[1]
        m_prime = l_prime - n_prime

        self.o_prime = slice(0, n_prime)
        self.v_prime = slice(n_prime, l_prime)

        _amp = self.construct_amplitude_template(
            self.truncation, n_prime, m_prime, np=self.np
        )
        self._amp_template = OACCVector(*_amp, C, C_tilde, np=self.np)

        self.last_timestep = None

    @staticmethod
    def construct_amplitude_template(truncation, n, m, np):
        """Constructs an empty AmplitudeContainer with the correct shapes, for
        convertion between arrays and amplitudes."""
        codes = {"S": 1, "D": 2, "T": 3, "Q": 4}
        levels = [codes[c] for c in truncation[2:]]

        # start with t_0
        t = [np.array([0], dtype=np.complex128)]
        l = []

        for lvl in levels:
            shape = lvl * [m] + lvl * [n]
            t.append(np.zeros(shape, dtype=np.complex128))
            l.append(np.zeros(shape[::-1], dtype=np.complex128))
        return AmplitudeContainer(t=t, l=l, np=np)

    def amplitudes_from_array(self, y):
        """Construct AmplitudeContainer from numpy array."""
        return self._amp_template.from_array(y)

    @property
    def amp_template(self):
        """Returns static _amp_template, for setting initial conditions etc"""
        return self._amp_template

    def rhs_t_0_amplitude(self, *args, **kwargs):
        return self.np.array([0 + 0j])

    def rhs_t_amplitudes(self):
        yield compute_t_2_amplitudes

    def rhs_l_amplitudes(self):
        yield compute_l_2_amplitudes

    def compute_left_reference_overlap(self, current_time, y):
        t_0, t_2, l_2, _, _ = self._amp_template.from_array(y).unpack()

        return 1 - 0.25 * self.np.tensordot(
            l_2, t_2, axes=((0, 1, 2, 3), (2, 3, 0, 1))
        )

    def compute_energy(self, current_time, y):

        t_0, t2, l2, C, C_tilde = self._amp_template.from_array(y).unpack()

        self.update_hamiltonian(current_time=current_time, y=y)
        h, f, u = self.h_prime, self.f_prime, self.u_prime
        o, v = self.o, self.v

        e_ref = (
            2 * self.np.trace(h[o, o])
            + 2 * self.np.trace(self.np.trace(u[o, o, o, o], axis1=1, axis2=3))
            - self.np.trace(self.np.trace(u[o, o, o, o], axis1=1, axis2=2))
            + self.system.nuclear_repulsion_energy
        )

        lag = 2 * contract("abij, ijab->", t2, u[o, o, v, v])
        lag -= contract("abij, ijba->", t2, u[o, o, v, v])
        lag += 0.5 * contract("ijab, abij->", l2, u[v, v, o, o])
        lag += contract("ab, ijac, bcij->", f[v, v], l2, t2)
        lag -= contract("ij, jkab, abik->", f[o, o], l2, t2)

        return e_ref + lag

    def one_body_density_matrix(self, t, l):
        t_2 = t[0]
        l_2 = l[0]

        return compute_one_body_density_matrix(
            t_2, l_2, self.o, self.v, np=self.np
        )

    def two_body_density_matrix(self, t, l):
        t_2 = t[0]
        l_2 = l[0]

        # Avoid re-allocating memory for two-body density matrix
        if not hasattr(self, "rho_qspr"):
            self.rho_qspr = None
        else:
            self.rho_qspr.fill(0)

        return compute_two_body_density_matrix(
            t_2, l_2, self.o, self.v, np=self.np, out=self.rho_qspr
        )

    def compute_one_body_density_matrix(self, current_time, y):
        t_0, t_2, l_2, _, _ = self._amp_template.from_array(y).unpack()

        return compute_one_body_density_matrix(
            t_2, l_2, self.o, self.v, np=self.np
        )

    def compute_two_body_density_matrix(self, current_time, y):
        t_0, t_2, l_2, _, _ = self._amp_template.from_array(y).unpack()

        return compute_two_body_density_matrix(
            t_2, l_2, self.o, self.v, np=self.np
        )

    def compute_overlap(self, current_time, y_a, y_b):
        """
        Computes time dependent overlap with respect to a given cc-state
        """
        t0a, t2a, l2a, _, _ = self._amp_template.from_array(y_a).unpack()
        t0b, t2b, l2b, _, _ = self._amp_template.from_array(y_b).unpack()

        return compute_orbital_adaptive_overlap(t2a, l2a, t2b, l2b, np=self.np)

    def compute_p_space_equations(self, t2):
        eta = compute_eta(
            self.f_prime,
            self.u_prime,
            self.rho_qp,
            t2,
            self.o,
            self.v,
            self.np,
        )

        return eta

    def compute_p_space_equations_MO_driven(self, t2, C):
        eta = compute_eta_MO_driven(
            self.f_prime,
            self.system.u,
            self.rho_qp,
            t2,
            C,
            self.o,
            self.v,
            self.np,
        )

        return eta

    def update_hamiltonian(self, current_time, y=None, C=None, C_tilde=None):
        if self.last_timestep == current_time:
            return

        self.last_timestep = current_time

        if y is not None:
            _, _, C, C_tilde = self._amp_template.from_array(y)
        elif C is not None and C_tilde is not None:
            pass
        else:
            raise ValueError(
                "either the amplitude-array or (C and C_tilde) has to be not "
                + "None."
            )

        # Evolve system in time
        if self.system.has_one_body_time_evolution_operator:
            self.h = self.system.h_t(current_time)

        # Change basis to C and C_tilde
        self.h_prime = self.system.transform_one_body_elements(
            self.h, C, C_tilde
        )

        self.u_prime = self.system.transform_two_body_elements(
            self.system.u, C, C_tilde
        )
        self.f_prime = self.system.construct_fock_matrix(
            self.h_prime, self.u_prime
        )

    def __call__(self, current_time, prev_amp):

        np = self.np
        o_prime, v_prime = self.o_prime, self.v_prime

        prev_amp = self._amp_template.from_array(prev_amp)
        t_old, l_old, C, C_tilde = prev_amp

        self.update_hamiltonian(current_time, C=C, C_tilde=C_tilde)

        # Remove t_0 phase as this is not used in any of the equations
        t_old = t_old[1:]

        # OATDCC procedure:
        # Do amplitude step

        t_new = [
            -1j
            * rhs_t_func(
                self.f_prime, self.u_prime, *t_old, o_prime, v_prime, np=self.np
            )
            for rhs_t_func in self.rhs_t_amplitudes()
        ]

        # Compute derivative of phase
        t_0_new = self.np.array([0 + 0j])

        t_new = [t_0_new, *t_new]

        t2_tt = 2 * t_new[1] - t_new[1].transpose(0, 1, 3, 2)
        l_new = [2 * t2_tt.conj().transpose(2, 3, 0, 1)]

        # Compute density matrices
        self.rho_qp = self.one_body_density_matrix(t_old, l_old)

        # Solve P-space equations for eta

        eta = self.compute_p_space_equations(*t_old)

        C_new = np.dot(C, eta)
        C_tilde_new = -np.dot(eta, C_tilde)

        self.last_timestep = current_time

        # Return amplitudes and C and C_tilde
        return OACCVector(
            t=t_new, l=l_new, C=C_new, C_tilde=C_tilde_new, np=self.np
        ).asarray()

    def compute_one_body_expectation_value(self, current_time, y, mat):

        rho_qp = self.compute_one_body_density_matrix(current_time, y)
        t, l, C, C_tilde = self._amp_template.from_array(y)

        return self.np.trace(self.np.dot(rho_qp, C_tilde @ mat @ C))
