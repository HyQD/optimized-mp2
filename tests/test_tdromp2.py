import os
import pytest

import numpy as np
from quantum_systems import construct_pyscf_system_rhf
from quantum_systems.time_evolution_operators import DipoleFieldInteraction

from optimized_mp2.romp2 import ROMP2, TDROMP2

from gauss_integrator import GaussIntegrator
from scipy.integrate import complex_ode

import tqdm


class LaserPulse:
    def __init__(self, t0=0, td=5, omega=0.1, E=0.03):
        self.t0 = t0
        self.td = td
        self.omega = omega
        self.E = E  # Field strength

    def __call__(self, t):
        T = self.td
        delta_t = t - self.t0
        return (
            -(np.sin(np.pi * delta_t / T) ** 2)
            * np.heaviside(delta_t, 1.0)
            * np.heaviside(T - delta_t, 1.0)
            * np.cos(self.omega * delta_t)
            * self.E
        )


omega = 0.057
E = 0.01
laser_duration = 5

system = construct_pyscf_system_rhf(
    molecule="li 0.0 0.0 0.0; h 0.0 0.0 3.08",
    basis="cc-pvdz",
    add_spin=False,
    anti_symmetrize=False,
)

omp2 = ROMP2(system, verbose=False)
omp2.compute_ground_state(
    max_iterations=100,
    num_vecs=10,
    tol=1e-10,
    termination_tol=1e-10,
    tol_factor=1e-1,
)
print(f"EROMP2: {omp2.compute_energy().real}")

tdomp2 = TDROMP2(system)

r = complex_ode(tdomp2).set_integrator("GaussIntegrator", s=3, eps=1e-6)
r.set_initial_value(omp2.get_amplitudes(get_t_0=True).asarray())

polarization = np.zeros(3)
polarization[2] = 1
system.set_time_evolution_operator(
    DipoleFieldInteraction(
        LaserPulse(td=laser_duration, omega=omega, E=E),
        polarization_vector=polarization,
    )
)

dt = 1e-1
T = 10
num_steps = int(T // dt) + 1
t_stop_laser = int(laser_duration // dt) + 1

time_points = np.linspace(0, T, num_steps)

td_energies = np.zeros(len(time_points), dtype=np.complex128)
dip_z = np.zeros(len(time_points), dtype=np.complex128)

td_energies[0] = tdomp2.compute_energy(r.t, r.y)
dip_z[0] = tdomp2.compute_one_body_expectation_value(
    r.t, r.y, system.dipole_moment[2]
)
print(dip_z[0].real)


for i in tqdm.tqdm(range(num_steps - 1)):

    r.integrate(r.t + dt)

    td_energies[i + 1] = tdomp2.compute_energy(r.t, r.y)
    dip_z[i + 1] = tdomp2.compute_one_body_expectation_value(
        r.t, r.y, system.dipole_moment[2]
    )

td_energies_omp2 = np.load("dat/tdomp2_energy.npy")
dip_z_omp2 = np.load("dat/tdomp2_dip_z.npy")

# np.save("tdromp2_energy", td_energies)
# np.save("tdromp2_dip_z", dip_z)

from matplotlib import pyplot as plt

plt.figure()
plt.plot(time_points, td_energies.real)
plt.plot(time_points, td_energies_omp2.real)

plt.figure()
plt.plot(time_points, dip_z.real)
plt.plot(time_points, dip_z_omp2.real)

plt.show()
