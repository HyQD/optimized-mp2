from opt_einsum import contract
import time

import numpy as np


def compute_eta(f, u, rho_qp, t2, o, v, np):

    eta = np.zeros(f.shape, dtype=np.complex128)
    A_ibaj = compute_A_ibaj(rho_qp, o, v, np=np)

    R_ia = -compute_R_tilde_ai(f, u, rho_qp, t2, o, v, np).conj().T

    A_iajb = A_ibaj.transpose(0, 2, 3, 1)
    eta_jb = -1j * np.linalg.tensorsolve(A_iajb, R_ia)

    eta[o, v] += eta_jb
    eta[v, o] -= eta_jb.conj().T

    return eta


def compute_A_ibaj(rho_qp, o, v, np):
    delta_ij = np.eye(o.stop)
    delta_ba = np.eye(v.stop - o.stop)

    A_ibaj = contract("ba, ij -> ibaj", delta_ba, rho_qp[o, o])
    A_ibaj -= contract("ij, ba -> ibaj", delta_ij, rho_qp[v, v])

    return A_ibaj


def compute_R_tilde_ai(f, u, rho_qp, t2, o, v, np):

    tt = 2 * (2 * t2 - t2.transpose(0, 1, 3, 2))

    R_tilde_ai = np.dot(rho_qp[v, v], f[v, o])
    R_tilde_ai -= np.dot(f[v, o], rho_qp[o, o])

    R_tilde_ai += contract("abkl, klib->ai", tt, u[o, o, o, v])
    R_tilde_ai -= contract("bcij, ajbc->ai", tt, u[v, o, v, v])

    R_tilde_ai += contract("acbi, bc->ai", u[v, v, v, o], rho_qp[v, v])
    R_tilde_ai -= 2 * contract("acib,bc->ai", u[v, v, o, v], rho_qp[v, v])

    delta = np.eye(o.stop)
    gamma_ij_corr = rho_qp[o, o] - 2 * delta

    R_tilde_ai -= 2 * contract("lj, ajil->ai", gamma_ij_corr, u[v, o, o, o])
    R_tilde_ai += contract("kj,ajki->ai", gamma_ij_corr, u[v, o, o, o])

    return R_tilde_ai
