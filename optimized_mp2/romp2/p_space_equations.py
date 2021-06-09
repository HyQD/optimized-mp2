from opt_einsum import contract
import time


def compute_eta(h, u, rho_qp, rho_qspr, o, v, np):
    eta = np.zeros(h.shape, dtype=np.complex128)
    A_ibaj = compute_A_ibaj(rho_qp, o, v, np=np)

    # R_ia = compute_R_ia(h, u, rho_qp, rho_qspr, o, v, np=np)
    R_ia = -compute_R_tilde_ai(h, u, rho_qp, rho_qspr, o, v, np=np).conj().T

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


"""
def compute_R_ia(h, u, rho_qp, rho_qspr, o, v, np):
    R_ia = np.dot(rho_qp[o, o], h[o, v])
    R_ia -= np.dot(h[o, v], rho_qp[v, v])

    R_ia += contract("ispr, pras->ia", rho_qspr[o, :, :, :], u[:, :, v, :])
    R_ia -= contract("irqs, qsar->ia", u[o, :, :, :], rho_qspr[:, :, v, :])

    return R_ia
"""


def compute_R_tilde_ai(h, u, rho_qp, rho_qspr, o, v, np):

    tic = time.time()
    R_tilde_ai = np.dot(rho_qp[v, v], h[v, o])
    R_tilde_ai -= np.dot(h[v, o], rho_qp[o, o])
    toc = time.time()
    print(f"One body contractions: {toc-tic}")

    tic = time.time()
    R_tilde_ai += contract(
        "abkl, klib->ai", rho_qspr[v, v, o, o], u[o, o, o, v]
    )
    R_tilde_ai += 2 * contract("ab, bkik->ai", rho_qp[v, v], u[v, o, o, o])
    R_tilde_ai -= contract("ab, kbik->ai", rho_qp[v, v], u[o, v, o, o])
    toc = time.time()
    print(f"u_ooov contractions: {toc-tic}")

    tic = time.time()
    R_tilde_ai -= contract(
        "bcij, ajbc->ai", rho_qspr[v, v, o, o], u[v, o, v, v]
    )
    toc = time.time()
    print(f"u_vvvo contraction 1: {toc-tic}")

    tic = time.time()
    R_tilde_ai += contract("bc, acbi->ai", rho_qp[v, v], u[v, v, v, o])
    R_tilde_ai -= 2 * contract("bc, acib->ai", rho_qp[v, v], u[v, v, o, v])
    toc = time.time()
    print(f"u_vvvo contractions 2: {toc-tic}")

    tic = time.time()
    R_tilde_ai -= contract(
        "klij, ajkl->ai", rho_qspr[o, o, o, o], u[v, o, o, o]
    )
    toc = time.time()
    print(f"gamma_oooo contraction: {toc-tic}")

    return R_tilde_ai


def compute_R_tilde_ai_MO_driven(f, u, rho_qp, l2, t2, o, v, np):

    tic = time.time()
    tt = 2 * (2 * t2 - t2.transpose(0, 1, 3, 2))
    toc = time.time()
    print(f"Compute tt: {toc-tic}")

    tic = time.time()
    R_tilde_ai = np.dot(rho_qp[v, v], f[v, o])
    R_tilde_ai -= np.dot(f[v, o], rho_qp[o, o])
    toc = time.time()
    print(f"One body contractions: {toc-tic}")

    tic = time.time()
    R_tilde_ai += contract("abkl, klib->ai", tt, u[o, o, o, v])

    ########################################################################
    # These two terms can be combined with rho_qp[v,v]*h[v,o] to form rho_qp[v,v]*f[v,o]
    # R_tilde_ai += 2 * contract("ab, bkik->ai", rho_qp[v, v], u[v, o, o, o])
    # R_tilde_ai -= contract("ab, kbik->ai", rho_qp[v, v], u[o, v, o, o])
    ########################################################################
    toc = time.time()
    print(f"u_ooov contractions: {toc-tic}")

    tic = time.time()
    R_tilde_ai -= contract("bcij, ajbc->ai", tt, u[v, o, v, v])
    toc = time.time()
    print(f"u_vvvo contraction 1: {toc-tic}")

    tic = time.time()
    R_tilde_ai += contract("acbi, bc->ai", u[v, v, v, o], rho_qp[v, v])
    R_tilde_ai -= 2 * contract("acib,bc->ai", u[v, v, o, v], rho_qp[v, v])
    toc = time.time()
    print(f"u_vvvo contractions 2: {toc-tic}")

    tic = time.time()

    # Gamma^{kl}_{ij}*u^{aj}_{kl}
    # These terms combine with h[v,o]*rho_qp[o,o] to form f[v,o]*rho_qp[o,o]
    # rho_oooo += 4 * contract("ik,jl->klij", delta, delta)
    # rho_oooo -= 2 * contract("il,jk->klij", delta, delta)
    # rho_oooo -= contract("kj,li->klij", delta, gamma_ij_corr)
    # rho_oooo += 2 * contract("lj,ki->klij", delta, gamma_ij_corr)

    delta = np.eye(o.stop)
    gamma_ij_corr = rho_qp[o, o] - 2 * delta

    R_tilde_ai -= 2 * contract("lj, ajil->ai", gamma_ij_corr, u[v, o, o, o])
    R_tilde_ai += contract("kj,ajki->ai", gamma_ij_corr, u[v, o, o, o])
    # rho_oooo = np.zeros((o.stop, o.stop, o.stop, o.stop), dtype=t2.dtype)
    # rho_oooo += 2 * contract("ki,lj->klij", delta, gamma_ij_corr)
    # rho_oooo -= contract("li,kj->klij", delta, gamma_ij_corr)
    # R_tilde_ai -= contract("klij, ajkl->ai", rho_oooo, u[v, o, o, o])

    toc = time.time()
    print(f"gamma_oooo contraction: {toc-tic}")

    return R_tilde_ai
