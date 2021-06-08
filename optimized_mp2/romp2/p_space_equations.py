from opt_einsum import contract


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
    R_tilde_ai = np.dot(rho_qp[v, v], h[v, o])
    R_tilde_ai -= np.dot(h[v, o], rho_qp[o, o])

    R_tilde_ai += contract(
        "abkl, klib->ai", rho_qspr[v, v, o, o], u[o, o, o, v]
    )
    R_tilde_ai += 2 * contract("ab, bkik->ai", rho_qp[v, v], u[v, o, o, o])
    R_tilde_ai -= contract("ab, kbik->ai", rho_qp[v, v], u[o, v, o, o])

    R_tilde_ai -= contract(
        "klij, ajkl->ai", rho_qspr[o, o, o, o], u[v, o, o, o]
    )

    R_tilde_ai -= contract(
        "bcij, ajbc->ai", rho_qspr[v, v, o, o], u[v, o, v, v]
    )

    R_tilde_ai += contract("bc, acbi->ai", rho_qp[v, v], u[v, v, v, o])

    R_tilde_ai -= 2 * contract("bc, acib->ai", rho_qp[v, v], u[v, v, o, v])

    return R_tilde_ai
