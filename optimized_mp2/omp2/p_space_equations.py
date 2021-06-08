from opt_einsum import contract
import time


def compute_eta(h, u, rho_qp, rho_qspr, o, v, np):
    eta = np.zeros(h.shape, dtype=np.complex128)
    A_ibaj = compute_A_ibaj(rho_qp, o, v, np=np)
    R_ia = compute_R_ia(h, u, rho_qp, rho_qspr, o, v, np=np)

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


def compute_R_ia(h, u, rho_qp, rho_qspr, o, v, np):
    R_ia = np.dot(rho_qp[o, o], h[o, v])
    R_ia -= np.dot(h[o, v], rho_qp[v, v])
    R_ia += 0.5 * np.tensordot(
        # rho^{is}_{pr}
        rho_qspr[o, :, :, :],
        # u^{pr}_{as}
        u[:, :, v, :],
        # axes=((s, p, r), (s, p, r))
        axes=((1, 2, 3), (3, 0, 1)),
    )
    R_ia -= 0.5 * np.tensordot(
        # u^{ir}_{qs}
        u[o, :, :, :],
        # rho^{qs}_{ar}
        rho_qspr[:, :, v, :],
        # axes=((r, q, s), (r, q, s))
        axes=((1, 2, 3), (3, 0, 1)),
    )

    return R_ia


def compute_R_tilde_ai(h, u, rho_qp, rho_qspr, o, v, np):
    R_tilde_ai = np.dot(rho_qp[v, v], h[v, o])
    R_tilde_ai -= np.dot(h[v, o], rho_qp[o, o])
    R_tilde_ai += 0.5 * np.einsum(
        "pqir, arpq->ai", u[:, :, o, :], rho_qspr[v, :, :, :], optimize=True
    )
    R_tilde_ai -= 0.5 * np.einsum(
        "aqrs, rsiq->ai", u[v, :, :, :], rho_qspr[:, :, o, :], optimize=True
    )

    return R_tilde_ai


def compute_R_tilde_ai_MO_driven(h, u, t, C, o, v, np):

    # Build one particle density matrix
    rho_vv = (1 / 2) * contract("ijac,bcij -> ba", t.T.conj(), t)

    rho_oo = np.identity(o.stop) - (1 / 2) * contract(
        "jkab,abik -> ji", t.T.conj(), t
    )

    C_tilde = C.conj().T

    R_tilde_ai = np.dot(rho_vv, h[v, o])
    R_tilde_ai -= np.dot(h[v, o], rho_oo)

    tic = time.time()
    u_kjrs = contract("kp, pqrs, jq->kjrs", C_tilde[o, :], u, C_tilde[o, :])
    u_kjib = contract("ri, kjrs, sb->kjib", C[:, o], u_kjrs, C[:, v])
    tmp_ai = contract("abkj, kjib->ai", t, u_kjib)
    toc = time.time()
    print(f"u_kjib: {toc-tic}")

    tic = time.time()
    tmp_BD = contract("jB, Dj->BD", C_tilde[o, :], C[:, o])
    tmp_AG = contract("ABGD, BD->AG", u, tmp_BD)
    tmp_Ai = contract("AG, Gi->Ai", tmp_AG, C[:, o])
    tmp_aA = contract("ab,bA->aA", rho_vv, C_tilde[v, :])
    tmp_ai += contract("aA,Ai", tmp_aA, tmp_Ai)
    toc = time.time()
    print(f"u_bkij: {toc-tic}")

    tic = time.time()
    tmp_DA = contract("jA, Dj->DA", C_tilde[o, :], C[:, o])
    tmp_BG = contract("ABGD, DA->BG", u, tmp_DA)
    tmp_aB = contract("ab, bB->aB", -rho_vv, C_tilde[v, :])
    tmp_Bi = contract("BG,Gi->Bi", tmp_BG, C[:, o])
    tmp_ai += contract("aB,Bi->ai", tmp_aB, tmp_Bi)
    # tmp_ai -= contract("ab, jbij->ai", rho_qp[v, v], u_kbij)
    toc = time.time()
    print(f"u_kbij: {toc-tic}")

    R_tilde_ai += 0.5 * tmp_ai

    ###########################################################################################
    # tmp_ai = contract("bcij,ajbc->ai", rho_qspr[v, v, o, o], u_vovv)
    # tmp_ai += contract("bjic, acbj->ai", rho_qspr[v, o, o, v], u_vvvo)
    # tmp_ai += contract("jbic, acjb->ai", rho_qspr[o, v, o, v], u_vvov)

    # rho^{bc}_{ij}*u^{aj}_{bc}
    tic = time.time()
    rho_rsij = contract("rb, bcij, sc->rsij", C[:, v], t, C[:, v])
    u2_pqij = contract("rsij,pqrs->pqij", rho_rsij, u)
    tmp2_ai = contract(
        "ap, pqij, jq->ai", C_tilde[v, :], u2_pqij, C_tilde[o, :]
    )
    toc = time.time()
    print(f"rho_bcij*u_ajbc: {toc-tic}")

    # rho^{bj}_{ic}*u^{ac}_{bj}
    tic = time.time()
    tmp_BG = contract("bB, cb, Gc->BG", C_tilde[v, :], -rho_vv, C[:, v])
    tmp_AD = contract("ABGD, BG->AD", u, tmp_BG)
    tmp2_ai += contract("aA, AD, Di->ai", C_tilde[v, :], tmp_AD, C[:, o])
    toc = time.time()
    print(f"rho_bjic*u_acbj: {toc-tic}")

    # rho^{jb}_{ic}*u^{ac}_{jb}
    tic = time.time()
    tmp_BD = contract("bB, cb, Dc->BD", C_tilde[v, :], rho_vv, C[:, v])
    tmp_AG = contract("ABGD, BD->AG", u, tmp_BD)
    tmp2_ai += contract("aA, AG, Gi->ai", C_tilde[v, :], tmp_AG, C[:, o])
    toc = time.time()
    print(f"rho_jbic*u_acjb: {toc-tic}")
    # print(f"contract 3 v 1 o: {toc-tic}")
    ###########################################################################################

    tic = time.time()
    rho_oooo = np.zeros((o.stop, o.stop, o.stop, o.stop), dtype=t.dtype)
    delta = np.eye(o.stop)

    term = contract("ki, lj -> klij", delta, delta)
    term -= term.swapaxes(2, 3)
    rho_oooo += term

    term_lj = -0.5 * np.tensordot(t.T.conj(), t, axes=((1, 2, 3), (3, 0, 1)))
    term = contract("ki, lj -> klij", delta, term_lj)
    term -= term.swapaxes(0, 1)
    term -= term.swapaxes(2, 3)
    rho_oooo += term
    tmp2_ai += contract("klij, lkja->ai", rho_oooo, u_kjib.conj())

    # tmp_ai = contract('ki, aA, jB, ABGD, Gk, Dj->ai', rho_qp[o,o], C_tilde[v,:], C_tilde[o,:], u, C[:,o], C[:,o])

    toc = time.time()
    print(f"rho_oooo: {toc-tic}")

    R_tilde_ai -= 0.5 * tmp2_ai

    return R_tilde_ai
