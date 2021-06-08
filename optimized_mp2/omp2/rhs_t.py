from opt_einsum import contract


def compute_t_2_amplitudes(f, u, t, o, v, np, out=None):

    nocc = t.shape[2]
    nvirt = t.shape[0]

    r_T2 = np.zeros((nvirt, nvirt, nocc, nocc), dtype=t.dtype)
    r_T2 += u[v, v, o, o]

    Pij = contract("abik,kj->abij", t, f[o, o])
    r_T2 -= Pij
    r_T2 += Pij.swapaxes(2, 3)

    Pab = contract("ac,cbij->abij", f[v, v], t)
    r_T2 += Pab
    r_T2 -= Pab.swapaxes(0, 1)

    return r_T2


def compute_t_2_amplitudes_MO_driven(f, u, t, C, o, v, np):

    nocc = t.shape[2]
    nvirt = t.shape[0]

    C_tilde = C.conj().T

    tmp_uvij = contract("ui, vj->uvij", C[:, o], C[:, o])
    X_spij = contract("uvij, spuv->spij", tmp_uvij, u)
    tmp_sbij = contract("spij, pb->sbij", X_spij, C[:, v].conj())
    u_vvoo = contract("sa, sbij->abij", C[:, v].conj(), tmp_sbij)

    r_T2 = np.zeros((nvirt, nvirt, nocc, nocc), dtype=t.dtype)
    r_T2 += u_vvoo

    Pij = contract("abik,kj->abij", t, f[o, o])
    r_T2 -= Pij
    r_T2 += Pij.swapaxes(2, 3)

    Pab = contract("ac,cbij->abij", f[v, v], t)
    r_T2 += Pab
    r_T2 -= Pab.swapaxes(0, 1)

    return r_T2


def compute_l_2_amplitudes(f, u, t, l, o, v, np, out=None):
    return t.T.conj()


"""
i t2 = R
-i l2 = L

Riktig: L = R^*
Feil: L = t2^*

Riktig: l2 = t2^*
"""
