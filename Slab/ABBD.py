import numpy as np


def v_b(K, Sb, eps1, eps2, E_b):
    """Коэффициент упругости бетона."""

    vb = np.ones((K, 2))
    for i in range(0, K):
        if eps1[i] != 0:
            vb[i, 0] = Sb[i][0] / E_b[i][0] / eps1[i]
        else:
            vb[i, 0] = 1.0
        if eps2[i] != 0:
            vb[i, 1] = Sb[i][1] / E_b[i][1] / eps2[i]
        else:
            vb[i, 1] = 1.0
    return vb


def sxyb(K, orientation, Sb):
    c = np.cos(orientation)
    s = np.sin(orientation)
    Sxyb = np.zeros((K, 3, 1))
    for i in range(0, K):
        Cb = np.array([[c[i] ** 2, s[i] ** 2], [s[i] ** 2, c[i] ** 2], [s[i] * c[i], -s[i] * c[i]]])
        Sxyb[i][:, :] = Cb @ Sb[i][:, :]
    return Sxyb


def conc(u, v01, v10, plb, K, vsigmab, e_b, s_b, E_b):
    vv = 1 - v01 * v10
    epsb = (plb @ u).reshape(K, 3, 1)
    exx = epsb[:, 0].reshape(K)
    eyy = epsb[:, 1].reshape(K)
    gxy = epsb[:, 2].reshape(K)
    ee1 = exx + eyy
    ee2 = exx - eyy
    emax = (ee1 / 2 + np.sqrt((ee2 / 2) ** 2 + (gxy / 2) ** 2))
    emin = (ee1 / 2 - np.sqrt((ee2 / 2) ** 2 + (gxy / 2) ** 2))
    eps1 = (emax + v01 * emin) / vv
    eps2 = (v10 * emax + emin) / vv
    orientation = 0.5 * np.arctan2(gxy, ee2)
    kRb = np.ones(K)
    for i in range(0, K):
        if eps1[i] > 0.002 and eps2[i] < 0:
            kRb[i] = 1.0 / (0.8 + 100 * eps1[i])
        else:
            kRb[i] = 1.0
    Sb = np.vstack((vsigmab(eps1, *e_b, *s_b, E_b[:,0], 1), vsigmab(eps2, *e_b, *s_b, E_b[:,0], kRb[i]))).transpose().reshape(K, 2, 1)
    vb = v_b(K, Sb, eps1, eps2, E_b)
    Sxyb = sxyb(K, orientation, Sb)
    return vb, Sb, Sxyb, orientation, eps1, eps2


def cQb(E_b, v01, K):
    Qb = np.zeros((K, 3, 3))
    G01 = np.zeros(K)
    v10 = np.zeros(K)
    for i in range(0, K):
        if E_b[i, 0] != 0.0:
            v10[i] = E_b[i, 1] * v01[i] / E_b[i, 0]
        else:
            v10[i] = 0.0
        G01[i] = E_b[i][0] / (2 * (1 + v10[i]))
    v01 = v10
    vv = 1 - v01 * v10
    Qb[:, 0, 0] = E_b[:, 0] / vv
    Qb[:, 0, 1] = v01 * E_b[:, 1] / vv
    Qb[:, 1, 1] = E_b[:, 1] / vv
    Qb[:, 1, 0] = v10 * E_b[:, 1] / vv
    Qb[:, 2, 2] = G01
    return Qb, v01, v10


def cT(a):
    n = len(a)
    c = np.cos(a)
    s = np.sin(a)
    T = np.zeros((n, 3, 3))
    for i in range(0, n):
        T[i, 0, :] = np.array((c[i] ** 2, s[i] ** 2, (2 * c[i] * s[i])))
        T[i, 1, :] = np.array((s[i] ** 2, c[i] ** 2, (-2 * c[i] * s[i])))
        T[i, 2, :] = np.array(((-c[i] * s[i]), c[i] * s[i], (c[i] ** 2 - s[i] ** 2)))
    return T


def cD(A, Z, T, Q):
    R = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 2]])
    R_ = np.linalg.inv(R)
    T_ = np.linalg.inv(T)
    n = len(A)
    M = T_ @ Q @ R @ T @ R_
    A_ = sum(M * A.reshape(n, 1, 1))
    B_ = sum(M * Z.reshape(n, 1, 1) * A.reshape(n, 1, 1))
    D_ = sum(M * Z.reshape(n, 1, 1) ** 2 * A.reshape(n, 1, 1))
    Di = np.vstack((np.hstack((A_, B_)), np.hstack((B_, D_))))
    return Di


def v_s(ns, strain, stress, E_s):
    """Коэффициент упругости арматуры."""

    vs = np.ones(ns)
    for i in range(0, ns):
        if strain[i] != 0:
            vs[i] = stress[i] / E_s[i] / strain[i]
        else:
            vs[i] = 1.0
    return vs


def sxys(ns, c, s, stress):
    Sxys = np.zeros((ns, 3, 1))
    for i in range(0, ns):
        Cs = np.array([[c[i] ** 2, s[i] ** 2], [s[i] ** 2, c[i] ** 2], [s[i] * c[i], -s[i] * c[i]]])
        STs = np.hstack([stress[i], 0])
        Sxys[i][:, :] = (Cs @ STs).reshape(3, 1)
    return Sxys


def reb(u, ns, pls, alpha, vsigmas, e_s, s_s, E_s):
    epss = (pls @ u).reshape(ns, 3, 1)
    c = np.cos(alpha)
    s = np.sin(alpha)
    strain = np.zeros(ns)
    stress = np.zeros(ns)
    for i in range(0, ns):
        dc = np.array([c[i] ** 2, s[i] ** 2, 2 * s[i] * c[i]])
        strain[i] = dc @ epss[i]
        stress[i] = vsigmas(strain[i], *e_s[i, :, :][0], *s_s[i, :, :][0], E_s[i])
    vs = v_s(ns, strain, stress, E_s)
    Sxys = sxys(ns, c, s, stress)
    return vs, Sxys, strain, stress


def d(E_b, K, vb, E_s, vs, orientation, v01, t, Zb, ns, alpha, As, Zs):
    """Жесткостные характеристики плоских выделенных элементов жб оболочек."""

    Qb, v01, v10 = cQb(E_b * vb, v01, K)
    T = cT(orientation)
    Db = cD(t, Zb, T, Qb)
    Qs = np.zeros((ns, 3, 3))
    Qs[:, 0, 0] = E_s * vs
    T = cT(alpha)
    Ds = cD(As, Zs, T, Qs)
    D = Db + Ds
    return D, v01, v10


if __name__ == '__main__':
    print(d.__doc__)
    print(v_b.__doc__)
    print(v_s.__doc__)
    input('Press Enter:')
