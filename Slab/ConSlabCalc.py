import numpy as np
import pandas as pd
import prop.ConcreteDiagram as CD
import prop.RebarDiagram as RD
import Slab.ABBD as sl
import Calc as clc
import Slab.ConSlabCharts as CSch
import Ccharts as Cch
import data as dat

'''Расчет плоских выделенных элементов жб оболочек на основе нелинейной деформационной модели.'''


def slab(data, reb, loads, t, Zb):
    """Расчет плоских выделенных элементов жб оболочек на основе нелинейной деформационной модели."""

    print(slab.__doc__)
    pl1 = np.eye(3)
    Nxx = loads['Nxx'].values[data['Load'][0]]
    Nyy = loads['Nyy'].values[data['Load'][0]]
    Nxy = loads['Nxy'].values[data['Load'][0]]
    Mxx = loads['Mxx'].values[data['Load'][0]]
    Myy = loads['Myy'].values[data['Load'][0]]
    Mxy = loads['Mxy'].values[data['Load'][0]]
    con = dat.create_connection('prop/materials.db')
    Conc = dat.db(con, 'concrete')
    Fg = np.array([Nxx, Nyy, Nxy, Mxx, Myy, Mxy]) / 1000
    vsigmab = np.vectorize(CD.Sigma)
    K = len(Zb)
    plb = np.zeros((K, 3, 6))
    for i in range(0, K):
        pl2 = pl1 * Zb[i]
        plb[i, :, :] = np.hstack((pl1, pl2))
    v = data['v'].values[0]
    v0_1 = np.ones(K) * v
    e_b, s_b, Eb = CD.Getdiag(data['Concrete'].values[0], data, Conc)
    sigmab = vsigmab(e_b, *e_b, *s_b, Eb)
    Eb_ = np.linspace(Eb, Eb, K)
    E_b = np.stack((Eb_, Eb_), axis=-1)
    vsigmas = np.vectorize(RD.Sigma)
    Reb = dat.db(con, 'rebarsteel')
    con.close()
    Zs = reb['Z'].values
    ds = reb['d'].values
    n_s = reb['n'].values
    alpha_ = reb['alpha'].values
    alpha = np.radians(alpha_)
    ns = len(Zs)
    E_s = np.zeros(ns)
    e_s = np.zeros((ns, 1, 4))
    s_s = np.zeros((ns, 1, 2))
    for i in range(ns):
        e_s[i, :, :], s_s[i, :, :], E_s[i] = RD.Getdiag(reb['Grade'][i], Reb)
    As = np.pi * (ds / 1000) ** 2 / 4 * n_s
    pls = np.zeros((ns, 3, 6))
    for i in range(0, ns):
        pl2 = np.eye(3) * Zs[i]
        pls[i, :, :] = np.hstack((pl1, pl2))
    v01 = v0_1
    args = [v01, K, E_b, plb, E_s, pls, Fg, t, Zb, ns, alpha, As, Zs, vsigmab, vsigmas, e_b, s_b, e_s, s_s]
    eps1, eps2, Sig1, Sig2, Sxyb, Sxys, orientation, strain, stress, eps, Sig, u = itrn(*args)
    result = rslt(orientation, alpha, Zb, K, eps1, eps2, Sig1, Sig2, t, Zs, ns, strain, stress, As)
    cvrg = converg(Fg, Zb, Zs, Sxyb, Sxys, t, As, u)
    print('Номер нагружения:', data['Load'][0])
    print('Результаты расчета:')
    print(result)
    min_ = min(min(eps1), min(eps2), min(strain))
    max_ = max(max(eps1), max(eps2), max(strain))
    ep = np.zeros((1, 2))
    ep[0, 0] = round(min_, 6)
    ep[0, 1] = round(max_, 6)
    res = pd.DataFrame(ep, columns=['Strain_min', 'Strain_max'])
    print(res.head(np.size(res)))
    print('Проверка сходимости:')
    print(cvrg)
    Cch.loads('Slab/slab_element.png')  # Правило знаков нагрузок
    print('Бетон класса:', data['Concrete'].values[0])
    print('Коэффициент работы бетона на растяжение: ', data['kt'][0])
    print('Коэффициент gb3: ', data['gb3'][0])
    print('Коэффициент Пуассона: ', data['v'][0])
    CD.Strainstress(e_b, sigmab, data['Concrete'].values[0])  # Диаграмма состояния бетона
    rbrs = {}  # Словарь арматурных стержней с уникальными классами
    Es = np.zeros(ns)
    for i in range(ns):
        es, ss, Es[i] = RD.Getdiag(reb['Grade'][i], Reb)
        rbrs[reb['Grade'][i]] = [es, ss, Es[i]]
    for key in rbrs.keys():
        sigmas = vsigmas(rbrs[key][0], *rbrs[key][0], *rbrs[key][1], rbrs[key][2])
        print('Арматура класса:', key)
        RD.Strainstress(rbrs[key][0], sigmas, key)  # Диаграмма состояния арматуры
    df = result.iloc[:K, :]
    CSch.strain(df)
    rstress = result['Stress1'].values[K:]
    Z = result['Z'].values[K:]
    CSch.stress(Z, df, rstress)


def itrn(v01, K, E_b, plb, E_s, pls, Fg, t, Zb, ns, alpha, As, Zs, vsigmab, vsigmas, e_b, s_b, e_s, s_s):
    """Итерации."""

    acc = 0.0000001
    orientation = np.zeros(K)
    vb = np.ones((K, 2))
    vs = np.ones(ns)
    D, v01, v10 = sl.d(E_b, K, vb, E_s, vs, orientation, v01, t, Zb, ns, alpha, As, Zs)
    u = clc.calc(D, Fg)
    Sb = np.zeros(K)
    eps1 = np.zeros(K)
    eps2 = np.zeros(K)
    strain = np.zeros(ns)
    stress = np.zeros(ns)
    Sxyb = np.zeros((K, 2))
    Sxys = np.zeros((ns, 2))
    du = 0.1
    it = 0
    while du >= acc:
        it += 1
        vb, Sb, Sxyb, orientation, eps1, eps2 = sl.conc(u, v01, v10, plb, K, vsigmab, e_b, s_b, E_b)
        vs, Sxys, strain, stress = sl.reb(u, ns, pls, alpha, vsigmas, e_s, s_s, E_s)
        D, v01, v10 = sl.d(E_b, K, vb, E_s, vs, orientation, v01, t, Zb, ns, alpha, As, Zs)
        u_f = clc.calc(D, Fg)
        du = np.max(abs(u - u_f))
        u = u_f
    print('Решение получено')
    print('Выполнено', it, 'итераций')
    Sig1 = Sb[:][:, 0]
    Sig2 = Sb[:][:, 1]
    eps = np.append(eps1.reshape(K, 1), strain.reshape(ns, 1))
    Sig = np.append(Sig1.reshape(K, 1), stress.reshape(ns, 1))
    return eps1, eps2, Sig1, Sig2, Sxyb, Sxys, orientation, strain, stress, eps, Sig, u


def rslt(orientation, alpha, Zb, K, eps1, eps2, Sig1, Sig2, t, Zs, ns, strain, stress, As):
    """Результаты расчета."""

    ang_c = np.degrees(orientation)
    ang_s = np.degrees(alpha)
    res1_b = np.hstack((Zb.reshape(K, 1),
                        np.round(eps1.reshape(K, 1), 5),
                        np.round(eps2.reshape(K, 1), 5),
                        np.round(Sig1.reshape(K, 1), 2),
                        np.round(Sig2.reshape(K, 1), 2),
                        np.round(ang_c.reshape(K, 1), 3),
                        t.reshape(K, 1)))
    res1_s = np.hstack((Zs.reshape(ns, 1),
                        np.round(strain.reshape(ns, 1), 5),
                        np.zeros(ns).reshape(ns, 1),
                        np.round(stress.reshape(ns, 1), 2),
                        np.zeros(ns).reshape(ns, 1),
                        np.round(ang_s.reshape(ns, 1), 3),
                        As.reshape(ns, 1)))
    res = np.vstack((res1_b, res1_s))
    result = pd.DataFrame(res, columns=['Z', 'Strain1', 'Strain2', 'Stress1', 'Stress2', 'Angle', 'Area'])
    return result


def converg(Fg, Zb, Zs, Sxyb, Sxys, Ab, As, u):
    """Проверка."""

    Z = np.hstack((Zb, Zs))
    Sxy = np.vstack((Sxyb, Sxys))
    A = np.hstack((Ab, As))
    Ff = np.zeros(6)
    n = len(Z)
    for i in range(0, n):
        Ff[0] = Ff[0] + Sxy[i][0] * A[i]
        Ff[1] = Ff[1] + Sxy[i][1] * A[i]
        Ff[2] = Ff[2] + Sxy[i][2] * A[i]
        Ff[3] = Ff[3] + Sxy[i][0] * A[i] * Z[i]
        Ff[4] = Ff[4] + Sxy[i][1] * A[i] * Z[i]
        Ff[5] = Ff[5] + Sxy[i][2] * A[i] * Z[i]
    cvr = np.hstack((Fg.reshape(6, 1), np.round(Ff.reshape(6, 1), 4), u.reshape(6, 1)))
    cvrg = pd.DataFrame(cvr, index=['Nxx', 'Nyy', 'Nxy', 'Mxx', 'Myy', 'Mxy'], columns=['Дано:', 'Получено:', 'u:'])
    return cvrg


if __name__ == '__main__':
    print(slab.__doc__)
    input('Press Enter:')
