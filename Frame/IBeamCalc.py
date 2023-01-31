import numpy as np
import pandas as pd
import prop.SteelDiagram as SD
import Frame.D as fr
import Calc as clc
import Frame.SteelSecCharts as SSch
from prettytable import PrettyTable
import data as dat


def ibeam(data, loads, Xf, Yf, Xw, Yw, Af_i, nf, Aw_j, nw, A, Ix, Iy):
    """Расчет стального сварного двутавра."""
    nfy = data['nfy'].values[0]
    nwx = data['nwx'].values[0]
    N = loads['N'].values[data['Load'][0]]
    Mx = loads['Mx'].values[data['Load'][0]]
    My = loads['My'].values[data['Load'][0]]
    vV = np.vectorize(fr.vi)
    vsigma = np.vectorize(SD.Sigma)
    n_f = nf + 4 * nfy
    n_w = nw + 2 * nwx
    x = np.hstack([Xf, Xw])
    y = np.hstack([Yf, Yw])
    Zf = np.transpose(np.array([np.ones(n_f), Xf, Yf]))
    Zw = np.transpose(np.array([np.ones(n_w), Xw, Yw]))
    dc_f, e_f, s_f, E = SD.Getdiag(data['f'].values[0])
    dc_w, e_w, s_w, E = SD.Getdiag(data['w'].values[0])
    F = np.array([N, Mx, My]) / 1000
    Ef = np.linspace(dc_f['E'], dc_f['E'], n_f)
    Ew = np.linspace(dc_w['E'], dc_w['E'], n_w)
    sigmaf = vsigma(e_f, *e_f, *s_f)
    sigmaw = vsigma(e_w, *e_w, *s_w)
    vf = np.linspace(1, 1, n_f)
    vw = np.linspace(1, 1, n_w)
    D = fr.d(Af_i, Xf, Yf, Ef, vf) + fr.d(Aw_j, Xw, Yw, Ew, vw)
    u = clc.calc(D, F)
    ef = Zf.dot(u)
    ew = Zw.dot(u)
    sf = vsigma(ef, *e_f, *s_f)
    sw = vsigma(ew, *e_w, *s_w)
    acc = 0.0000001
    du = 0.1
    it = 0
    while du >= acc:
        it += 1
        vf = vV(sf, Ef, ef)
        vw = vV(sw, Ew, ew)
        D = fr.d(Af_i, Xf, Yf, Ef, vf) + fr.d(Aw_j, Xw, Yw, Ew, vw)
        u_f = clc.calc(D, F)
        ef = Zf.dot(u_f)
        ew = Zw.dot(u_f)
        sf = vsigma(ef, *e_f, *s_f)
        sw = vsigma(ew, *e_w, *s_w)
        du = np.max(abs(u - u_f))
        u = u_f
    eps = np.hstack([ef, ew])
    Sig = np.hstack([sf, sw])
    Xsec = np.hstack([Xf, Xw])
    Ysec = np.hstack([Yf, Yw])
    p1 = nf
    p2 = nf + nfy - 1
    p3 = nf + nfy
    p4 = nf + 2 * nfy - 1
    p5 = nf + 2 * nfy
    p6 = nf + 3 * nfy - 1
    p7 = nf + 3 * nfy
    p8 = nf + 4 * nfy - 1
    p9 = nf + nw + 4 * nfy
    p10 = nf + nw + 4 * nfy + nwx
    p11 = nf + nw + 4 * nfy + nwx - 1
    p12 = nf + nw + 4 * nfy + 2 * nwx - 1
    ci = [p1, p2, p4, p10, p12, p6, p8, p7, p5, p11, p9, p3, p1]
    pi = [p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, p12]
    S = np.round(Sig, 3)
    Sigma = pd.Series([S[p1], S[p2], S[p3], S[p4], S[p5], S[p6], S[p7], S[p8],
                       S[p9], S[p10], S[p11], S[p12]],
                      index=["Точка 1", "Точка 2", "Точка 3", "Точка 4", "Точка 5", "Точка 6", "Точка 7", "Точка 8",
                             "Точка 9", "Точка 10", "Точка 11", "Точка 12"])
    print(ibeam.__doc__)
    print('Решение получено')
    print('Выполнено', it, 'итераций')
    print('Материал полок:', data['f'].values[0])
    print('Материал стенки:', data['w'].values[0])
    print('Номер нагружения:', data['Load'][0])
    print("A= ", np.round(A * 10000, 3), "см^2")
    print("Ix= ", np.round(Ix * 100000000, 3), "см^4")
    print("Iy= ", np.round(Iy * 100000000, 3), "см^4")
    print('Напряжения в угловых точках [МПа]:')
    print(Sigma)
    maxs = max(S)
    mins = min(S)
    maxxp = np.round(Xsec[S == maxs][0] * 1000, 3)
    maxyp = np.round(Ysec[S == maxs][0] * 1000, 3)
    minxp = np.round(Xsec[S == mins][0] * 1000, 3)
    minyp = np.round(Ysec[S == mins][0] * 1000, 3)
    min_ = min(min(ef), min(ew))
    max_ = max(max(ef), max(ew))
    ep = np.zeros(2)
    ep[0] = np.round(min_, 6)
    ep[1] = np.round(max_, 6)
    print('Min/Max деформации:')
    df = PrettyTable(['Min', 'Max'])
    df.add_row([ep[0], ep[1]])
    print(df)
    print('Проверка:')
    nr = 6
    floads = np.round(1000 * np.array((sum(sf * np.array(Af_i)) + sum(sw * np.array(Aw_j)),
                                       sum(sf * np.array(Af_i) * np.array(Xf)) + sum(
                                           sw * np.array(Aw_j) * np.array(Xw)),
                                       sum(sf * np.array(Af_i) * np.array(Yf)) + sum(
                                           sw * np.array(Aw_j) * np.array(Yw)))), nr)
    lds = PrettyTable(["Нагрузка", "N, кН", "Mx, кН*м", "My, кН*м"])
    lds.add_row(["Заданная", F[0] * 1000, F[1] * 1000, F[2] * 1000])
    lds.add_row(["Полученная", floads[0], floads[1], floads[2]])
    lds.add_row(["u", np.round(u[0], nr), np.round(u[1], nr), np.round(u[2], nr)])
    print(lds)

    SD.Strainstress(e_f, sigmaf, data['f'].values[0])
    SD.Strainstress(e_w, sigmaw, data['w'].values[0])
    arge = [eps, x, y, ci, pi]
    args = [Sig, x, y, ci, pi]
    SSch.strain2D(*arge)
    SSch.strain3D(x, y, eps)
    SSch.stress2D(*args)
    SSch.stress3D(x, y, Sig)


if __name__ == '__main__':
    print(ibeam.__doc__)
    input('Press Enter:')
