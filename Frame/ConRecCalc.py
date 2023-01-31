import numpy as np
import prop.ConcreteDiagram as CD
import prop.RebarDiagram as RD
import Frame.D as fr
import Calc as clc
import Frame.ConRecCharts as CRch
from prettytable import PrettyTable
import Ccharts as Cch
import data as dat

'''Расчет по нелинейной деформационной модели железобетонного сечения'''


def rect(data, reb, loads, Xbi, Ybi, Abi, Zb, ci, A, Ix, Iy):
    """Расчет железобетонного сечения."""
    con = dat.create_connection('prop/materials.db')
    N = loads['N'].values[data['Load'][0]]
    Mx = loads['Mx'].values[data['Load'][0]]
    My = loads['My'].values[data['Load'][0]]
    vV = np.vectorize(fr.vi)
    vsigmab = np.vectorize(CD.Sigma)
    Conc = dat.db(con, 'concrete')
    e_b, s_b, Eb = CD.Getdiag(data['Concrete'].values[0], data, Conc)
    sigmab = vsigmab(e_b, *e_b, *s_b, Eb)
    vsigmas = np.vectorize(RD.Sigma)
    Reb = dat.db(con, 'rebarsteel')
    con.close()
    Xsj = np.array(reb['X'].values / 1000)
    Ysj = np.array(reb['Y'].values / 1000)
    n_b = len(Xbi)
    ns = len(Xsj)
    Zs = np.transpose(np.array([np.ones(ns), Xsj, Ysj]))
    dsj = reb['d'].values
    Asj = np.pi * (dsj / 1000) ** 2 / 4
    F = np.array([N, Mx, My]) / 1000
    Ebi = np.linspace(Eb, Eb, n_b)
    vbi = np.linspace(1, 1, n_b)
    vsj = np.linspace(1, 1, ns)
    Esj = np.zeros(ns)
    e_s = np.zeros((ns, 1, 4))
    s_s = np.zeros((ns, 1, 2))
    for i in range(ns):
        e_s[i, :, :], s_s[i, :, :], Esj[i] = RD.Getdiag(reb['Grade'][i], Reb)
    D = fr.d(Abi, Xbi, Ybi, Ebi, vbi) + fr.d(Asj, Xsj, Ysj, Esj, vsj)
    u = clc.calc(D, F)
    eb = Zb.dot(u)
    sb = vsigmab(eb, *e_b, *s_b, Ebi)
    es = Zs.dot(u)
    ss = np.zeros(ns)
    for i in range(ns):
        ss[i] = RD.Sigma(es[i], *e_s[i, :, :][0], *s_s[i, :, :][0], Esj[i])
    acc = 0.0000001
    du = 0.1
    it = 0
    while du >= acc:
        it += 1
        vb = vV(sb, Ebi, eb)
        vs = vV(ss, Esj, es)
        D = fr.d(Abi, Xbi, Ybi, Ebi, vb) + fr.d(Asj, Xsj, Ysj, Esj, vs)
        u_f = clc.calc(D, F)
        eb = Zb.dot(u_f)
        es = Zs.dot(u_f)
        sb = vsigmab(eb, *e_b, *s_b, Ebi)
        for i in range(ns):
            ss[i] = RD.Sigma(es[i], *e_s[i, :, :][0], *s_s[i, :, :][0], Esj[i])
        du = np.max(abs(u - u_f))
        u = u_f
    print(rect.__doc__)
    print('Решение получено')
    print('Выполнено', it, 'итераций')
    print('Бетон класса:', data['Concrete'].values[0])
    print('Номер нагружения:', data['Load'][0])
    print("Геометрические характеристики сечения:")
    print("A= ", np.round(A * 10000, 3), "см^2")
    print("Ix= ", np.round(Ix * 100000000, 3), "см^4")
    print("Iy= ", np.round(Iy * 100000000, 3), "см^4")
    print('Коэффициент работы бетона на растяжение: ', data['kt'][0])
    print('Коэффициент gb3: ', data['gb3'][0])
    ep = np.zeros(2)
    ep[0] = round(min(eb), 6)
    ep[1] = round(max(es), 6)
    print('Min/Max деформации:')
    df = PrettyTable(['Бетон', 'Арматура'])
    df.add_row([ep[0], ep[1]])
    print(df)
    print('Проверка:')
    nr = 6
    cloads = np.round(1000 * np.array((sum(sb * np.array(Abi)) + sum(ss * np.array(Asj)),
                                       sum(sb * np.array(Abi) * np.array(Xbi)) + sum(
                                           ss * np.array(Asj) * np.array(Xsj)),
                                       sum(sb * np.array(Abi) * np.array(Ybi)) + sum(
                                           ss * np.array(Asj) * np.array(Ysj)))), nr)
    ptl = PrettyTable(["Нагрузка", "N, кН", "Mx, кН*м", "My, кН*м"])
    ptl.add_row(["Заданная", F[0] * 1000, F[1] * 1000, F[2] * 1000])
    ptl.add_row(["Полученная", cloads[0], cloads[1], cloads[2]])
    ptl.add_row(["u", np.round(u[0], nr), np.round(u[1], nr), np.round(u[2], nr)])
    print(ptl)
    print('Арматурные стержни:')
    ptr = PrettyTable(["Имя", "X, мм", "Y, мм", "Диаметр, мм", "Класс", "Деформации", "Напряжения, МПа"])
    for i in range(ns):
        ptr.add_row([i, Xsj[i] * 1000, Ysj[i] * 1000, dsj[i], reb['Grade'].values[i], np.round(es[i], nr),
                     np.round(ss[i], nr)])
    print(ptr)
    Cch.loads('Frame/frame_section.png')  # Правило знаков нагрузок
    CD.Strainstress(e_b, sigmab, data['Concrete'].values[0])
    rbrs = {}  # Словарь арматурных стержней с уникальными классами
    for i in range(ns):
        e_s, s_s, Esj[i] = RD.Getdiag(reb['Grade'][i], Reb)
        rbrs[reb['Grade'][i]] = [e_s, s_s, Esj[i]]
    for key in rbrs.keys():
        sigmas = vsigmas(rbrs[key][0], *rbrs[key][0], *rbrs[key][1], rbrs[key][2])
        print('Арматура класса:', key)
        RD.Strainstress(rbrs[key][0], sigmas, key)  # Диаграмма состояния арматуры
    arge = [eb, es, Xbi, Ybi, Xsj, Ysj, Asj, ci]
    args = [sb, ss, Xbi, Ybi, Xsj, Ysj, Asj, ci]
    CRch.strain2D(*arge)  # График деформаций в сечении 2D
    CRch.strain3D(eb, Xbi, Ybi)  # График деформаций в сечении 3D
    CRch.stress2D(*args)  # График напряжений в сечении 2D
    CRch.stress3D(sb, Xbi, Ybi)  # График напряжений в сечении 3D


if __name__ == '__main__':
    print(rect.__doc__)
    input('Press Enter:')
