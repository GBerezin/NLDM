import numpy as np
import matplotlib.pyplot as plt
import data as dat

plt.style.use('seaborn-whitegrid')


def Getdiag(stgrade):
    """Характеристики сталей для расчетной диаграммы работы."""

    con = dat.create_connection('prop/materials.db')
    B9 = dat.db(con, 'b9')
    con.close()
    dc = B9[stgrade]
    eps_ = [-dc['eF'], -dc['eE'], -dc['eD'], -dc['eC'], -dc['eA'], dc['eA'], dc['eC'], dc['eD'], dc['eE'], dc['eF']]
    e = np.array(eps_) * dc['Ry'] / dc['E']
    sig_ = [-dc['SF'], -dc['SE'], -dc['SD'], -dc['SC'], -dc['SA'], dc['SA'], dc['SC'], dc['SD'], dc['SE'], dc['SF']]
    s = np.array(sig_) * dc['Ry']
    Es = dc['E']
    return dc, e, s, Es


def Strainstress(x, y, gr):
    """Диаграмма работы строительной стали. """

    fig, ax = plt.subplots(num=Strainstress.__doc__ + gr)
    ax.plot(x, y)
    ax.set_xlabel('Относительные деформации')
    ax.set_ylabel('Напряжения, МПа')
    plt.title(Strainstress.__doc__ + gr)
    ax.scatter(x, y, c='red', alpha=0.5)
    plt.show()


def Sigma(eps, e0, e1, e2, e3, e4, e5, e6, e7, e8, e9, s0, s1, s2, s3, s4, s5, s6, s7, s8, s9):
    """Функция диаграммы работы строительной стали."""

    if e1 >= eps >= e0:
        S = s0 + (s1 - s0) / (e1 - e0) * (eps - e0)
    elif e2 >= eps > e1:
        S = s1 + (s2 - s1) / (e2 - e1) * (eps - e1)
    elif e3 >= eps > e2:
        S = s2 + (s3 - s2) / (e3 - e2) * (eps - e2)
    elif e4 >= eps > e3:
        S = s3 + (s4 - s3) / (e4 - e3) * (eps - e3)
    elif e5 >= eps > e4:
        S = s4 + (s5 - s4) / (e5 - e4) * (eps - e4)
    elif e6 >= eps > e5:
        S = s5 + (s6 - s5) / (e6 - e5) * (eps - e5)
    elif e7 >= eps > e6:
        S = s6 + (s7 - s6) / (e7 - e6) * (eps - e6)
    elif e8 >= eps > e7:
        S = s7 + (s8 - s7) / (e8 - e7) * (eps - e7)
    elif e9 >= eps > e8:
        S = s8 + (s9 - s8) / (e9 - e8) * (eps - e8)
    else:
        S = 0.0
    return S


def diag(stgrade):
    """Подготовка диаграммы работы строительной стали."""
    dc, eps, sig, E = Getdiag(stgrade)
    vsigma = np.vectorize(Sigma)
    sigma = vsigma(eps, *eps, *sig)
    Strainstress(eps, sigma, stgrade)


if __name__ == '__main__':
    print(Getdiag.__doc__)
    print(Strainstress.__doc__)
    print(Sigma.__doc__)
    print(diag.__doc__)
    input('Press Enter:')
