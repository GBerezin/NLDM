import numpy as np
import matplotlib.pyplot as plt

'''Функции для бетона'''


def Getdiag(congrade, data, Conc):
    """Характеристики бетона для диаграммы состояния."""

    dc = Conc[congrade]
    kt = data['kt'][0]
    gb3 = data['gb3'][0]
    Eb = dc['E']
    s = [dc['Rb'] * gb3, dc['Rb'] * 0.6 * gb3, dc['Rbt'] * 0.6 * kt, dc['Rbt'] * kt]
    e = [dc['eb2'], dc['eb0'], s[1] / Eb, s[2] / Eb * kt, dc['ebt0'] * kt, dc['ebt2'] * kt]
    return e, s, Eb


def Strainstress(x, y, gr):
    """Диаграмма состояния бетона. """

    fig, ax = plt.subplots(num=Strainstress.__doc__ + gr)
    ax.plot(x, y)
    ax.set_xlabel('Относительные деформации')
    ax.set_ylabel('Напряжения, МПа')
    plt.title(Strainstress.__doc__ + gr)
    ax.scatter(x, y, c='red', alpha=0.5)
    plt.show()


def Sigma(e, eb2, eb0, eb1, ebt1, ebt0, ebt2, Rb, Sb1, Sbt1, Rbt, E, kRb):
    """Функция диаграммы состояния бетона."""

    Rb = Rb * kRb
    Sb1 = Sb1 * kRb
    if eb0 >= e:
        S = Rb
    elif eb0 < e < eb1:
        S = ((1 - Sb1 / Rb) * (e - eb1) / (eb0 - eb1) + Sb1 / Rb) * Rb
    elif eb1 <= e < 0.0:
        S = E * e
    elif 0.0 < e <= ebt1 and Sbt1 != 0.0:
        S = E * e
    elif ebt1 < e < ebt0 and Rbt != 0.0:
        S = ((1 - Sbt1 / Rbt) * (e - ebt1) / (ebt0 - ebt1) + Sbt1 / Rbt) * Rbt
    elif ebt0 <= e:
        S = Rbt
    else:
        S = 0.0
    return S


def diag(congrade, data):
    """Подготовка диаграммы состояния бетона."""

    dc, eps, sig, E = Getdiag(congrade, data)
    vsigma = np.vectorize(Sigma)
    sigma = vsigma(eps, *eps, *sig, E)
    Strainstress(eps, sigma, congrade)


if __name__ == '__main__':
    print(Getdiag.__doc__)
    print(Strainstress.__doc__)
    print(Sigma.__doc__)
    print(diag.__doc__)
    input('Press Enter:')
