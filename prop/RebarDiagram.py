import numpy as np
import matplotlib.pyplot as plt

'''Функции для арматуры'''

plt.style.use('seaborn-whitegrid')


def Getdiag(rebgrade, Reb):
    """Характеристики арматуры для диаграммы состояния."""

    ds = Reb[rebgrade]
    s = [ds['Rsc'], ds['Rs']]
    e = [ds['esc2'], ds['esc0'], ds['es0'], ds['es2']]
    Es = ds['E']
    return e, s, Es


def Strainstress(x, y, gr):
    """Диаграмма состояния арматурной стали. """

    fig, ax = plt.subplots(num=Strainstress.__doc__ + gr)
    ax.plot(x, y)
    ax.set_xlabel('Относительные деформации')
    ax.set_ylabel('Напряжения, МПа')
    plt.title(Strainstress.__doc__ + gr)
    ax.scatter(x, y, c='red', alpha=0.5)
    plt.show()


def Sigma(e, esc2, esc0, es0, es2, Rsc, Rs, E):
    """Функция диаграммы состояния арматурной стали."""
    if esc0 >= e:
        S = Rsc
    elif esc0 < e < es0:
        S = E * e
    elif es0 <= e:
        S = Rs
    else:
        S = 0.0
    return S


def diag(rebgrade):
    """Подготовка диаграммы состояния арматурной стали."""
    eps, sig, E = Getdiag(rebgrade)
    vsigma = np.vectorize(Sigma)
    sigma = vsigma(eps, *eps, *sig, E)
    Strainstress(eps, sigma, rebgrade)


if __name__ == '__main__':
    print(Getdiag.__doc__)
    print(Strainstress.__doc__)
    print(Sigma.__doc__)
    print(diag.__doc__)
    input('Press Enter:')
