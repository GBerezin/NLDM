import numpy as np


def Geom(H, nH):
    """Геометрические характеристики выделенного жб элемента оболочки. """

    dH = H / nH
    Abi = dH
    nb = nH
    Abi = np.linspace(Abi, Abi, nb)
    Zbi = np.linspace(H / 2 - dH / 2, -H / 2 + dH / 2, nb)
    t = np.concatenate([[0], Abi, [0]])
    Zb = np.concatenate([[H / 2], Zbi, [-H / 2]])
    geom = [t, Zb]
    return geom


if __name__ == '__main__':
    print(Geom.__doc__)
    input('Press Enter:')
