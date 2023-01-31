import numpy as np


def Geom(H, B, nH, nB):
    """Геометрические характеристики прямоугольного сечения. """

    dH = H / nH
    dB = B / nB
    Ab = dH * dB
    nb = nH * nB
    Ab_i = np.linspace(Ab, Ab, nb)
    Secb = np.zeros((nb, 2))
    for i in range(0, nB):
        for j in range(0, nH):
            Secb[j + i * nH, 0] = -H / 2 + dH * j + dH / 2
            Secb[j + i * nH, 1] = -B / 2 + dB * i + dB / 2
    xb1_2 = np.linspace(-H / 2, H / 2, nH + 1)
    yb1_2 = np.linspace(-B / 2, -B / 2, nH + 1)
    xb2_4 = np.linspace(H / 2, H / 2, nB - 2)
    yb2_4 = np.linspace(-B / 2 + dB, B / 2 - dB, nB - 2)
    xb4_3 = np.linspace(H / 2, -H / 2, nH + 1)
    yb4_3 = np.linspace(B / 2, B / 2, nH + 1)
    xb3_1 = np.linspace(-H / 2, -H / 2, nB - 2)
    yb3_1 = np.linspace(B / 2 - dB, -B / 2 + dB, nB - 2)
    Abi = np.hstack([Ab_i, np.zeros(2 * (len(xb1_2) + len(xb2_4)))])
    Xbi = np.hstack([Secb[:, 0], xb1_2, xb2_4, xb4_3, xb3_1])
    Ybi = np.hstack([Secb[:, 1], yb1_2, yb2_4, yb4_3, yb3_1])
    A = np.sum(Abi)
    Ix = np.sum(Ab_i * Secb[:, 1] ** 2)
    Iy = np.sum(Ab_i * Secb[:, 0] ** 2)
    ci = [nb, nb + nH, nb + nH + nB - 1, nb + 2 * nH + nB - 1]
    Zb = np.transpose([np.ones(len(Xbi)), np.array(Xbi), np.array(Ybi)])
    geom = [Xbi, Ybi, Abi, Zb, ci, A, Ix, Iy]
    return geom


if __name__ == '__main__':
    print(Geom.__doc__)
    input('Press Enter:')
