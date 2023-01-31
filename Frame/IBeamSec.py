import numpy as np


def Geom(H, B, tf, tw, nfx, nfy, nwx, nwy):
    """Геометрические характеристики стального сварного двутавра."""

    hw = H - 2 * tf
    dtf = tf / nfx
    dB = B / nfy
    dhw = hw / nwx
    dtw = tw / nwy
    Afi = dtf * dB
    Awj = dtw * dhw
    nf = 2 * nfx * nfy
    nw = nwx * nwy
    Af_i = np.linspace(Afi, Afi, nf)
    Aw_j = np.linspace(Awj, Awj, nw)
    Secf1 = np.zeros((nfx * nfy, 2))
    Secf2 = np.zeros((nfx * nfy, 2))
    Secw = np.zeros((nwx * nwy, 2))
    for i in range(0, nfy):
        for j in range(0, nfx):
            Secf1[j + i * nfx, 0] = -H / 2 + dtf * j + dtf / 2
            Secf1[j + i * nfx, 1] = -B / 2 + dB * i + dB / 2
            Secf2[j + i * nfx, 0] = H / 2 - tf + dtf * j + dtf / 2
            Secf2[j + i * nfx, 1] = -B / 2 + dB * i + dB / 2
    xf1_2 = np.linspace(-H / 2, -H / 2, nfy)
    xf3_4 = xf1_2 + tf
    xf5_6 = np.linspace(H / 2 - tf, H / 2 - tf, nfy)
    xf7_8 = xf5_6 + tf
    xw9_11 = np.linspace(-H / 2 + tf, H / 2 - tf, nwx)
    xw10_12 = xw9_11
    yf1_2 = np.linspace(-B / 2, B / 2, nfy)
    yf3_4 = yf1_2
    yf5_6 = yf1_2
    yf7_8 = yf1_2
    yw9_11 = np.linspace(-tw / 2, -tw / 2, nwx)
    yw10_12 = yw9_11 + tw
    Xfc = np.hstack([xf1_2, xf3_4, xf5_6, xf7_8])
    Yfc = np.hstack([yf1_2, yf3_4, yf5_6, yf7_8])
    Xwc = np.hstack([xw9_11, xw10_12])
    Ywc = np.hstack([yw9_11, yw10_12])
    Af_i = np.hstack([Af_i, np.zeros(4 * nfy)])
    Aw_j = np.hstack([Aw_j, np.zeros(2 * nwx)])
    Secf = np.vstack([Secf1, Secf2])
    for i in range(0, nwy):
        for j in range(0, nwx):
            Secw[j + i * nwx, 0] = -hw / 2 + dhw * j + dhw / 2
            Secw[j + i * nwx, 1] = -tw / 2 + dtw * i + dtw / 2
    Xf = np.hstack([Secf[:, 0], Xfc])
    Yf = np.hstack([Secf[:, 1], Yfc])
    Xw = np.hstack([Secw[:, 0], Xwc])
    Yw = np.hstack([Secw[:, 1], Ywc])
    A = np.sum(Af_i) + np.sum(Aw_j)
    Ix = np.sum(Af_i * Yf ** 2) + np.sum(Aw_j * Yw ** 2)
    Iy = np.sum(Af_i * Xf ** 2) + np.sum(Aw_j * Xw ** 2)
    geom = [Xf, Yf, Xw, Yw, Af_i, nf, Aw_j, nw, A, Ix, Iy]
    return geom


if __name__ == '__main__':
    print(Geom.__doc__)
    input('Press Enter:')
