import numpy as np


def d(Ai, Xi, Yi, Ei, vi):
    """Жесткостные характеристики поперечного сечения."""

    d11 = sum(Ai * Ei * vi)
    d22 = sum(Ai * Xi ** 2 * Ei * vi)
    d33 = sum(Ai * Yi ** 2 * Ei * vi)
    d12 = sum(Ai * Xi * Ei * vi)
    d13 = sum(Ai * Yi * Ei * vi)
    d23 = sum(Ai * Xi * Yi * Ei * vi)
    D = np.array([
        [d11, d12, d13],
        [d12, d22, d23],
        [d13, d23, d33]])
    return D


def vi(S, E, e):
    """Коэффициенты упругости."""

    if e != 0.0:
        v = S / E / e
    else:
        v = 1.0
    return v


if __name__ == '__main__':
    print(d.__doc__)
    print(vi.__doc__)
    input('Press Enter:')
