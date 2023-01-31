import numpy as np

'''Решение'''


def calc(D, F):
    """Получение вектора деформаций."""
    try:
        u = np.linalg.solve(D, F)
    except np.linalg.LinAlgError as var1:
        D = np.eye(3)
        Fi = np.array(([0.0, 0.0, 0.0]))
        u = np.linalg.solve(D, Fi)
        print('Решение невозможно:', var1)
        quit()
    return u


if __name__ == '__main__':
    print(calc.__doc__)
    input('Press Enter:')
