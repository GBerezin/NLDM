import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from matplotlib.path import Path
from matplotlib.patches import PathPatch
from matplotlib.ticker import LinearLocator, FormatStrFormatter

plt.style.use('seaborn-whitegrid')


def strain2D(z, x, y, ci, pi):
    """Относительные деформации в стальном сечении."""

    xc = x
    yc = y
    zc = z
    x1 = np.linspace(xc.min(), xc.max(), len(xc))
    y1 = np.linspace(yc.min(), yc.max(), len(yc))
    x2, y2 = np.meshgrid(x1, y1)
    z2 = griddata((xc, yc), zc, (x1[None, :], y1[:, None]), method='linear')
    clipindex = ci
    fig, ax = plt.subplots(num=strain2D.__doc__)
    ax.set_aspect('equal')
    ax.set_xlabel('X, м')
    ax.set_ylabel('Y, м')
    plt.title(strain2D.__doc__, pad=20)
    cont = ax.contourf(x2, y2, z2, 50, alpha=0.6, cmap="rainbow")
    clippath = Path(np.c_[x[clipindex], y[clipindex]])
    patch = PathPatch(clippath, facecolor='none', edgecolor='k')
    ax.add_patch(patch)
    for col in cont.collections:
        col.set_clip_path(patch)
    plt.colorbar(cont)
    for i in range(0, 12):
        ax.annotate(i + 1, (x[pi[i]], y[pi[i]]), size=10, xytext=(
            0, 0), ha='left', textcoords='offset points')
    plt.show()


def strain3D(x, y, eps):
    """Относительные деформации в стальном сечении."""

    c = eps
    fig = plt.figure(num=strain3D.__doc__)
    plt.subplots_adjust(left=0.1, right=0.9, bottom=0.1, top=0.9)
    ax = plt.axes(projection='3d')
    ax.set_xlabel('X, м')
    ax.set_ylabel('Y, м')
    ax.set_zlabel(zlabel='Напряжения, МПа')
    plt.title(strain3D.__doc__, pad=10)
    ax.zaxis.set_major_locator(LinearLocator(10))
    ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
    p = ax.scatter3D(x, y, eps, c=c, cmap=plt.cm.rainbow)
    fig.colorbar(p, ax=ax)
    plt.show()


def stress2D(z, x, y, ci, pi):
    """Напряжения в стальном сечении, МПа."""
    xc = x
    yc = y
    zc = z
    x1 = np.linspace(xc.min(), xc.max(), len(xc))
    y1 = np.linspace(yc.min(), yc.max(), len(yc))
    x2, y2 = np.meshgrid(x1, y1)
    z2 = griddata((xc, yc), zc, (x1[None, :], y1[:, None]), method='linear')
    clipindex = ci
    fig, ax = plt.subplots(num=stress2D.__doc__)
    ax.set_aspect('equal')
    ax.set_aspect('equal')
    ax.set_xlabel('X, м')
    ax.set_ylabel('Y, м')
    plt.title(stress2D.__doc__, pad=20)
    cont = ax.contourf(x2, y2, z2, 50, alpha=0.6, cmap="rainbow")
    clippath = Path(np.c_[x[clipindex], y[clipindex]])
    patch = PathPatch(clippath, facecolor='none', edgecolor='k')
    ax.add_patch(patch)
    for col in cont.collections:
        col.set_clip_path(patch)
    plt.colorbar(cont)
    for i in range(0, 12):
        ax.annotate(i + 1, (x[pi[i]], y[pi[i]]), size=10, xytext=(
            0, 0), ha='left', textcoords='offset points')
    plt.show()


def stress3D(x, y, Sig):
    """Напряжения в стальном сечении, МПа."""

    c = Sig
    fig = plt.figure(num=stress3D.__doc__)
    plt.subplots_adjust(left=0.1, right=0.9, bottom=0.1, top=0.9)
    ax = plt.axes(projection='3d')
    ax.set_xlabel('X, м')
    ax.set_ylabel('Y, м')
    ax.set_zlabel(zlabel='Напряжения, МПа')
    plt.title(stress3D.__doc__, pad=10)
    ax.zaxis.set_major_locator(LinearLocator(10))
    ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
    p = ax.scatter3D(x, y, Sig, c=c, cmap=plt.cm.rainbow)
    fig.colorbar(p, ax=ax)
    plt.show()


if __name__ == '__main__':
    print(strain2D.__doc__)
    print(strain3D.__doc__)
    print(stress2D.__doc__)
    print(stress3D.__doc__)
    input('Press Enter:')
