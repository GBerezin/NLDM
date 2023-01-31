import numpy as np
import matplotlib.pyplot as plt

plt.style.use('seaborn-whitegrid')


def strain(df):
    """Относительные деформации в слоях железобетонной оболочки."""

    fig = plt.figure(num=strain.__doc__)
    ax = plt.gca()
    df.plot(kind='line', x='Z', y='Strain1', color='green', ax=ax)
    df.plot(kind='line', x='Z', y='Strain2', color='red', ax=ax)
    plt.title('Относительные деформации в слоях бетона')
    ax.set_xlabel('Центры слоев бетона оболочки, м')
    ax.set_ylabel('Относительные деформации в слоях бетона')
    plt.subplots_adjust(left=0.185, right=0.815, bottom=0.1, top=0.85)
    plt.show()


def stress(Z, df, rstress):
    """Напряжения в слоях железобетонной оболочки, МПа."""

    fig = plt.figure(num=stress.__doc__)
    ax = plt.gca()
    df.plot(kind='line', x='Z', y='Stress1', color='green', ax=ax)
    df.plot(kind='line', x='Z', y='Stress2', color='red', ax=ax)
    ax.scatter(Z, np.zeros(len(Z)), s=50, c='green', alpha=0.5)
    for i, txt in enumerate(rstress):
        ax.annotate(round(txt, 2), (Z[i], 0.0), rotation=90, size=10, xytext=(0, 0), va='top',
                    textcoords='offset points')
    plt.title('Напряжения в слоях бетона и арматуры , МПа')
    ax.set_xlabel('Центры слоев плиты и арматуры, м')
    ax.set_ylabel('Напряжения в слоях бетона и арматуры, МПа')
    plt.subplots_adjust(left=0.185, right=0.815, bottom=0.1, top=0.9)
    plt.show()


if __name__ == '__main__':
    print(strain.__doc__)
    print(stress.__doc__)
    input('Press Enter:')
