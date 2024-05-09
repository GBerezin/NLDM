import matplotlib.pyplot as plt
import matplotlib.image as mpimg


def loads(img):
    """
    Правило знаков нагрузок.

    :param img: Файл рисунка
    :return:
    """
    fig, ax = plt.subplots(num='Правило знаков нагрузок')
    ax.imshow(mpimg.imread(img))
    ax.axis('off')
    plt.title('Правило знаков нагрузок')
    plt.show()
