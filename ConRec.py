import pandas as pd
import Frame.ConRecSec as Rec
import Frame.ConRecCalc as RC

'''Расчет по нелинейной деформационной модели прямоугольного железобетонного сечения'''

if __name__ == '__main__':
    data = pd.read_csv('Frame/data/ConRecData.csv', sep=';')
    reb = pd.read_csv('Frame/data/RebRecData.csv', sep=';')
    loads = pd.read_csv('Frame/data/Loads.csv', sep=';')
    H = data['H'].values[0]
    B = data['B'].values[0]
    nH = data['nH'].values[0]
    nB = data['nB'].values[0]
    geom = Rec.Geom(H / 1000, B / 1000, nH, nB)
    RC.rect(data, reb, loads, *geom)
