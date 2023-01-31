import pandas as pd
import Slab.ConSlabSec as Slab
import Slab.ConSlabCalc as SC

'''Расчет по нелинейной деформационной модели железобетонного сечения оболочки'''

if __name__ == '__main__':
    data = pd.read_csv('Slab/data/ConSlabData.csv', sep=';')
    reb = pd.read_csv('Slab/data/RebSlabData.csv', sep=';')
    loads = pd.read_csv('Slab/data/Loads.csv', sep=';')
    H = data['H'].values[0]
    nH = data['nH'].values[0]
    geom = Slab.Geom(H / 1000, nH)
    SC.slab(data, reb, loads, *geom)
