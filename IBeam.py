import pandas as pd
import Frame.IBeamSec as IBS
import Frame.IBeamCalc as IBC

'''Расчет по нелинейной деформационной модели сварного двутавра'''

if __name__ == '__main__':
    data = pd.read_csv('Frame/data/IBeamData.csv', sep=';')
    loads = pd.read_csv('Frame/data/Loads.csv', sep=';')
    H = data['H'].values[0]
    B = data['B'].values[0]
    tf = data['tf'].values[0]
    tw = data['tw'].values[0]
    nfx = data['nfx'].values[0]
    nfy = data['nfy'].values[0]
    nwx = data['nwx'].values[0]
    nwy = data['nwy'].values[0]
    geom = IBS.Geom(H / 1000, B / 1000, tf / 1000, tw / 1000, nfx, nfy, nwx, nwy)
    IBC.ibeam(data, loads, *geom)
