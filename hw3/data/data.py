import pandas as pd
import scipy.io as sio


def load_data():
    df = pd.read_csv('data/car.data', header=None,
                     names=['b', 'm', 'd', 'p', 'l', 's', 'e'])
    return df


def load_news():
    return sio.loadmat('data/Train_data.mat'), sio.loadmat('data/Test_Data.mat')
