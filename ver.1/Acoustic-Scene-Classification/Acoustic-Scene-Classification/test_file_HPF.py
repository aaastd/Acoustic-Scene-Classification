import scipy
import librosa
import numpy as np
import h5py
from modules import *


for fold_num in range(1, 5):
    meta = read_meta("D:/DCASE 2018 Dataset/Development dataset/DCASE2018-task5-dev/evaluation_setup/fold%d_test.txt"
                    % fold_num)
    # to_hdf5('E:/ETRI/DCASE2018/feature2/', meta, 'fold%d_test' % fold_num)
    to_hdf5_ver2('F:/DCASE2018/fixed variable/', meta, 'fold%d_test' % fold_num)
