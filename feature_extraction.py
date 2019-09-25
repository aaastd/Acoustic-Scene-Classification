import librosa
import h5py
import numpy as np
from modules import extract_feature_1ch


def read_meta(filename):
    i = 0
    fp = open(filename, 'r')
    text_str = fp.read().split("\n")
    file = [['' for cols in range(3)] for rows in range(0)]
    for x in text_str:
        tmp = x.split("\t")
        file.append(tmp)
        i += 1
    tmp.pop()
    fp.close()
    return file


def extract_feature(filename, channel):
    y, sr = librosa.load(filename, mono=False, sr=16000)
    mel = librosa.feature.melspectrogram(y[channel-1], sr=sr, n_fft=1024, n_mels=40, hop_length=320, power=1, fmax=16000)
    mel = np.log10(np.maximum(mel, 0.000001)).T
    print(mel.shape)
    return mel


def to_hdf5(path, meta, mode, channel):
    with h5py.File(path+'%s.hdf5' % (mode)) as f:
        f.create_dataset('data', (len(meta)-1, 501, 40), dtype='float32')
        data = f['data']
        for i in range(len(meta)-1):
            # data[i] = extract_feature('D:/DCASE 2018 Dataset/DCASE2018-task5-dev/' + meta[i][0], channel)
            y, sr = librosa.load('E:/DCASE2018/eval/DCASE2018-task5-eval/' + meta[i][0], mono=True, sr=16000)
            data[i] = extract_feature_1ch(y, sr)

# save_path = 'E:/ETRI/DCASE2018/'
save_path = 'F:/DCASE2018/evaluation/'

meta = read_meta('E:/DCASE2018/eval/DCASE2018-task5-eval.meta/DCASE2018-task5-eval/unknown_mic_meta.txt')
to_hdf5(save_path, meta, 'eval_unknown', 1)

# for fold_num in range(1, 5):
#     # meta = read_meta('C:/Users/MIN/PycharmProjects/ETRI/notch_filter/fold%d_train.txt' % fold_num)
#     meta = read_meta('D:/DCASE 2018 Dataset/Development dataset/DCASE2018-task5-dev/evaluation_setup/fold%d_train.txt' % fold_num)
#     for channel in range(1, 2):
#         to_hdf5(save_path, meta, 'fold%d_train' % fold_num, channel)
