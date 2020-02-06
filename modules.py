import librosa
from librosa import display
from scipy import signal
import h5py
import numpy as np
import random
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import *
from keras.utils import multi_gpu_model
from keras import optimizers


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


def extract_feature(y, sr):
    mel1 = librosa.feature.melspectrogram(y[0], sr=sr, n_fft=1024, n_mels=40, hop_length=320, power=1,
                                          fmax=16000)
    mel1 = np.log10(np.maximum(mel1, 0.000001)).T
    mel2 = librosa.feature.melspectrogram(y[1], sr=sr, n_fft=1024, n_mels=40, hop_length=320, power=1,
                                          fmax=16000)
    mel2 = np.log10(np.maximum(mel2, 0.000001)).T
    mel3 = librosa.feature.melspectrogram(y[2], sr=sr, n_fft=1024, n_mels=40, hop_length=320, power=1,
                                          fmax=16000)
    mel3 = np.log10(np.maximum(mel3, 0.000001)).T
    mel4 = librosa.feature.melspectrogram(y[3], sr=sr, n_fft=1024, n_mels=40, hop_length=320, power=1,
                                          fmax=16000)
    mel4 = np.log10(np.maximum(mel4, 0.000001)).T
    return mel1, mel2, mel3, mel4


def extract_feature_1ch(y, sr):
    mel = librosa.feature.melspectrogram(y, sr=sr, n_fft=1024, n_mels=40, hop_length=320, power=1, fmax=16000)
    mel = np.log10(np.maximum(mel, 0.000001)).T
    return mel


def plot_figure(mat, title):
    plt.figure(figsize=(10, 4))
    display.specshow(librosa.amplitude_to_db(mat, ref=np.max), y_axis='linear', x_axis='time')
    # display.waveplot(mat, sr=44100)
    plt.title(title)
    plt.tight_layout()
    plt.show()


def to_hdf5(path, meta, mode):
    with h5py.File(path+'%s_ch%d.hdf5' % (mode, 1)) as f1, \
            h5py.File(path+'%s_ch%d.hdf5' % (mode, 2)) as f2, \
            h5py.File(path+'%s_ch%d.hdf5' % (mode, 3)) as f3, \
            h5py.File(path+'%s_ch%d.hdf5' % (mode, 4)) as f4:
        f1.create_dataset('data1', (len(meta)-1, 501, 40), dtype='float32')
        f2.create_dataset('data2', (len(meta)-1, 501, 40), dtype='float32')
        f3.create_dataset('data3', (len(meta)-1, 501, 40), dtype='float32')
        f4.create_dataset('data4', (len(meta)-1, 501, 40), dtype='float32')

        data1, data2, data3, data4 = f1['data1'], f2['data2'], f3['data3'], f4['data4']

        for i in range(len(meta)-1):
            # print("%s/%d/%d" % (mode, i, len(meta)-1))
            print(meta[i][0])
            cutoff = random.uniform(100.0, 1000.0)
            y, sr = librosa.load('D:/DCASE 2018 Dataset/DCASE2018-task5-dev/' + meta[i][0], mono=False, sr=16000)
            y = HPF(y, sr, cutoff)
            librosa.output.write_wav('test.wav', y.T, sr)
            plot_figure(y, "Testfile")
            data1[i], data2[i], data3[i], data4[i] = extract_feature(y, sr)
            # librosa.output.write_wav('test.wav', y.T, sr)
            # plot_figure(y, "Testfile")


def to_hdf5_ver2(path, meta, mode):
    with h5py.File(path+'%s.hdf5' % mode) as f1:
        f1.create_dataset('data', (len(meta)-1, 501, 40), dtype='float32')

        data = f1['data']

        for i in range(len(meta)-1):
            cutoff = 1000
            y, sr = librosa.load('D:/DCASE 2018 Dataset/DCASE2018-task5-dev/' + meta[i][0], mono=True, sr=16000)
            y = HPF(y, sr, cutoff)
            data[i] = extract_feature_1ch(y, sr)


def HPF(y, sr, cutoff, order=8):
    nyq = 0.5*sr
    normal_cutoff = cutoff/nyq
    b, a = signal.butter(order, normal_cutoff, btype='high')
    y = signal.filtfilt(b, a, y)

    return y


def cat(arr):

    if arr[1] == 'absence':
        return 0
    elif arr[1] == 'cooking':
        return 1
    elif arr[1] == 'dishwashing':
        return 2
    elif arr[1] == 'eating':
        return 3
    elif arr[1] == 'other':
        return 4
    elif arr[1] == 'social_activity':
        return 5
    elif arr[1] == 'vacuum_cleaner':
        return 6
    elif arr[1] == "watching_tv":
        return 7
    elif arr[1] == 'working':
        return 8
    elif arr[1] == 'baby_cry':
        return 9
    else:
        print("Categorize Error")


def from_hdf5(file, data_name):
    h5 = h5py.File(file)
    data = h5.get(data_name).value
    return data


def cnn_model():
    model = Sequential()
    model.add(Conv2D(64, kernel_size=(7, 1), padding='same', input_shape=(501, 40, 1)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(4, 1)))
    model.add(Dropout(0.2))
    model.add(Conv2D(128, (10, 1), padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Conv2D(256, (1, 7), padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(GlobalMaxPool2D())
    model.add(Dropout(0.5))
    model.add(Dense(9, activation='softmax'))

    adam = optimizers.adam(lr=0.0001)
    model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])

    return model


def main():

    return


if __name__ == '__main__':
    main()
