import librosa
from librosa import display
import matplotlib.pyplot as plt
from scipy import signal
import random
import math as m


def notch_filter(y, fs):
    f0 = random.uniform(30.0, 2000)
    print(f0)
    w0 = f0/(fs/2)
    Q = 0.01
    b, a = signal.iirnotch(w0, Q)
    # y = signal.filtfilt(b, a, y)
    y = signal.lfilter(b, a, y)

    return y


def time_masking(y, sr):
    start = random.uniform(0.0, 10.0)
    width = random.uniform(1, 4)
    print(start)
    print(width)
    start = m.ceil(start*sr)
    for i in range(m.ceil(width*sr)):
        if (start+i) > len(y):
            return y
        else:
            y[start+i] = 0
    return y


y, sr = librosa.load('D:/DCASE 2018 Dataset/DCASE2018-task5-dev/audio/DevNode4_ex230_553.wav', mono=True, sr=16000)

# y, sr = librosa.load('D:/DCASE 2018 Dataset/DCASE2018-task5-dev/audio/DevNode4_ex87_2_a_n.wav', mono=False, sr=16000)
# y, sr = librosa.load('test.wav', mono=False, sr=16000)

y = notch_filter(y, sr)
y = time_masking(y, sr)
librosa.output.write_wav('test.wav', y.T, sr)

plt.plot()
display.waveplot(y)
plt.show()