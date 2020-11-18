# import math as mt
# from modules import from_hdf5
# import h5py
# import numpy as np
# import matplotlib.pyplot as plt
# from librosa import display
# from modules import read_meta
#
# # for i in range(1, 5):
# #     tmp = from_hdf5('F:/DCASE2018/traindata_without_log/fold%d_train.hdf5' % i, 'data')
# #     with h5py.File('F:/DCASE2018/traindata_without_log/fold%d_train_.hdf5' % i) as f1:
# #         f1.create_dataset('data', (len(tmp)-1, 501, 40), dtype='float32')
# #         data = f1['data']
# #         for j in range(len(tmp)-1):
# #             data[j] = tmp[j].T
# meta = read_meta('D:/DCASE 2018 Dataset/Development dataset/DCASE2018-task5-dev/evaluation_setup/fold4_train.txt')
# new = from_hdf5('F:/DCASE2018/DA_Gaussian/fold4_train.hdf5', 'data')
#
# print(len(new))
#
# for i in range(len(meta)-1):
#     new[i] = np.log10(np.maximum(new[i], 0.000001))
#
# print(len(new))
#
# h5f = h5py.File('F:/DCASE2018/DA_gaussian/fold4_train.hdf5', 'w')
# h5f.create_dataset('data', data=new)
# h5f.close()
########################################################################################
# import numpy as np
#
# wav = np.load('C:/Users/MIN/PycharmProjects/ETRI/DCASE2018/files/Sample_audio.npy')
#
# print(wav)
#
# data = np.fromstring(wav, dtype=np.float32)
#
# print(np.shape(data))
########################################################################################
# import pyaudio
# import wave
# import numpy as np
#
# FORMAT = pyaudio.paInt16
# CHANNELS = 4
# RATE = 48000
# CHUNK = 1024
# RECORD_SECONDS = 5
# WAVE_OUTPUT_FILENAME = "file.wav"
#
# p = pyaudio.PyAudio()
#
# # start Recording
#
# stream = p.open(format=pyaudio.paInt16,
#                 channels=CHANNELS,
#                 rate=RATE,
#                 input=True,
#                 input_device_index=12,
#                 frames_per_buffer=CHUNK)
#
# print("recording...")
#
# frames = []
#
# for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
#     data = stream.read(CHUNK)
#     frames.append(data)
#
# print("finished recording")
#
# # np.save('Sample_audio.npy', frames)
#
# # stop Recording
# stream.stop_stream()
# stream.close()
# p.terminate()
#
# waveFile = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
# waveFile.setnchannels(CHANNELS)
# waveFile.setsampwidth(p.get_sample_size(FORMAT))
# waveFile.setframerate(RATE)
# waveFile.writeframes(b''.join(frames))
# waveFile.close()
#########################################################################################

import math as m
omega = 0
sigma1, sigma2= 0.2, 0.2
rho = 0

eq = m.sqrt(m.pi/2)*omega**2*sigma1*sigma2*((1+rho*(1+m.sqrt(2)))*(omega**2*sigma2**2+sigma1**2)**(-3/2)-2*
                                            rho*((2*omega**2*sigma2**2+sigma1**2)**(-3/2)+(omega**2*sigma2**2+2*sigma1**2)**(-3/2)))

