import sounddevice as sd
import librosa
from operator import itemgetter
from keras.models import load_model
import numpy as np
import time
import csv
import queue
from datetime import date
from collections import Counter
import sys
import os
import pandas as pd
from tkinter import messagebox, Tk

# path = 'C:/Users/MIN/PycharmProjects/AI Flagship/Test_dcase/softmax/'
path = './softmax/'
model1 = load_model(path + 'fold1_ch1_model.h5')
model2 = load_model(path + 'fold2_ch1_model.h5')
model3 = load_model(path + 'fold3_ch1_model.h5')
model4 = load_model(path + 'fold4_ch1_model.h5')


def extract_feature(y, sr):
    if len(y) == 1:
        mel1 = librosa.feature.melspectrogram(np.asfortranarray(y), sr=sr, n_fft=1024, n_mels=40, hop_length=320,
                                              power=1, fmax=16000)
        mel1 = np.log10(np.maximum(mel1, 0.000001)).T
        return mel1, None, None, None
    elif len(y) == 2:
        mel1 = librosa.feature.melspectrogram(np.asfortranarray(y[0]), sr=sr, n_fft=1024, n_mels=40, hop_length=320,
                                              power=1, fmax=16000)
        mel1 = np.log10(np.maximum(mel1, 0.000001)).T
        mel2 = librosa.feature.melspectrogram(np.asfortranarray(y[1]), sr=sr, n_fft=1024, n_mels=40, hop_length=320,
                                              power=1, fmax=16000)
        mel2 = np.log10(np.maximum(mel2, 0.000001)).T
        return mel1, mel2, None, None
    elif len(y) == 3:
        mel1 = librosa.feature.melspectrogram(np.asfortranarray(y[0]), sr=sr, n_fft=1024, n_mels=40, hop_length=320,
                                              power=1, fmax=16000)
        mel1 = np.log10(np.maximum(mel1, 0.000001)).T
        mel2 = librosa.feature.melspectrogram(np.asfortranarray(y[1]), sr=sr, n_fft=1024, n_mels=40, hop_length=320,
                                              power=1, fmax=16000)
        mel2 = np.log10(np.maximum(mel2, 0.000001)).T
        mel3 = librosa.feature.melspectrogram(np.asfortranarray(y[3]), sr=sr, n_fft=1024, n_mels=40, hop_length=320,
                                              power=1, fmax=16000)
        mel3 = np.log10(np.maximum(mel3, 0.000001)).T
        return mel1, mel2, mel3, None
    else:
        mel1 = librosa.feature.melspectrogram(np.asfortranarray(y[0]), sr=sr, n_fft=1024, n_mels=40, hop_length=320,
                                              power=1, fmax=16000)
        mel1 = np.log10(np.maximum(mel1, 0.000001)).T
        mel2 = librosa.feature.melspectrogram(np.asfortranarray(y[1]), sr=sr, n_fft=1024, n_mels=40, hop_length=320,
                                              power=1, fmax=16000)
        mel2 = np.log10(np.maximum(mel2, 0.000001)).T
        mel3 = librosa.feature.melspectrogram(np.asfortranarray(y[3]), sr=sr, n_fft=1024, n_mels=40, hop_length=320,
                                              power=1, fmax=16000)
        mel3 = np.log10(np.maximum(mel3, 0.000001)).T
        mel4 = librosa.feature.melspectrogram(np.asfortranarray(y[4]), sr=sr, n_fft=1024, n_mels=40, hop_length=320,
                                              power=1, fmax=16000)
        mel4 = np.log10(np.maximum(mel4, 0.000001)).T
        return mel1, mel2, mel3, mel4


def prediction(ch1=None, ch2=None, ch3=None, ch4=None, nb_channels=4):

    if ch1 is not None:
        ch1 = ch1[np.newaxis, :, :, np.newaxis]
        pred1_1 = model1.predict_proba(ch1)
        pred1_2 = model2.predict_proba(ch1)
        pred1_3 = model3.predict_proba(ch1)
        pred1_4 = model4.predict_proba(ch1)
        pred1 = (1 / 4) * (pred1_1 + pred1_2 + pred1_3 + pred1_4)
    if ch2 is not None:
        ch2 = ch2[np.newaxis, :, :, np.newaxis]
        pred2_1 = model1.predict_proba(ch2)
        pred2_2 = model2.predict_proba(ch2)
        pred2_3 = model3.predict_proba(ch2)
        pred2_4 = model4.predict_proba(ch2)
        pred2 = (1 / 4) * (pred2_1 + pred2_2 + pred2_3 + pred2_4)
    if ch3 is not None:
        ch3 = ch3[np.newaxis, :, :, np.newaxis]
        pred3_1 = model1.predict_proba(ch3)
        pred3_2 = model2.predict_proba(ch3)
        pred3_3 = model3.predict_proba(ch3)
        pred3_4 = model4.predict_proba(ch3)
        pred3 = (1 / 4) * (pred3_1 + pred3_2 + pred3_3 + pred3_4)
    if ch4 is not None:
        ch4 = ch4[np.newaxis, :, :, np.newaxis]
        pred4_1 = model1.predict_proba(ch4)
        pred4_2 = model2.predict_proba(ch4)
        pred4_3 = model3.predict_proba(ch4)
        pred4_4 = model4.predict_proba(ch4)
        pred4 = (1 / 4) * (pred4_1 + pred4_2 + pred4_3 + pred4_4)

    if nb_channels == 1:
        pred1 = sorted(pred1, key=itemgetter(0))
        merge = [np.max(pred1), np.argmax(pred1), (pred1[2][1], )]
        merge_new = pred1
    elif nb_channels == 2:
        merge = [(np.max(pred1), np.argmax(pred1)), (np.max(pred2), np.argmax(pred2))]
        merge_new = [(np.max(pred1), np.argmax(pred1)), (np.max(pred2), np.argmax(pred2)),
                     (np.max(pred2), np.argmax(pred2)), (np.max(pred2), np.argmax(pred2))]
    else:
        merge = [(np.max(pred1), np.argmax(pred1)), (np.max(pred3), np.argmax(pred3))]
        merge_new = [(np.max(pred1), np.argmax(pred1)), (np.max(pred2), np.argmax(pred2)),
                     (np.max(pred3), np.argmax(pred3))
                    , (np.max(pred4), np.argmax(pred4))]

    # merge = [(np.max(pred1), np.argmax(pred1)), (np.max(pred3), np.argmax(pred3))]

    # merge = (1 / 4) * (pred1 + pred2 + pred3 + pred4)
    # merge = [(np.max(pred1), np.argmax(pred1)), (np.max(pred2), np.argmax(pred2)),
    #          (np.max(pred3), np.argmax(pred3)), (np.max(pred4), np.argmax(pred4))]

    # chan1 = [(np.max(pred1_1), np.argmax(pred1_1)), (np.max(pred2_1), np.argmax(pred2_1)),
    #          (np.max(pred3_1), np.argmax(pred3_1)), (np.max(pred4_1), np.argmax(pred4_1))]
    #
    # chan2 = [(np.max(pred1_2), np.argmax(pred1_2)), (np.max(pred2_2), np.argmax(pred2_2)),
    #          (np.max(pred3_2), np.argmax(pred3_2)), (np.max(pred4_2), np.argmax(pred4_2))]
    #
    # chan3 = [(np.max(pred1_3), np.argmax(pred1_3)), (np.max(pred2_3), np.argmax(pred2_3)),
    #          (np.max(pred3_3), np.argmax(pred3_3)), (np.max(pred4_3), np.argmax(pred4_3))]
    #
    # chan4 = [(np.max(pred1_4), np.argmax(pred1_4)), (np.max(pred2_4), np.argmax(pred2_4)),
    #          (np.max(pred3_4), np.argmax(pred3_4)), (np.max(pred4_4), np.argmax(pred4_4))]

    # print("chan1:")
    # print(chan1)
    # print("chan2:")
    # print(chan2)
    # print("chan3:")
    # print(chan3)
    # print("chan4:")
    # print(chan4)
    # print(merge)

    # merge_new = [(np.max(pred1), np.argmax(pred1)), (np.max(pred2), np.argmax(pred2)), (np.max(pred3), np.argmax(pred3))
    #              , (np.max(pred4), np.argmax(pred4))]

    merge_new = sorted(merge_new, key=itemgetter(0))
    # merge = sorted(merge, key=itemgetter(0))
    if merge[1][1] == merge[0][1] or merge[1][1] == 0:
        result = merge[1][1]
        result2 = False
    elif merge[0][0] >= 0.5 and (merge[1][0] - merge[0][0]) > 0.2:
        result = merge[1][1]
        result2 = merge[0][1]
    else:
        result = merge[1][1]
        result2 = False
    print(merge_new)
    if merge_new[3][1] == merge_new[2][1] or merge_new[3][1] == 0:
        if merge_new[2][1] != merge_new[1][1]:
            result_new = merge_new[3][1]
            result2_new = merge_new[1][1]
        else:
            result_new = merge_new[3][1]
            result2_new = False
    elif merge_new[2][0] >= 0.0 :#band (merge_new[3][0]-merge_new[2][0]) < 0.2:
        result_new = merge_new[3][1]
        result2_new = merge_new[2][1]
    else:
        result_new = merge_new[3][1]
        result2_new = False

    ## result = np.argmax(merge)
    #
    # result = merge[1][1]
    # result2 = False
    # return result, result2
    return result_new, result2_new


def num_to_list(result):
    tmp_list = [0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    tmp_list[result] = 0.5263
    return tmp_list


def num_to_event(result):
    if result == 0:
        return 'absence'
    elif result == 1:
        return 'cooking'
    elif result == 2:
        return 'dishwashing'
    elif result == 3:
        return 'eating'
    elif result == 4:
        return 'other'
    elif result == 5:
        return 'social_activity'
    elif result == 6:
        return 'vaccum_cleaner'
    elif result == 7:
        return 'watching_tv'
    elif result == 8:
        return 'working'
    elif result == 9:
        return 'baby_cry'
    else:
        print("Categorize error")
        return False


def init_py(pointer):
    # csv 파일 기록을 위한 초기화
    today = date.today()
    yesterday = date.fromtimestamp(time.time() - 60 * 60 * 24)

    # 전날 날짜로된 파일이 존재하면 읽어오고 존재하지 않는다면 False return
    if not os.path.exists('./log_file/' + yesterday.strftime("%Y-%m-%d") + '.csv'):
        pre_csvExist = False
        pre_csvLog = False
    else:
        pre_csvExist = True
        pre_csvLog = np.loadtxt('./log_file/' + yesterday.strftime("%Y-%m-%d") + '.csv', delimiter=',', dtype=str)

    # 오늘 날짜로 된 파일이 존재한다면 읽어오고 아니라면 오늘 날짜로 된 파일을 만들어 줌
    if not os.path.exists('./log_file/' + today.strftime("%Y-%m-%d") + '.csv'):
        todayLog = open('./log_file/' + today.strftime("%Y-%m-%d") + '.csv', 'w', encoding='utf-8', newline='')
        file_exist = False
    else:
        todayLog = open('./log_file/' + today.strftime("%Y-%m-%d") + '.csv', 'r', encoding='utf-8', newline='')
        file_exist = True

    # 비교를 위한 queue 초기화
    if not pre_csvExist:
        old_queue = False
        new_queue = queue.Queue(queue_size)
    else:
        old_queue = queue.Queue(queue_size)
        if pointer > 60:
            for j in range(pointer - queue_size, pointer):
                old_queue.put(np.argmax(pre_csvLog[j][1:]))
        else:
            for k in range(pointer):
                old_queue.put(np.argmax(pre_csvLog[k][1:]))
        new_queue = queue.Queue(queue_size)

    return pre_csvLog, pre_csvExist, todayLog, old_queue, new_queue, file_exist


def read_audio(fs, duration, mul, negative, nb_channels):

    # 녹음을 수행하고 prediction을 수행하는 부분
    audio = sd.rec(duration * fs, samplerate=fs, channels=nb_channels, dtype='float64')

    # print("Recording Audio")
    sd.wait()
    audio = np.multiply(audio, negative)
    audio = np.multiply(audio, mul)
    audio = np.multiply(audio, negative)
    # sf.write('./test audio file/False_alarm.wav', audio, fs)
    # print('recorded!')

    audio = audio.T
    y = librosa.resample(audio, fs, 16000)
    y = np.asfortranarray(y)
    fe1, fe2, fe3, fe4 = extract_feature(y, sr=16000)
    rr1, rr2 = prediction(fe1, fe2, fe3, fe4, nb_channels=nb_channels)
    result = ''
    if not rr2:
        result = rr1
    else:
        result = rr1

    return result, rr2


def add_two_list(list1, list2):
    sum = [0, 0, 0, 0, 0, 0, 0, 0, 0]
    try:
        for i in range(0, len(list1)):
            sum[i] = list1[i] + float(list2[i])
        return sum
    except len(list1) is not len(list2):
        print('Two list length is not equal.')
        sys.exit()


def get_pointer(pre_time):
    pointer = 360 * (int(time.strftime('%H', time.localtime(pre_time)))) + \
              6 * (int(time.strftime('%M', time.localtime(pre_time)))) + \
              round(int(time.strftime('%S', time.localtime(pre_time))) / 10)
    return pointer-4000


# 입력으로 받을 채널 수
nb_channels = 2

# audio 입력 부분 초기화
# 사용 가능한 Device list 를 나열함
print(sd.query_devices())
# sd.default.device = 'Clarett 4Pre USB, Core Audio'
fs = 44100
duration = 10
mul = np.full((441000, nb_channels), 3)
negative = np.full((441000, nb_channels), -1)

# 코드 처음 실행시 초기화 하는 부분
init_time = time.time()
pointer = get_pointer(init_time)
# print('pointer at start time : %s' % pointer)
queue_size = 60
# alert = Tk()

pre_csvLog, pre_csvExist, todayLog, old_queue, new_queue, file_exist = init_py(pointer)

# 만약 오늘 날짜로 된 파일이 존재한다면 기존 파일에 이어서 기록
if file_exist:
    today = date.today()
    tr = csv.reader(todayLog)
    lines = []
    for line in tr:
        lines.append(line)

    tmp = pointer - len(lines)

    todayLog = open('./log_file/' + today.strftime("%Y-%m-%d") + '.csv', 'w', encoding='utf-8', newline='')
    tw = csv.writer(todayLog)

    tw.writerows(lines)
    for i in range(tmp):
        tw.writerow([0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
else:
    tw = csv.writer(todayLog)
    for i in range(pointer):
        tw.writerow([0, 0, 0, 0, 0, 0, 0, 0, 0, 0])

pre_pointer = pointer


labels = ['Absence', 'Cooking', 'Dish washing', 'Eating', 'Other', 'Social activity', 'Vacuum cleaning', 'TV',
          'Working']
first_flag = True

while True:
    try:

        p_time = time.time()
        predict, predict2 = read_audio(fs, duration, mul, negative, nb_channels)

        # p_time : prediction을 시작한 시간, pointer : csv log 파일에서 현재 결과값이 저장되어야할 위치
        pointer = get_pointer(p_time)

        if not predict2:
            print('결과 : %s' % num_to_event(predict))
        else:
            print('결과 : %s, %s' % (num_to_event(predict), num_to_event(predict2)))

        if not pre_csvExist:
            result = num_to_list(predict)
        else:
            result_display = []
            tmp_result = num_to_list(predict)
            tmp_result2 = num_to_list(predict2)
            ###############################################################################
            tmp_precsv = pre_csvLog[pointer][1:]
            tmp_precsv[predict] = str(float(tmp_precsv[predict]) * 0.4737)
            result = add_two_list(tmp_result, pre_csvLog[pointer][1:])
            if not predict2:
                print(predict2)
            else:
                result = add_two_list(tmp_result2, result)

            # result = map(add, tmp_result, pre_csvLog[pointer][1:])
            # print(result[1:])
            ###############################################################################
            for i in range(-5, 5):
                if i == 0:
                    result_display.append(result)
                elif i < 0:
                    if first_flag:
                        result_display.append(pre_csvLog[pointer + i][1:].tolist())
                    else:
                        result_display.append(pre_result[i + 5])
                elif i > 0:
                    result_display.append(pre_csvLog[pointer + i][1:].tolist())
            first_flag = False

            pre_result = result_display[1:6]

            t = time.strftime('%X', time.localtime(p_time))
            # result = np.array(result)
            # print(result_display)
            nums = []
            for p in range(pointer - 5, pointer + 5):
                if p == pointer:
                    nums.append('->')
                else:
                    nums.append(p)

            pd_result = pd.DataFrame(result_display, index=nums, columns=labels)
            print(pd_result)
        # print('pointer : %d' % pointer)

        if (pointer - pre_pointer) == 2:
            tw.writerow([0, 0, 0, 0, 0, 0, 0, 0, 0, 0])

        pre_pointer = pointer

        tw.writerow([time.strftime('%X', time.localtime(p_time)), result[0], result[1], result[2], result[3], result[4],
                     result[5], result[6], result[7], result[8]])
        # print('')
        # print('%s, %s' % (time.strftime('%X', time.localtime(p_time)), result))

        # 날이 지난 경우 다시 초기화해주는 부분
        if pointer == 8640:
            todayLog.close()
            pre_csvLog, pre_csvExist, todayLog, old_queue, new_queue, state = init_py(0)
            continue

        # queue에 결과값을 넣고 비교
        if not pre_csvExist:
            if new_queue.full():
                new_queue.get()
            new_queue.put(result.index(max(result)))
            new_act = Counter(list(new_queue.queue)).most_common(1)[0][0]
            # print(new_act)
            continue
        else:
            if old_queue.full():
                old_queue.get()
            old_queue.put(np.argmax(pre_csvLog[pointer][1:]))
            if new_queue.full():
                new_queue.get()
            new_queue.put(result.index(max(result)))
            old_act = Counter(list(old_queue.queue)).most_common(1)[0][0]
            new_act = Counter(list(new_queue.queue)).most_common(1)[0][0]

            if old_act == new_act:
                print('같은 활동 중')
            else:
                # messagebox.showinfo(title='Alarm', message='다른 활동 중')
                print('다른 활동 중')
            # print("--- %s seconds ---" % (time.time() - start_time))

    except KeyboardInterrupt:
        # np.savetxt('./log_file/' + today.strftime("%Y-%m-%d") + '.csv', full_log, delimiter=',', fmt='%s')
        todayLog.close()
        print("프로그램 종료")
        sys.exit()
