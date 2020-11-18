from sklearn.preprocessing import minmax_scale
from sklearn.metrics import classification_report
from keras.models import load_model
from operator import itemgetter
import numpy as np
from modules import from_hdf5
from modules import cnn_model
import time


def compile_model(model, x_test):

    proba = model.predict(x_test)
    return proba


def list_to_string(list):
    return str(list).replace('[', '').replace(']', '')


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
    else:
        print("Categorize Error")


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


def find_answer(meta, file_name):
    for i in range(0, len(meta)-1):
        if meta[i][0] == file_name:
            return cat(meta[i])


def calc_prob(x_test, fold_num, ch):

    # path3 = 'C:/Users/MIN/PycharmProjects/AI Flagship/Test_dcase/softmax/'
    # path3 = 'F:/DCASE2018/notch #1/model/'
    # path3 = 'F:/DCASE2018/original/feature/'
    path3 = 'E:/notch #2/'
    for i in range(0, len(x_test) - 1):
        minmax_scale(x_test[i], feature_range=(0, 1), axis=0)
    x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], x_test.shape[2], 1).astype(np.float64)

    # loaded1 = cnn_model()
    # loaded1.load_weights('model3/Fold%d_ch1_Model.h5' % fold_num)
    # loaded2 = cnn_model()
    # loaded2.load_weights('model3/Fold%d_ch2_Model.h5' % fold_num)
    # loaded3 = cnn_model()
    # loaded3.load_weights('model3/Fold%d_ch3_Model.h5' % fold_num)
    # loaded4 = cnn_model()
    # loaded4.load_weights('model3/Fold%d_ch4_Model.h5' % fold_num)
    #
    # pred1 = compile_model(loaded1, x_test)
    # pred2 = compile_model(loaded2, x_test)
    # pred3 = compile_model(loaded3, x_test)
    # pred4 = compile_model(loaded4, x_test)

    # prediction 실행 부분
    model1 = load_model('F:/DCASE2018/DA_Gaussian/model/fold%d_ch1_model.h5' % fold_num)
    # model1 = load_model(path3+'fold%d_ch1_model.h5' % fold_num)
    # model2 = load_model(path3+'fold%d_ch2_model.h5' % fold_num)
    # model3 = load_model(path3+'fold%d_ch3_model.h5' % fold_num)
    # model4 = load_model(path3+'fold%d_ch4_model.h5' % fold_num)

    print("Fold%d_%s is in process..." % (fold_num, ch))
    print(x_test.shape)
    pred1 = model1.predict_proba(x_test)
    # pred2 = model2.predict_proba(x_test)
    # pred3 = model3.predict_proba(x_test)
    # pred4 = model4.predict_proba(x_test)

    # merge = (1/4)*(pred1 + pred2 + pred3 + pred4)
    merge = pred1

    return merge


def run_model(fold_num):
    # binary, micro, macro, weighted, samples available
    f1_meta = 'weighted'

    path = 'D:/DCASE 2018 Dataset/Development dataset/DCASE2018-task5-dev/'
    path2 = 'E:/ETRI/DCASE2018/feature2/'

    # meta = read_meta(path + "evaluation_setup/fold%d_test.txt" % fold_num)
    test_meta = read_meta('F:/DCASE2018/evaluation/unknown_mic_meta.txt')
    meta_origin = read_meta(path + 'meta.txt')

    x_test1 = from_hdf5('F:/DCASE2018/evaluation/eval_unknown.hdf5', 'data')
    # x_test1 = from_hdf5(path2+'fold%d_test_ch%d.hdf5' %(fold_num, 1), 'data%d' % 1)
    # x_test2 = from_hdf5(path2+'fold%d_test_ch%d.hdf5' %(fold_num, 2), 'data%d' % 2)
    # x_test3 = from_hdf5(path2+'fold%d_test_ch%d.hdf5' %(fold_num, 3), 'data%d' % 3)
    # x_test4 = from_hdf5(path2+'fold%d_test_ch%d.hdf5' %(fold_num, 4), 'data%d' % 4)

    # 정답지 만들기
    cnt_ans = np.zeros([9, 2], dtype=int)
    y_test = np.zeros([len(x_test1), ], dtype=int)
    # y_tmp = []
    for i in range(0, len(x_test1) - 1):

        y_test[i] = cat(test_meta[i])

        # ans = find_answer(meta_origin, meta[i][0])
        # y_test[i] = ans
        # y_tmp.append(find_answer(meta_origin, meta[i][0]))
        # cnt_ans[ans] += 1

    # y_test = np.array(y_tmp)

    # 채널 별로 확률 값을 가져와서 결과값 도출
    pre1 = calc_prob(x_test1, fold_num, 'ch1')
    # pre2 = calc_prob(x_test2, fold_num, 'ch2')
    # pre3 = calc_prob(x_test3, fold_num, 'ch3')
    # pre4 = calc_prob(x_test4, fold_num, 'ch4')

    # 채널별로 가장 높은 확률을 가진 값을 가져옴(각각의 채널마다)
    # 그 후, 확률이 높은 순으로 정렬
    predict_list = np.zeros([len(pre1), 4])
    predict = np.zeros([len(pre1)], dtype='int')
    for i in range(0, len(pre1)-1):
        # a = [(np.max(pre1[i]), np.argmax(pre1[i])), (np.max(pre2[i]), np.argmax(pre2[i])),
        #      (np.max(pre3[i]), np.argmax(pre3[i])), (np.max(pre4[i]), np.argmax(pre4[i]))]
        # a = sorted(a, key=itemgetter(0))
        #
        # predict_list[i][0] = a[3][1]
        # predict_list[i][1] = a[2][1]
        # predict_list[i][2] = a[1][1]
        # predict_list[i][3] = a[0][1]

        # # 그 중에서 1 개를 뽑아서 정답 리스트에 넣음
        # predict[i] = predict_list[i][0]
        predict[i] = np.argmax(pre1[i])
    # add_pre = (1/4) * (pre1+pre2+pre3+pre4)
    # prediction = np.zeros([len(pre1), 1])

    # for cnt in range(0, len(pre1)):
    #     prediction[cnt] = np.argmax(add_pre[cnt])

    # print(len(prediction))
    # text = open('single_fold%d_result.txt' % fold_num, 'at')
    # for x in range(0, len(meta) - 1):
    #     text.write("%s\t%s\t%s\n" % (meta[x][0], list_to_string(y_test[x]), list_to_string(predict[x])))
    # text.close()

    # # f1_score 함수에 넣기 위해서 형 변환
    # y_pred = MultiLabelBinarizer().fit_transform(predict)
    # y_test = MultiLabelBinarizer().fit_transform(y_test)
    y_pred = predict.tolist()
    # sklearn metric 함수 사용
    target_names = ['absence', 'cooking', 'dishwashing', 'eating', 'other'
        , 'social_activity', 'vacuum_cleaner', 'watching_tv', 'working']
    print(classification_report(y_test, y_pred, target_names=target_names))

    return


start = time.time()
for fold_num in range(1, 5):
        run_model(fold_num)

print("time :", time.time() - start)