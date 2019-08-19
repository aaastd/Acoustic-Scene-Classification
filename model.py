import os
import matplotlib.pyplot as plt
from sklearn.preprocessing import minmax_scale
from keras.utils.np_utils import to_categorical
from keras.utils.training_utils import multi_gpu_model
from keras.models import Sequential
from keras.layers import *
from keras import optimizers
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint
from modules import read_meta
from modules import cat
from modules import from_hdf5


def draw_plot(model, fold, ch):

    hist, accuracy = model
    print(hist.history.keys())
    fig, loss_ax = plt.subplots()
    fig.canvas.set_window_title('Fold %d, Accuracy : %s' % (fold, accuracy))
    acc_ax = loss_ax.twinx()
    loss_ax.plot(hist.history['loss'], 'y', label='train loss')

    loss_ax.plot(hist.history['val_loss'], 'r', label='val loss')
    acc_ax.plot(hist.history['acc'], 'b', label='train acc')
    acc_ax.plot(hist.history['val_acc'], 'g', label='val acc')

    loss_ax.set_xlabel('epoch')
    loss_ax.set_ylabel('loss')
    acc_ax.set_ylabel('accuray')

    loss_ax.legend(loc='lower left')
    acc_ax.legend(loc='upper left')

    fig.savefig(r'Fold%d_%s Model.png' % (fold, ch), format='png')


def model(fold_num, ch):

    path = 'D:/DCASE 2018 Dataset/Development dataset/DCASE2018-task5-dev/'
    train_path = 'C:/Users/MIN/PycharmProjects/Ai Flagship/augment_by_mels/'

    meta_train = read_meta("C:/Users/MIN/PycharmProjects/ETRI/notch_filter/fold%d_train.txt" % fold_num)
    print(len(meta_train))
    meta_evaluate = read_meta(path + "evaluation_setup/fold%d_evaluate.txt" % fold_num)
    y_tmp1, y_tmp2 = [], []

    # Train set 데이터 불러오기, 정답지 생성
    # train_x = HDF5Matrix('E:/ETRI/DCASE2018/feature/fold%d_train_%s.hdf5' % (fold_num, ch), 'data')
    # train_x = np.load(path + 'feature_extraction_5/fold%d_%s_train.npy' % (fold_num, ch))
    train_x = from_hdf5('E:/ETRI/DCASE2018/feature/fold%d_train_%s.hdf5' % (fold_num, ch), 'data')
    train_y = np.zeros([len(train_x), ])
    for i in range(0, len(train_x)-1):
        train_y[i] = cat(meta_train[i])

    print(train_y)

    # Evaluate set 데이터 불러오기, 정답지 생성
    te = np.load(path + 'feature_extraction_2/%s/fold%d_%s_evaluate.npy' % (ch, fold_num, ch))
    evaluate_x = te[:, 0, :, :]
    del(te)
    evaluate_y = np.zeros([len(evaluate_x), ])

    for j in range(0, len(evaluate_x)-1):
        evaluate_y[j] = cat(meta_evaluate[j])

    # Train 데이터 포맷 변환
    num_train = train_x.shape[0]
    width = train_x.shape[1]
    height = train_x.shape[2]

    # 데이터를 0~1 값으로 바꿔줌
    for i in range(0, num_train - 1):
        minmax_scale(train_x[i], feature_range=(0, 1), axis=0)
    train_x = train_x.reshape(num_train, width, height, 1).astype(np.float64)

    # Evaluate 데이터 포맷 변환
    num_evaluate = evaluate_x.shape[0]

    for j in range(0, num_evaluate - 1):
        minmax_scale(evaluate_x[j], feature_range=(0, 1), axis=0)
    evaluate_x = evaluate_x.reshape(num_evaluate, width, height, 1).astype(np.float64)

    # 정답 값을 케라스에서 사용하는 값으로 생성
    train_y = to_categorical(train_y)
    evaluate_y = to_categorical(evaluate_y)

    # CNN 구현 부분
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
    # model.add(Dense(9, activation='sigmoid'))

    # model = multi_gpu_model(model, gpus=2)
    # optimizer 설정
    adam = optimizers.adam(lr=0.0001)

    model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])

    training_epochs = 500
    batch_size = 128

    # Early stopping
    # mode : min(감소하기를 멈췄을 때 중지), max(증가하기를 멈췄을 때 중지), auto
    # verbose : 자세히 표시하는 정도
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, verbose=2, mode='min')

    # Model checkpoint, 가장 좋은 성능일 때의 모델을 저장
    pa = './model3/'
    if not os.path.exists(pa):
        os.mkdir(pa)
    model_path = pa + 'fold%d_%s_model.h5' % (fold_num, ch)
    checkpoint = ModelCheckpoint(filepath=model_path, monitor='val_loss', verbose=1, save_best_only=True)
    # checkpoint = model.save_weights('./model2/fold%d_%s_model.h5' % (fold_num, ch))

    # 모델 학습시키기
    hist = model.fit(train_x, train_y, validation_split=0.33, nb_epoch=training_epochs, batch_size=batch_size,
                     validation_data=(evaluate_x, evaluate_y), callbacks=[early_stopping, checkpoint])

    print("모델 평가")
    evaluation = model.evaluate(evaluate_x, evaluate_y, batch_size=batch_size)
    print('%s_Model%d Accuracy: ' % (ch, fold_num) + str(evaluation[1]))

    # 학습 모델 저장
    # model.save('fold%d_%s_cnn_model.h5' % (fold_num, ch))
    # model.save_weights('fold%d_%s_cnn_model.h5' % (fold_num, ch))

    return hist, str(evaluation[1])


# 학습과정 그리기
for fold in range(1, 5):
    for i in range(1, 5):
        ch = "ch%d" % i
        draw_plot(model(fold, ch), fold, ch)

plt.show()
