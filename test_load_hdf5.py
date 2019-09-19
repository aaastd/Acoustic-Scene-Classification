import h5py


def from_hdf5(file, data_name):
    h5 = h5py.File(file)
    data = h5.get(data_name).value
    print(type(data))


from_hdf5('E:/ETRI/DCASE2018/feature/fold1_train_ch1.hdf5', 'data')