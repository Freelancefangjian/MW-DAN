from __future__ import absolute_import, division
import tensorflow as tf
import numpy as np
tf.test.is_gpu_available()
import cv2 as cv
import tensorflow.compat.v1.keras.backend as K
import os
from keras.callbacks import ModelCheckpoint
# tf.config.experimental.list_physical_devices('GPU')
def set_gpu_config(device = "0",fraction=0.25):
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = fraction
    config.gpu_options.visible_device_list = device
    K.set_session(tf.compat.v1.Session(config=config))

set_gpu_config("0",0.25)
from keras.layers import Input, Conv2D,Activation, BatchNormalization, UpSampling2D,MaxPooling2D
from DataSet import DataSet
from config import FLAGES
import pywt
from keras.models import Model
from keras.optimizers import Adam
from sklearn.utils import shuffle
def tf_log10(x):
    n = tf.compat.v1.log(x)
    d = tf.compat.v1.log(tf.constant(10, dtype=n.dtype))
    return n/d
def psnr(y_ture, y_pred):
    max_pixel = 1.0
    return 10.0*tf_log10((max_pixel**2)/(K.mean(K.square(y_pred-y_ture))))
def RFA(layer, layer2, layer3, layer4, channel = 128):
    layer1 = layer

    layer11 = tf.concat([layer1, layer2],3)
    layer11 = Conv2D(channel, (3, 3), padding='same')(layer11)
    layer11 = Activation('relu')(layer11)
    layer11 = Conv2D(channel, (3, 3), padding='same')(layer11)
    layer12 = tf.add(layer11, layer1)

    layer12 = tf.concat([layer12, layer3], 3)
    layer12 = Conv2D(channel, (3, 3), padding='same')(layer12)
    layer12 = Activation('relu')(layer12)
    layer12 = Conv2D(channel, (3, 3), padding='same')(layer12)
    layer13 = tf.add(layer11, layer12)

    layer13 = tf.concat([layer13, layer4], 3)
    layer13 = Conv2D(channel, (3, 3), padding='same')(layer13)
    layer13 = Activation('relu')(layer13)
    layer13 = Conv2D(channel, (3, 3), padding='same')(layer13)
    layer14 = tf.add(layer12, layer13)

    layer15 = tf.concat([layer11, layer12, layer13, layer14], 3)

    layer16 = Conv2D(channel, (1, 1), padding='same')(layer15)
    outputs = tf.add(layer16, layer)
    return outputs
def schedule(epoch):
    if epoch < 200:
        return .001
    if epoch < 1000:
        return 0.0001
    else:
        return .00001

def get_cnn(channel=128):
    inputs = l = Input((4, 4, 31), name='inputs')
    inputs1 = l1 = Input((32, 32, 3), name='inputs1')
    inputs2 = l2 = Input((32, 32, 3), name='inputs2')
    inputs3 = l3 = Input((32, 32, 3), name='inputs3')
    inputs4 = l4 = Input((32, 32, 3), name='inputs4')
    inputs5 = l5 = Input((32, 32, 3), name='inputs5')
    inputs6 = l6 = Input((32, 32, 3), name='inputs6')
    inputs7 = l7 = Input((32, 32, 3), name='inputs7')


    # conv11
    l = UpSampling2D((8, 8))(l)

    l = tf.concat([l, l1], 3)
    #l = tf.concat([l, l1], 3)

    l = Conv2D(channel, (3, 3), padding='same')(l)
    l = RFA(l, l2, l3, l4)

    l = Conv2D(channel, (3, 3), padding='same')(l)
    l = RFA(l, l5, l6, l7)

    l3 = Conv2D(31, (5, 5), padding='same')(l)
    l3 = Activation('relu')(l3)

    output = l3

    return inputs, inputs1,inputs2,inputs3,inputs4, inputs5,inputs6,inputs7, output
def compute_charbonnier_loss(tensor1, tensor2, is_mean=True):
    epsilon = 1e-6
    if is_mean:
        loss = tf.reduce_mean(tf.reduce_mean(tf.sqrt(tf.square(tf.subtract(tensor1,tensor2))+epsilon), [1, 2, 3]))
    else:
        loss = tf.reduce_mean(tf.reduce_sum(tf.sqrt(tf.square(tf.subtract(tensor1,tensor2))+epsilon), [1, 2, 3]))

    return loss
dec_lo = [1/16, 4/16, 6/16, 4/16, 1/16]
dec_lo, dec_hi, rec_lo, rec_hi = [1/16, 4/16, 6/16, 4/16, 1/16], [-1/16, -4/16, 10/16, -4/16, -1/16], [1/16, 4/16, 6/16, 4/16, 1/16], [-1/16, -4/16, 10/16, -4/16, -1/16]
filter_bank = [dec_lo, dec_hi, rec_lo, rec_hi]
myWavelet = pywt.Wavelet(name="myHaarWavelet", filter_bank=filter_bank)


class HaarFilterBank(object):
    @property
    def filter_bank(self):
        #c = math.sqrt(2) / 2
        dec_lo, dec_hi, rec_lo, rec_hi = [1/16, 4/16, 6/16, 4/16, 1/16], [-1/16, -4/16, 10/16, -4/16, -1/16], [1/16, 4/16, 6/16, 4/16, 1/16], [-1/16, -4/16, 10/16, -4/16, -1/16]
        return [dec_lo, dec_hi, rec_lo, rec_hi]
if __name__ == "__main__":
    dataset = DataSet(FLAGES.pan_size, FLAGES.ms_size, FLAGES.img_path, FLAGES.data_path, FLAGES.batch_size,
                      FLAGES.stride)
    MSI = dataset.pan
    HSI = dataset.ms
    GT = dataset.gt
    #MSI_edge = dataset.pan_edge
    MSI, HSI, GT = shuffle(MSI, HSI, GT, random_state=0)
    #MSI, HSI, GT = shuffle_in_unison(MSI, HSI, GT)
    inputs, inputs1,inputs2,inputs3,inputs4, inputs5,inputs6,inputs7, outputs = get_cnn()
    model = Model(inputs=[inputs, inputs1,inputs2, inputs3, inputs4, inputs5, inputs6, inputs7], outputs=outputs)
    model.summary()
    optim = Adam(1e-4)
    loss = compute_charbonnier_loss
    model.compile(optim, loss='mae', metrics=[psnr])
    filepath = "model_{epoch:05d}-{val_loss:.5f}.h5"
    save_dir = os.path.join(os.getcwd(), 'model')
    checkpointer = ModelCheckpoint(os.path.join(save_dir, filepath), monitor='val_loss', verbose=1, save_best_only=False, mode='auto', period=1)
    #lr_scheduler = tf.keras.callbacks.LearningRateScheduler(schedule)
    data11 = np.zeros(np.shape(MSI))
    data12 = np.zeros(np.shape(MSI))
    data13 = np.zeros(np.shape(MSI))
    data14 = np.zeros(np.shape(MSI))
    data15 = np.zeros(np.shape(MSI))
    data16 = np.zeros(np.shape(MSI))
    data17 = np.zeros(np.shape(MSI))
    data18 = np.zeros(np.shape(MSI))
    for i in range(np.size(MSI,0)):
        for j in range(np.size(MSI,3)):
            x = MSI[i,:,:,j]
            myOtherWavelet = pywt.Wavelet(name="myHaarWavelet", filter_bank=filter_bank)
            cA1, cA2 = pywt.swtn(x, myOtherWavelet, level=2)
            data1 = cA1['aa']
            data2 = cA1['ad']
            data3 = cA1['da']
            data4 = cA1['dd']
            data5 = cA2['aa']
            data6 = cA2['ad']
            data7 = cA2['da']
            data8 = cA2['dd']

            data11[i, :, :, j] = data5
            data12[i, :, :, j] = data2
            data13[i, :, :, j] = data3
            data14[i, :, :, j] = data4
            data15[i, :, :, j] = data6
            data16[i, :, :, j] = data7
            data17[i, :, :, j] = data8
    x_train = HSI
    x_train1 = data11
    x_train2 = data12
    x_train3 = data13
    x_train4 = data14
    x_train5 = data15
    x_train6 = data16
    x_train7 = data17
    #x_train2 = MSI_edge
    y_train = GT

    model.fit([x_train, x_train1, x_train2, x_train3, x_train4, x_train5, x_train6, x_train7],  y_train, epochs=1000, batch_size=128, validation_split=0.2, initial_epoch=0,
              steps_per_epoch=100, callbacks=[checkpointer])

