import numpy as np
from keras.layers import Input, Conv2D,Activation, BatchNormalization, UpSampling2D
from keras.models import Model
import scipy.io as sio
import os
import pywt
import tensorflow as tf
import tensorflow.compat.v1.keras.backend as K
from edge_detected import edge_detected
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
    epsilon = 0
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
    inputs, inputs1,inputs2,inputs3,inputs4, inputs5,inputs6,inputs7, outputs = get_cnn()
    model = Model(inputs=[inputs, inputs1, inputs2, inputs3, inputs4, inputs5,inputs6,inputs7], outputs=outputs)
    path = 'G:\\OneDrive - njust.edu.cn\\Hyperspectral_Image_Benchmarkx8\\harvard\\method\\Edge+RFA+compute_charbonnier_loss-4\\model'
    dirs = os.listdir(path)
    for file in dirs:
        print(file)
        file = 'model_01000-0.00314.h5'
        model.load_weights('./model/' + file, by_name=True)
        test_path = 'G:\\OneDrive - njust.edu.cn\\Hyperspectral_Image_Benchmarkx8\\harvard\\method\\test'
        psnr1 = np.zeros(20)

        for i in range(20):
            ind = i + 1
            path = str(i + 1) + '.mat'
            #print('processing for %d' % ind)
            source_hs_path = os.path.join(test_path, 'hs', path)
            data = sio.loadmat(source_hs_path)
            data = data['I']
            data = np.expand_dims(data, 0)
            source_ms_path = os.path.join(test_path, 'ms', path)
            data1 = sio.loadmat(source_ms_path)
            MSI = data1['I']

            source_gt_path = os.path.join(test_path, 'gt', path)
            data_gt = sio.loadmat(source_gt_path)
            GT = data_gt['I']
            #img = edge_detected(data1)
            data11 = np.zeros(np.shape(MSI))
            data12 = np.zeros(np.shape(MSI))
            data13 = np.zeros(np.shape(MSI))
            data14 = np.zeros(np.shape(MSI))
            data15 = np.zeros(np.shape(MSI))
            data16 = np.zeros(np.shape(MSI))
            data17 = np.zeros(np.shape(MSI))
            data18 = np.zeros(np.shape(MSI))

            for j in range(np.size(MSI,2)):
                x = MSI[:, :, j]
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

                data11[:, :, j] = data5
                data12[:, :, j] = data2
                data13[:, :, j] = data3
                data14[:, :, j] = data4
                data15[:, :, j] = data6
                data16[:, :, j] = data7
                data17[:, :, j] = data8
            data11 = np.expand_dims(data11, 0)
            data12 = np.expand_dims(data12, 0)
            data13 = np.expand_dims(data13, 0)
            data14 = np.expand_dims(data14, 0)
            data15 = np.expand_dims(data15, 0)
            data16 = np.expand_dims(data16, 0)
            data17 = np.expand_dims(data17, 0)
            #data2 = np.expand_dims(img, 0)


            data_get = model.predict([data, data11, data12, data13, data14, data15, data16, data17], batch_size=1, verbose=1)
            data_get = np.reshape(data_get, (1024, 1024, 31))
            psnr1[i] = psnr(GT,data_get)
            data_get = np.array(data_get, dtype=np.float64)
            sio.savemat('./get/eval_%d.mat' % ind, {'b': data_get})
        print(np.mean(psnr1))