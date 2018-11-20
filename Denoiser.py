from keras.datasets import mnist
import numpy as np
import matplotlib.pyplot as plt
from keras.callbacks import TensorBoard
from keras.layers import Conv2D, UpSampling2D, MaxPooling2D
from keras.models import Model
from keras import Input, losses
from PIL import Image
import cv2
import os

def get_image_path(is_test, s, num):
    assert (s == 128 or s == 64)
    path = os.path.join('/Users/hthieu/PycharmProjects/CS446_Final_Project', "xray_images/")
    image_name = ""
    if is_test:
        path += 'test_images_'
        image_name += 'test_'
    else:
        path += 'train_images_'
        image_name += 'train_'
    if s == 64:
        path += '64x64'
    elif s == 128:
        path += '128x128'
    num_str = format(num, "05")
    image_name += num_str + ".png"
    return path+"/"+image_name

def load_data():
    start_idx = 4000
    end_idx = 19000
    N = end_idx - start_idx
    x_train = np.zeros((N, 128, 128))
    y_train = np.zeros((N, 128, 128))

    for i in range(start_idx, end_idx):
        clean_img_128 = cv2.imread(get_image_path(False, 128, i), cv2.IMREAD_GRAYSCALE)
        noisy_img_64 = cv2.imread(get_image_path(False, 64, i), cv2.IMREAD_GRAYSCALE)
        noisy_img_128 = cv2.resize(noisy_img_64, (128, 128))
        x_train[i - start_idx] = noisy_img_128
        y_train[i - start_idx] = clean_img_128

    start_idx = 19000
    end_idx = 19100
    N = end_idx - start_idx
    x_test = np.zeros((N, 128, 128))
    y_test = np.zeros((N, 128, 128))

    for i in range(start_idx, end_idx):
        clean_img_128 = cv2.imread(get_image_path(False, 128, i), cv2.IMREAD_GRAYSCALE)
        noisy_img_64 = cv2.imread(get_image_path(False, 64, i), cv2.IMREAD_GRAYSCALE)
        noisy_img_128 = cv2.resize(noisy_img_64, (128, 128))
        x_test[i - start_idx] = noisy_img_128
        y_test[i - start_idx] = clean_img_128
    return x_train, y_train, x_test, y_test

# def load_latest_model():


x_train, y_train, x_test, y_test = load_data()
x_train = x_train.astype('float32') / 255.
x_train = np.reshape(x_train, (len(x_train), 128, 128, 1))  # adapt this if using `channels_first` image data format
x_test = x_test.astype('float32') / 255.
x_test = np.reshape(x_test, (len(x_test), 128, 128, 1))  # adapt this if using `channels_first` image data format


y_train = y_train.astype('float32') / 255.
y_train = np.reshape(y_train, (len(y_train), 128, 128, 1))  # adapt this if using `channels_first` image data format
y_test = y_test.astype('float32') / 255.
y_test = np.reshape(y_test, (len(y_test), 128, 128, 1))  # adapt this if using `channels_first` image data format


input_img = Input(shape=(128, 128, 1))  # adapt this if using `channels_first` image data format

x = Conv2D(32, (3, 3), activation='relu', padding='same')(input_img)
x = MaxPooling2D((2, 2), padding='same')(x)
x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
encoded = MaxPooling2D((2, 2), padding='same')(x)

# at this point the representation is (7, 7, 32)

x = Conv2D(32, (3, 3), activation='relu', padding='same')(encoded)
x = UpSampling2D((2, 2))(x)
x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
x = UpSampling2D((2, 2))(x)
decoded = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)


model = Model(input_img, decoded)
model.compile(loss=losses.mean_squared_error, optimizer='sgd')

num_epoch = 10000
save_every_epoch = 50
cur_epoch = 0
while cur_epoch < num_epoch:
    model.fit(x_train, y_train,
              epochs=save_every_epoch,
              batch_size=128,
              shuffle=True,
              validation_data=(x_test, y_test))
    cur_epoch += save_every_epoch
    model.summary()
    model.save('normal_'+str(cur_epoch) + '.h5')
    test_sample = x_test[0]
    test_prediction = model.predict(np.array([test_sample]))[0]
    test_sample = test_sample * 255
    test_sample = test_sample.astype('uint8').reshape((128, 128))
    test_prediction = test_prediction * 255
    test_prediction = test_prediction.astype('uint8').reshape((128, 128))
    im = Image.fromarray(test_prediction)
    im.save('output_noise_' + str(cur_epoch) + '.png')


# test_sample = x_test[0]
# test_prediction = model.predict(np.array([test_sample]))[0]
# test_sample = test_sample*255
# test_sample = test_sample.astype('uint8').reshape((128,128))
# test_prediction = test_prediction*255
# test_prediction = test_prediction.astype('uint8').reshape((128,128))
# # test_sample = np.array(255*test_sample, dtype='uint8')
# # test_prediction = np.array(255*test_prediction, dtype='uint8')
#
# im = Image.fromarray(test_sample)
# im.save('input_noise_' + str(num_epoch) + '.png')
# im = Image.fromarray(test_prediction)
# im.save('output_noise_' + str(num_epoch) + '.png')


