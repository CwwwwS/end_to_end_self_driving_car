# -*- coding: utf-8 -*-
import pandas as pd
import cv2, os
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import * # Lambda, Conv2D, Dropout, Dense,MaxPooling2D, Flatten,LSTM,TimeDistributed,ConvLSTM2D
from keras.callbacks import ModelCheckpoint
import numpy as np
from keras.utils import print_summary
import matplotlib.image as mpimg
from keras.optimizers import Adam
import tensorflow as tf
from keras.models import load_model
from keras.layers import add
from keras.models import Model
from keras import backend as K

import netron

class PowerMode_autopilot:

    def __init__(self, data_path='speed75', learning_rate=1.0e-4, keep_prob=0.5, batch_size=32,
                 save_best_only=True, test_size=0.2, steps_per_epoch=800, epochs=10):
        self.IMAGE_HEIGHT, self.IMAGE_WIDTH, self.IMAGE_CHANNELS = 66, 200, 3
        self.data_path = data_path
        self.learning_rate = learning_rate
        self.keep_prob = keep_prob
        self.save_best_only = save_best_only
        self.batch_size = batch_size
        self.test_size = test_size
        self.steps_per_epoch = steps_per_epoch
        self.epochs = epochs

    # 加载训练数据并将其分解为训练和验证集
    def load_data(self):
        # 从csv读取数据
        data_df = pd.read_csv(os.path.join(os.getcwd(), self.data_path, 'driving_log.csv')
                              , names=['center', 'left', 'right', 'steering', 'throttle', 'reverse', 'speed'])
        X = data_df[['center', 'left', 'right']].values



        y = data_df['steering'].values
        # 随机划分训练集和测试集
        X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=self.test_size, random_state=0,shuffle=False) #
        return X_train, X_valid, y_train, y_valid



    # 构建模型
    def build_model(self):
        model = Sequential()
        model.add(
            Lambda(lambda x: x / 127.5 - 1.0, input_shape=(self.IMAGE_HEIGHT, self.IMAGE_WIDTH, self.IMAGE_CHANNELS)))

        model.add(TimeDistributed(Conv2D(32, (5, 5), activation='elu', strides=2)))
        model.add(TimeDistributed(MaxPooling2D((2, 2), border_mode='valid')))
        # model.add(Conv2D(36, (5, 5), activation='elu', strides=2))
        # model.add(Conv2D(48, (5, 5), activation='elu', strides=2))
        model.add(TimeDistributed(Conv2D(48, (5, 5), activation='elu')))
        model.add(TimeDistributed(MaxPooling2D((2, 2), border_mode='valid')))
        model.add(TimeDistributed(Conv2D(32, (3, 3), activation='elu')))
        #model.add(TimeDistributed(MaxPooling2D((2, 2), border_mode='valid')))
        model.add(TimeDistributed(Conv2D(64, (3, 3), activation='elu')))
        model.add(Dropout(self.keep_prob))


        model.add(ConvLSTM2D(32,(3,3), padding='same',activation='elu',return_sequences=True))

        model.add(ConvLSTM2D(64,(3,3), padding='same',activation='elu', return_sequences=True))

        model.add(Flatten())


        model.add(Dense(128, activation='elu'))
        model.add(Dense(64, activation='elu'))
        model.add(Dense(16, activation='elu'))
        model.add(Dense(1))
        model.summary()
        return model

    # 加载图片
    def load_image(self, image_file):
        return mpimg.imread(os.path.join(self.data_path, image_file.strip()))

    # ---------增强处理-------
    # 从中心、左或右随机选择一个图像，并进行调整
    def choose_image(self, center, left, right, steering_angle):
        #choice = np.random.choice(3)
        choice = 2
        if choice == 0:
            return self.load_image(left), float(steering_angle) + 0.2
        elif choice == 1:
            return self.load_image(right), float(steering_angle) - 0.2
        return self.load_image(center), float(steering_angle)

    # 随机反转图片
    def random_flip(self, image, steering_angle):
        if np.random.rand() < 0.5:
            image = cv2.flip(image, 1)
            steering_angle = -steering_angle
        return image, steering_angle

    # 随机平移变换model.summary()
    def random_translate(self, image, steering_angle, range_x, range_y):
        trans_x = range_x * (np.random.rand() - 0.5)
        trans_y = range_y * (np.random.rand() - 0.5)
        steering_angle += trans_x * 0.002
        trans_m = np.float32([[1, 0, trans_x], [0, 1, trans_y]])
        height, width = image.shape[:2]
        image = cv2.warpAffine(image, trans_m, (width, height))
        return image, steering_angle

    # 添加随机阴影
    def random_shadow(self, image):

        x1, y1 = self.IMAGE_WIDTH * np.random.rand(), 0
        x2, y2 = self.IMAGE_WIDTH * np.random.rand(), self.IMAGE_HEIGHT
        # xm, ym = np.mgrid[0:self.IMAGE_HEIGHT, 0:self.IMAGE_WIDTH]
        xm, ym = np.mgrid[0:image.shape[0], 0:image.shape[1]]

        mask = np.zeros_like(image[:, :, 1])
        mask[(ym - y1) * (x2 - x1) - (y2 - y1) * (xm - x1) > 0] = 1
        cond = mask == np.random.randint(2)
        s_ratio = np.random.uniform(low=0.2, high=0.5)

        hls = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
        hls[:, :, 1][cond] = hls[:, :, 1][cond] * s_ratio
        return cv2.cvtColor(hls, cv2.COLOR_HLS2RGB)

    # 随机调整亮度
    def random_brightness(self, image):
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        ratio = 1.0 + 0.4 * (np.random.rand() - 0.5)
        hsv[:, :, 2] = hsv[:, :, 2] * ratio
        return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)

    # 产生一个增强图像和调整转向角
    def augument(self, center, left, right, steering_angle, range_x=100, range_y=10):
        image, steering_angle = self.choose_image(center, left, right, steering_angle)  # 随机选择一个图像，并进行调整
        image, steering_angle = self.random_flip(image, steering_angle)  # 翻转
        image, steering_angle = self.random_translate(image, steering_angle, range_x, range_y)  # 移动
        #image = self.random_shadow(image)  # 加阴影
        #image = self.random_brightness(image)  # 亮度

        return image, steering_angle

    # ------图像预处理-------------

    # 除去顶部的天空和底部的汽车正面
    def crop(self, image):
        return image[60:-25, :, :]

    # 调整图像大小
    def resize(self, image):
        return cv2.resize(image, (self.IMAGE_WIDTH, self.IMAGE_HEIGHT), cv2.INTER_AREA)

    # 转换RGB为YUV格式
    def rgb2yuv(self, image):
        return cv2.cvtColor(image, cv2.COLOR_RGB2YUV)

    # 图像预处理
    def preprocess(self, image):
        image = self.crop(image)
        image = self.resize(image)
        #image = self.rgb2yuv(image)
        return image

    # -------生成训练图像--------------
    # 生成训练图像，给出图像路径和相关的转向角
    def batch_generator(self, image_paths, steering_angles, is_training):
        images = np.empty([self.batch_size, self.IMAGE_HEIGHT, self.IMAGE_WIDTH, self.IMAGE_CHANNELS])
        steers = np.empty(self.batch_size)
        while True:
            i = 0
            for index in np.random.permutation(image_paths.shape[0]):
                center, left, right = image_paths[index]
                steering_angle = steering_angles[index]
                # 0.6的概率做图像增强
                if is_training and np.random.rand() < 0.6:
                    image, steering_angle = self.augument(center, left, right, steering_angle)
                else:
                    image = self.load_image(center)
                # 将图像和转向角度添加到批处理中
                images[i] = self.preprocess(image)
                steers[i] = steering_angle
                i += 1
                if i == self.batch_size:
                    break
            yield images, steers

    # 训练数据
    def train_model(self, model, X_train, X_valid, y_train, y_valid):
        # 用户每次epoch之后保存模型数据
        checkpoint = ModelCheckpoint('model75-resnet_nobn-4-{epoch:03d}.h5',
                                     monitor='val_loss',
                                     verbose=0,
                                     # 如果save_best_only = True，则最近验证误差最好的模型数据会被保存下来
                                     save_best_only=self.save_best_only,
                                     mode='auto')

        model.compile(loss='mean_squared_error', optimizer=Adam(lr=self.learning_rate))

        #CPU做图像的实时数据增强,并行在GPU训练模型
        model.fit_generator(self.batch_generator(X_train, y_train, True),
                            steps_per_epoch=self.steps_per_epoch,
                            epochs=self.epochs,
                            # max_queue_size=1,
                            validation_data=self.batch_generator(X_valid, y_valid, False),
                            validation_steps=len(X_valid),
                            callbacks=[checkpoint],
                            verbose=1)


def Conv2D_BN(x, filters, kernel_size, strides=(1, 1), padding='same', name=None):
    if name:
        bn_name = name + '_bn'
        conv_name = name + '_conv'
    else:
        bn_name = None
        conv_name = None
    x = Conv2D(filters, kernel_size, strides=strides, padding=padding, activation='relu', name=conv_name)(x)
    #x = BatchNormalization(name=bn_name)(x)
    # x = TimeDistributed(x)
    return x

def ConvLSTM2D_BN(x, filters, kernel_size, strides=(1, 1), padding='same', name=None):
    if name:
        bn_name = name + '_bn'
        conv_name = name + '_conv'
    else:
        bn_name = None
        conv_name = None
    x = ConvLSTM2D(32, (3, 3), padding='same', activation='relu', return_sequences=False, name=conv_name)(x)
    #x = BatchNormalization(name=bn_name)(x)
    return x

def squeeze_function(x):
    return tf.expand_dims(x, 1)

def squeeze_function_output_shape(input_data):
    return (None, 3, 7,24, 32)

def bottleneck_block(input_tensor, filters=(32, 32, 32), strides=(2, 2), is_conv_shortcuts=False):
    """

    :param input_tensor:
    :param filters:
    :param strides:
    :param is_conv_shortcuts: 直接连接或者投影连接
    :return:
    """
    filters_1, filters_2, filters_3 = filters
    # input_temp = K.eval(input_tensor)
    x = Conv2D_BN(input_tensor, filters=filters_1, kernel_size=(3, 3), strides=strides, padding='same')
    # x = K.eval(x)
    # x = Conv2D_BN(x, filters=filters_3, kernel_size=(3, 3))
    # x = x[:,np.newaxis,:,:,:]
    # x = RepeatVector(1)(x)
    # x = Lambda()
    # x = K.eval(x)
    # x = tf.expand_dims(x, 1)
    x = Lambda(squeeze_function,
               output_shape=squeeze_function_output_shape)(x)

    # x = [x]
    x = ConvLSTM2D_BN(x, filters=filters_3, kernel_size=(3, 3))

    if is_conv_shortcuts:
        short_cut = Conv2D_BN(input_tensor, filters=filters_3, kernel_size=(3, 3), strides=strides)
        x = add([x, short_cut])
        print(x)
    else:
        x = add([x, input_tensor])
    return x

def ResNet(input_shape_1=(66, 200, 3)):
    """

    :param input_shape:
    :param n_classes:
    :return:
    """
    #model = Sequential()
    # model.add(
    # x=  Lambda(lambda x: x / 127.5 - 1.0, input_shape=input_shape_1)
    input_layer = Input(shape=input_shape_1)
    x = Lambda(lambda x: x / 127.5 - 1.0)(input_layer)
    x = ZeroPadding2D((3, 3))(x)
    # block1
    x = Conv2D_BN(x, filters=32, kernel_size=(5, 5), strides=(2, 2), padding='valid')

    x = Conv2D_BN(x, filters=48, kernel_size=(5, 5), strides=(2, 2), padding='valid')
    # x = TimeDistributed(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(x))
    x = Conv2D_BN(x, filters=32, kernel_size=(3, 3), strides=(2, 2), padding='valid')
    #x = Conv2D_BN(x, filters=64, kernel_size=(3, 3), strides=(1, 1), padding='valid')
    # block2a
    x = bottleneck_block(x, filters=(32, 32, 32), strides=(1, 1), is_conv_shortcuts=True)

    x = Flatten()(x)
   # x = Dense(n_classes, activation='softmax')(x)

    x = Dense(128, activation='relu')(x)
    x = Dense(64, activation='relu')(x)
    x = Dense(16, activation='relu')(x)
    x = Dense(1, activation='relu')(x)
    model = Model(inputs=input_layer, outputs=x)
    return model


def main():
    autopilot = PowerMode_autopilot(data_path='speed75', learning_rate=1.0e-4, keep_prob=0.5, batch_size=32,
                                    save_best_only=True, test_size=0.2, epochs=10)

    #data = autopilot.load_data()

    #model = autopilot.build_model()
    #model_1 = ResNet()
    # model = load_model('model-xx.h5')
    # model.summary()
    #autopilot.train_model(model_1, *data)

    #autopilot.train_model(model, *data)
    #print_summary(model)
    #print_summary(model_1)

if __name__ == '__main__':
    main()
