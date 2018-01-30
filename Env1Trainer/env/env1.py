from __future__ import print_function
import gym
from gym import error, spaces, utils
from gym.utils import seeding
from gym.utils import seeding

from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.optimizers import Adam
from keras.layers.normalization import BatchNormalization
from keras.utils import np_utils
from keras.layers import Conv2D, MaxPooling2D, ZeroPadding2D, GlobalAveragePooling2D
from keras.preprocessing.image import ImageDataGenerator


class MNIST_CNN:
    def __init__(self):
        (self.X_train, self.y_train), (self.X_test, self.y_test) = mnist.load_data()
        self.X_train = self.X_train.reshape(self.X_train.shape[0], 28, 28, 1)
        self.X_test = self.X_test.reshape(self.X_test.shape[0], 28, 28, 1)

        self.X_train = self.X_train.astype('float32')
        self.X_test = self.X_test.astype('float32')

        self.X_train /= 255
        self.X_test /= 255

        self.number_of_classes = 10

        self.Y_train = np_utils.to_categorical(self.y_train, self.number_of_classes)
        self.Y_test = np_utils.to_categorical(self.y_test, self.number_of_classes)

    def make_model(self, n_conv):
        self.model = Sequential()

        self.model.add(Conv2D(32, (3, 3), input_shape=(28, 28, 1)))
        self.model.add(BatchNormalization(axis=-1))
        self.model.add(Activation('relu'))
        self.model.add(Conv2D(32, (3, 3)))
        self.model.add(BatchNormalization(axis=-1))
        self.model.add(Activation('relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))

        for i in range(0, n_conv):
            self.model.add(Conv2D(64, (3, 3)))
            self.model.add(BatchNormalization(axis=-1))
            self.model.add(Activation('relu'))

        if n_conv > 0:
            self.model.add(MaxPooling2D(pool_size=(2, 2)))

        self.model.add(Flatten())

        # Fully connected layer
        self.model.add(Dense(512))
        self.model.add(BatchNormalization())
        self.model.add(Activation('relu'))
        self.model.add(Dropout(0.2))
        self.model.add(Dense(10))

        self.model.add(Activation('softmax'))

        self.model.compile(loss='categorical_crossentropy', optimizer=Adam(), metrics=['accuracy'])

        self.gen = ImageDataGenerator(rotation_range=8, width_shift_range=0.08, shear_range=0.3,
                                      height_shift_range=0.08, zoom_range=0.08)

        self.test_gen = ImageDataGenerator()

        self.train_generator = self.gen.flow(self.X_train, self.Y_train, batch_size=64)
        self.test_generator = self.test_gen.flow(self.X_test, self.Y_test, batch_size=64)

    def fit_test(self, epochs, steps_per_epoch):
        return self.model.fit_generator(self.train_generator, steps_per_epoch=steps_per_epoch // 64, epochs=epochs,
                                        validation_data=self.test_generator, validation_steps=10000 // 64, verbose=0)


class Env1Class(gym.Env):
    def __init__(self):
        self.action_space = spaces.Discrete(6)
        self.mnist = MNIST_CNN()
        self.number_of_test = 0
        self.max = 100
        self.conv2d = 0
        self.epochs = 1
        self.steps_per_epoch = range(64, 64000, 64)
        self.steps_i = 50

        self.last_conv2d = None
        self.last_epochs = None
        self.last_steps_i = None
        
        self._seed()

    def recover_last_parameters(self):
        self.conv2d = self.last_conv2d
        self.epochs = self.last_epochs
        self.steps_i = self.last_steps_i

    def _step(self, action):

        self.last_conv2d = self.conv2d
        self.last_epochs = self.epochs
        self.last_steps_i = self.steps_i

        if action == 0:
            self.epochs -= 1
            if self.epochs < 1:
                self.epochs = 1
        elif action == 1:
            self.epochs += 1

        if action == 2:
            self.conv2d -= 1
            if self.conv2d < 0:
                self.conv2d = 0
        elif action == 3:
            self.conv2d += 1

        if action == 4:
            self.steps_i -= 1
            if self.steps_i < 0:
                self.steps_i = 0
        elif action == 5:
            self.steps_i += 1

        self.mnist.make_model(self.conv2d)
        test = self.mnist.fit_test(self.epochs, self.steps_per_epoch[self.steps_i])
        done = self.number_of_test > self.max
        self.number_of_test += 1
        reward = test.history['val_acc']
        return (self.conv2d, self.epochs, self.steps_per_epoch[self.steps_i]), reward[-1], done

    def _reset(self):
        self.mnist = MNIST_CNN()
        self.conv2d = int(self.np_random.uniform(0, 5))
        self.epochs = int(self.np_random.uniform(1, 10))
        self.steps_i = 50
        return self.conv2d, self.epochs, self.steps_per_epoch[self.steps_i]

    def _render(self, mode='human', close=False):
        pass

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]
