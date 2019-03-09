from keras.preprocessing.image import ImageDataGenerator
from matplotlib import pyplot
from keras import backend as K
import numpy as np


class ImageShifter():
    def __init__(self, images, lables, net_downsampling_factor, fill_mode='nearest'):
        """
        Args:
          images: implied format is according to K.set_image_dim_ordering('th') : [samples][pixels][width][height]
          lables: implied format is : [samples][lables]
          net_downsampling_factor: determined by strides on CNN
          fill_mode: One of {"constant", "nearest", "reflect" or "wrap"}. Default is 'nearest'. Points outside the boundaries of the input are filled according to the given mode:
          #    'constant': kkkkkkkk|abcd|kkkkkkkk (cval=k)
          #    'nearest': aaaaaaaa|abcd|dddddddd
          #    'reflect': abcddcba|abcd|dcbaabcd
          #    'wrap': abcdabcd|abcd|abcdabcd


        """
        self.images = images
        self.labels = lables
        self.fill_mode = fill_mode
        self.shift_width = (net_downsampling_factor - 0.5) / images.shape[2]
        self.shift_height = (net_downsampling_factor - 0.5) / images.shape[3]
        self.datagen = ImageDataGenerator(width_shift_range=self.shift_width, height_shift_range=self.shift_height,
                                          fill_mode=self.fill_mode)
        self.datagen.fit(self.images)

    def getBatch(self, batch_size):
        """
            Args:
              batch_size - number of  samples to return
              retuns batch of images, shifted in a random fashion according to net_downsampling_factor

            """
        for x_batch, y_batch in self.datagen.flow(self.images, self.labels, batch_size=batch_size):

            return x_batch, y_batch

def test_image_shifter():
    from keras.datasets import mnist
    K.set_image_dim_ordering('th')
    # load data
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    # reshape to be [samples][pixels][width][height]
    X_train = X_train.reshape(X_train.shape[0], 1, 28, 28)
    X_test = X_test.reshape(X_test.shape[0], 1, 28, 28)
    # convert from int to float
    X_train = X_train.astype('float32')
    X_train = X_train[0:100]
    y_train = y_train[0:100]
    X_test = X_test.astype('float32')
    net_downsampling_factor = 4
    imageShifter = ImageShifter(X_train,y_train,net_downsampling_factor)
    X_batch, y_batch = imageShifter.getBatch(9)

    for i in range(0, 9):
        pyplot.subplot(330 + 1 + i)
        pyplot.imshow(X_batch[i].reshape(28, 28), cmap=pyplot.get_cmap('gray'))

    pyplot.show()

if __name__ == '__main__':
  test_image_shifter()