from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
import os.path

import numpy as np
from optparse import OptionParser
from dataLoader import DataLoader

from keras.utils import to_categorical
from keras.layers import Dense
from keras.layers import Conv2D, MaxPooling2D, Flatten
from keras.models import Sequential
from keras.callbacks import ReduceLROnPlateau, ModelCheckpoint, TensorBoard
from keras.optimizers import SGD
from keras.preprocessing.image import ImageDataGenerator

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"


def shuffle_in_unison_inplace(a, b):
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    return a[p], b[p]

def train():
    parser = OptionParser()
    parser.add_option("--train_good",
                      dest="train_good",
                      help="Input good particles ",
                      metavar="FILE")
    parser.add_option("--train_bad",
                      dest="train_bad",
                      help="Input bad particles",
                      metavar="FILE")
    parser.add_option("--particle_number",
                      type="int",
                      dest="train_number",
                      help="Number of positive samples to train.",
                      metavar="VALUE",
                      default=-1)
    parser.add_option("--bin_size",
                      type="int",
                      dest="bin_size",
                      help="image size reduction",
                      metavar="VALUE",
                      default=3)

    parser.add_option("--coordinate_symbol",
                    dest="coordinate_symbol",
                    help="The symbol of the coordinate file, like '_manualPick'",
                    metavar="STRING")
    parser.add_option("--particle_size",
                      type="int",
                      dest="particle_size",
                      help="the size of the particle.",
                      metavar="VALUE",
                      default=-1)
    parser.add_option("--validation_ratio",
                      type="float",
                      dest="validation_ratio",
                      help="the ratio.",
                      metavar="VALUE",
                      default=0.1)
    parser.add_option("--model_retrain",
                    action="store_true",
                    dest="model_retrain",
                    help="train the model using the pre-trained model as parameters initialization .",
                    default=False)
    parser.add_option("--model_load_file",
                      dest="model_load_file",
                      help="pre-trained model",
                      metavar="FILE")
    parser.add_option("--logdir",
                      dest="logdir",
                      help="directory of logfiles",
                      metavar="DIRECTORY",
                      default="Logfile")
    parser.add_option("--model_save_file",
                      dest="model_save_file",
                      help="save the model to file",
                      metavar="FILE")
    (opt, args) = parser.parse_args()

    np.random.seed(1234)

    # define the input size of the model
    model_input_size = [100, 64, 64, 1]
    num_classes = 2                   # the number of output classes
    batch_size = model_input_size[0]

    if not os.access(opt.logdir, os.F_OK):
        os.mkdir(opt.logdir)


    # load training dataset
    dataLoader = DataLoader()
    train_data, train_label, eval_data, eval_label = dataLoader.load_trainData_From_RelionStarFile(
        opt.train_good, opt.particle_size, model_input_size,
        opt.validation_ratio, opt.train_number, opt.bin_size)

    # Check if train_data exist
    try:
        train_data
    except NameError:
        print("ERROR: in function load.loadInputTrainData.")
        return None
    else:
        print("Load training data successfully!")
    # shuffle training data
    train_data, train_label = shuffle_in_unison_inplace(train_data, train_label)
    eval_data, eval_label = shuffle_in_unison_inplace(eval_data, eval_label)

    train_x = train_data.reshape(train_data.shape[0], 64, 64, 1)
    test_x = eval_data.reshape(eval_data.shape[0], 64, 64, 1)
    print("shape of training data: ", train_x.shape, test_x.shape)
    train_y = to_categorical(train_label, 2)
    test_y = to_categorical(eval_label, 2)
    print(train_y.shape, test_y.shape)
    datagen = ImageDataGenerator(
     featurewise_center=True,
     featurewise_std_normalization=True,
     rotation_range=20,
     width_shift_range=0.0,
     height_shift_range=0.0,
     horizontal_flip=True,
     vertical_flip=True)
    datagen.fit(train_x)

    model = Sequential()
    model.add(
        Conv2D(32,
               kernel_size=(8, 8),
               strides=(1, 1),
               activation='relu',
               input_shape=(64, 64, 1)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(64, kernel_size=(8, 8), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(1024, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))
    for layer in model.layers:
        print(layer.name, layer.output_shape)

    logdir = opt.logdir+'/'+datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = TensorBoard(log_dir=logdir)
    checkpoint = ModelCheckpoint('best_model.h5',
                                 monitor='val_acc',
                                 verbose=1,
                                 save_best_only=True,
                                 period=1)
    reduce_lr_plateau = ReduceLROnPlateau(monitor='val_acc',
                                          patience=10,
                                          verbose=1)
    callbacks = [checkpoint, reduce_lr_plateau, tensorboard_callback]
    model.compile(optimizer=SGD(0.01),
                  loss="binary_crossentropy",
                  metrics=["accuracy"])

    model.fit_generator(datagen.flow(train_x, train_y, batch_size=batch_size),
                        steps_per_epoch=len(train_x) / 32,
                        epochs=30,
                        validation_data=(test_x, test_y),
                        callbacks=callbacks)
    model.save(opt.model_save_file)
    accuracy = model.evaluate(x=test_x, y=test_y, batch_size=batch_size)
    print("Accuracy:", accuracy[1])


if __name__ == '__main__':
    train()
