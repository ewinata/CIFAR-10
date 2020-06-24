import tensorflow as tf
import sys
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, Dense, Flatten, BatchNormalization, Activation
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.optimizers import SGD
from sklearn.model_selection import KFold
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import LearningRateScheduler
from tensorflow.keras.regularizers import l2
from matplotlib import pyplot

def load_data():
    (trainX, trainY), (testX, testY) = cifar10.load_data()
    trainX = trainX.astype('float32') / 255.0
    testX = testX.astype('float32') / 255.0
    trainY = to_categorical(trainY)
    testY = to_categorical(testY)
    return trainX, trainY, testX, testY

def define_model(length_label):
    model = Sequential()
    weight_decay = 1e-5

    #Stage 1 2D-Convolution
    model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', \
                kernel_regularizer=l2(weight_decay), input_shape=(32, 32, 3), padding="same"))
    model.add(BatchNormalization())
    model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', \
                kernel_regularizer=l2(weight_decay)))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size = (2, 2)))
    model.add(Dropout(0.2))
    
    #Stage 2 2D-Convolution
    model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', \
                kernel_regularizer=l2(weight_decay), padding="same"))
    model.add(BatchNormalization())
    model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', \
                kernel_regularizer=l2(weight_decay), padding="same"))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size = (2, 2)))
    model.add(Dropout(0.25))

    #Stage 3 2D-Convolution
    model.add(Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform', \
                kernel_regularizer=l2(weight_decay), padding="same"))
    model.add(BatchNormalization())
    model.add(Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform', \
                kernel_regularizer=l2(weight_decay), padding="same"))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))

    #Stage 1 Dense layer
    model.add(Flatten())
    model.add(Dense(1024, kernel_initializer='he_uniform'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.5))

    # output layer
    model.add(Dense(length_label, activation='softmax'))

	# compile model
    opt = SGD(lr=0.01, momentum=0.9, nesterov=True)
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'], callbacks=[LearningRateScheduler(get_lr)])
    model.summary()
    return model

def get_lr(num_epoch):
    lr = 0.01
    if num_epoch > 20:
        lr = 0.005
    elif num_epoch > 30:
        lr = 0.001
    elif num_epoch > 100:
        lr = 0.0005
    elif num_epoch > 150:
        lr = 0.0001
    return lr

def prepare_data(trainX):
    #data augmentation
    datagen = ImageDataGenerator(width_shift_range=0.1, \
        rotation_range=45, height_shift_range=0.1, horizontal_flip=True)
    datagen.fit(trainX)
    return trainX

def summarize_diagnostics(history):
	# plot loss
	pyplot.subplot(211)
	pyplot.title('Cross Entropy Loss')
	pyplot.plot(history.history['loss'], color='blue', label='train')
	pyplot.plot(history.history['val_loss'], color='orange', label='test')
	# plot accuracy
	pyplot.subplot(212)
	pyplot.title('Classification Accuracy')
	pyplot.plot(history.history['accuracy'], color='blue', label='train')
	pyplot.plot(history.history['val_accuracy'], color='orange', label='test')
	# save plot to file
	filename = sys.argv[0].split('/')[-1]
	pyplot.savefig(filename + 'final_plot.png')
	pyplot.close()

def main():
    trainX, trainY, testX, testY = load_data()
    trainX = prepare_data(trainX)

    trained_model = define_model(10)
    history = trained_model.fit(trainX, trainY, epochs=125, batch_size=32, validation_data=(testX, testY), verbose=2, callbacks=[LearningRateScheduler(get_lr)])
    _, acc = trained_model.evaluate(testX, testY, verbose=0)
    print('> %.3f' % (acc * 100.0))

    summarize_diagnostics(history)

    # save model
    trained_model.save('cifar10_final.h5')

main()