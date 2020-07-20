import tensorflow as tf
import sys
from tensorflow.keras import layers
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Input, Add, Conv2D, MaxPooling2D, GlobalAveragePooling2D, \
    Dropout, Dense, Flatten, BatchNormalization, Activation, ZeroPadding2D
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.optimizers import SGD
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

def identity_block(M, filters, kernel_size, stage, block, weight_decay):
    #define model name
    conv_name = 'conv_s' + str(stage) + '_b' + str(block)
    bn_name = 'batchN_s' + str(stage) + '_b' + str(block)

    #create shortcut placeholder
    M_sc = M

    M = Conv2D(filters, (kernel_size, kernel_size), activation='relu', kernel_initializer='he_uniform', \
            kernel_regularizer=l2(weight_decay), padding='same', name=conv_name+'a')(M)
    M = BatchNormalization(axis=3, name=bn_name+'a')(M)
    
    M = Conv2D(filters, (kernel_size, kernel_size), kernel_initializer='he_uniform', \
            kernel_regularizer=l2(weight_decay), padding='same', name=conv_name+'b')(M)
    M = BatchNormalization(axis=3, name=bn_name+'b')(M)

    M = Add()([M, M_sc])
    M = Activation('relu')(M)

    return M

def conv_block(M, filters, kernel_size, stage, block, weight_decay):
    #define model name
    conv_name = 'conv_s' + str(stage) + '_b' + str(block)
    bn_name = 'batchN_s' + str(stage) + '_b' + str(block)

    #create shortcut placeholder
    M_sc = M

    #Main path
    M = Conv2D(filters, (kernel_size, kernel_size), activation='relu', kernel_initializer='he_uniform', \
            kernel_regularizer=l2(weight_decay), padding='valid', name=conv_name+'a')(M)
    M = BatchNormalization(axis=3, name=bn_name+'a')(M)
    
    M = Conv2D(filters, (kernel_size, kernel_size), kernel_initializer='he_uniform', \
            kernel_regularizer=l2(weight_decay), padding='same', name=conv_name+'b', strides=(2,2))(M)
    M = BatchNormalization(axis=3, name=bn_name+'b')(M)

    #Shortcut path
    M_sc = Conv2D(filters, (kernel_size, kernel_size), kernel_initializer='he_uniform', \
            kernel_regularizer=l2(weight_decay), padding='valid', name=conv_name+'c', strides=(2,2))(M_sc)
    M_sc = BatchNormalization(axis=3, name=bn_name+'c')(M_sc)

    M = Add()([M, M_sc])
    M = Activation('relu')(M)

    return M


def ResNet20(num_classifiers, input_shape, padding):
    '''
    Good starting loss: -ln(0.1) = 2.3026
    '''
    #Definitions
    M_shape = Input(input_shape)
    weight_decay = 1e-4

    #Zero Padding
    M = ZeroPadding2D(padding=1)(M_shape)

    #Stage 3 2D-Convolution
    M = conv_block(M, 32, 3, 3, 1, weight_decay)
    M = identity_block(M, 32, 3, 3, 2, weight_decay)
    M = identity_block(M, 32, 3, 3, 3, weight_decay)
    M = ZeroPadding2D(padding=1)(M)


    #Stage 4 2D-Convolution
    M = conv_block(M, 64, 3, 4, 1, weight_decay)
    M = identity_block(M, 64, 3, 4, 2, weight_decay)
    M = identity_block(M, 64, 3, 4, 3, weight_decay)
    
    
    #AVGPOOL Layer
    M = GlobalAveragePooling2D(name="globalavg_pool")(M)

    #FULLY CONNECTED LAYER
    M = Flatten()(M)
    #Output layer
    M = Dense(num_classifiers, activation='softmax')(M)


	#Create and Compile model
    model = Model(inputs=M_shape, outputs=M, name='ResNet13')
    opt = SGD(momentum=0.9)
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
    model.summary()
    return model

def lr_schedule(epoch):
    '''
    Returns a custom learning rate that decreases as epochs progress.
    '''
    learning_rate = 0.1
    if epoch > 80:
        learning_rate = 0.01
    if epoch > 125:
        learning_rate = 0.001

    tf.summary.scalar('learning rate', data=learning_rate, step=epoch)
    return learning_rate

def prepare_data(trainX):
    #data augmentation
    datagen = ImageDataGenerator(width_shift_range=0.125, \
        height_shift_range=0.125, horizontal_flip=True)
    datagen.fit(trainX)
    return datagen

def summarize_diagnostics(history):
	# plot loss
	pyplot.subplot(211)
	pyplot.title('Cross Entropy Loss')
	pyplot.plot(history.history['loss'], color='blue', label='train')
	pyplot.plot(history.history['val_loss'], color='orange', label='test')
    # save plot to file
	pyplot.savefig('Loss.png')
	pyplot.close()
    # plot accuracy
	pyplot.subplot(211)
	pyplot.title('Classification Accuracy')
	pyplot.plot(history.history['accuracy'], color='blue', label='train')
	pyplot.plot(history.history['val_accuracy'], color='orange', label='test')
	# save plot to file
	pyplot.savefig('Accuracy.png')
	pyplot.close()

def main():
    trainX, trainY, testX, testY = load_data()
    datagen = prepare_data(trainX)

    lr_callback = LearningRateScheduler(lr_schedule, verbose=1)

    trained_model = ResNet20(10, (32, 32, 3), 0)
    history = trained_model.fit(datagen.flow(trainX, trainY, batch_size=128), epochs=165, validation_data=(testX, testY), verbose=1, callbacks=[lr_callback])
    _, acc = trained_model.evaluate(testX, testY, verbose=0)
    print('\n\n>Final Accuracy of Trained Model: %.3f' % (acc * 100.0))

    # save model
    trained_model.save('cifar10_final.h5')

    summarize_diagnostics(history)

main()