from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Conv2D
from keras.optimizers import RMSprop
from gpopy import FlowTunning 
import keras
import random
import mlflow.keras

#batch_size = 128
num_classes = 10
#epochs = 20

PARAMS = {
    'batch_size' : [256],# [8],#,16,32,64,128,256],
    'epochs' : [8, 16 ],#, 32, 64],
    'dense_layers' : [64, 128],#, 256, 512],
    'dropout' : {
        'func' : random.uniform, 
        'params' : [0.3, 0.7] 
    },
    'activation' : 'relu',
    'learning_rate': { 
       'func' :  random.uniform,
       'params' : [0.01, 0.0001]
    },
    'filters' : [10, 64, 128],
    'use_bias' : [True, False]
}

# the data, split between train and test sets
(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.reshape(60000, 784)
x_test = x_test.reshape(10000, 784)
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

def model (data, x_train = x_train, x_test = x_test, 
                 y_train = y_train, y_test = y_test):
    mlflow.keras.autolog() #This line will be used with gpopy 
    layer = data['dense_layers']
    dropout = data['dropout']
    activation = data['activation']
    batch_size = data['batch_size']
    epochs = data['epochs']
    learning_rate = data['learning_rate']
    filters = data['filters']
    use_bias = data['use_bias']

    model = Sequential()
    model.add(Dense(layer, activation= activation, input_shape=(784,)))
    model.add(Dropout(dropout))
    model.add(Dense(layer, activation=activation))
    model.add(Dropout(dropout))
    model.add(Dense(num_classes, activation='softmax'))

    model.summary()

    model.compile(loss='categorical_crossentropy',
                optimizer=RMSprop(lr = learning_rate),
                metrics=['accuracy'])

    history = model.fit(x_train, y_train,
                        batch_size=batch_size,
                        epochs=epochs,
                        verbose=1,
                        validation_data=(x_test, y_test))
    score = model.evaluate(x_test, y_test, verbose=0)

    print("#######################- RESULTS -####################################")
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])
    print("######################################################################")
    return (score[1], model)

tunning = FlowTunning(params=PARAMS, population_size=2, auto_track=False)
tunning.set_score(model)
tunning.run()

