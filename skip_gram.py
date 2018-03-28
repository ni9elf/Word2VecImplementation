import os
#os.environ['KERAS_BACKEND']='theano'
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import to_categorical
from keras import optimizers
from keras.initializers import TruncatedNormal

#one hot encoding
#y_train = to_categorical(y_train, num_classes)

def train_word_vectors(VOCAB_SIZE, BATCH_SIZE, EPOCHS, DIMENSIONS):
    model = Sequential()    
    trun_normal = TruncatedNormal(mean=0.0, stddev=0.005, seed=None)
    model.add(Dense(DIMENSIONS, input_shape=(VOCAB_SIZE,), activation='linear', use_bias=False, kernel_initializer=trun_normal))
    model.add(Dense(VOCAB_SIZE, activation='softmax', use_bias=False))

    sgd = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])    
    return model
