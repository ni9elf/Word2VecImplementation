import os
#os.environ['KERAS_BACKEND']='theano'
from keras.models import load_model
from keras.callbacks import EarlyStopping
from skip_gram import train_word_vectors
import simplejson
import numpy as np

VOCAB_SIZE = 2000
TRAINING_FILES = 100
BATCH_SIZE = 256
EPOCHS = 2000
DIMENSIONS = 50
#after how many training samples to save model
SAVE_MODEL = 10000

samples = 0
count = 0
w2v = train_word_vectors(VOCAB_SIZE, BATCH_SIZE, EPOCHS, DIMENSIONS)    
for i in range(TRAINING_FILES):    
    f = open('Samples/samples_{}'.format(i), 'r')    
    data = simplejson.load(f)    
    x_train = np.asarray(data[0])
    y_train = np.asarray(data[1])
    #print x_train.shape
    #print y_train.shape
    del data[:]
    count += len(x_train)
    if(count >= SAVE_MODEL):
        print 'Saved model after {} samples'.format(count)
        w2v.save('w2v.h5') 
        count = 0
    cbk = EarlyStopping(monitor='val_loss', min_delta=0, patience=3, verbose=0, mode='min')
    w2v.fit(x_train, y_train, batch_size=BATCH_SIZE, epochs=EPOCHS, validation_split=0.2, callbacks=[cbk], verbose=1) 
    samples += len(x_train)    
    print "Trained on {} samples".format(samples)                

print 'Saved model after {} samples'.format(samples)
w2v.save('w2v.h5')      
'''  
weights = w2v.get_weights()
word_vectors = weights[0]
print weights[0]
print len(weights[0])
print len(weights[0][0])
'''
