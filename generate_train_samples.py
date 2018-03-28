import os
#os.environ['KERAS_BACKEND']='theano'
from keras.preprocessing.text import Tokenizer, one_hot, text_to_word_sequence
from random import randint, choice
import simplejson #to write list to file
from nltk import tokenize

VOCAB_SIZE = 2000
DIMENSIONS = 50
#give size of left=right side of context window
#actual context window size is CONTEXT_WINDOW*2
CONTEXT_WINDOW = 5
#percentage of words from context window to sample
#1 would mean use all words in context window
RANDOM_SAMPLES = 0.5
#after how many samples to save into a file
SAMPLES_SAVE = 10000
MAX_SAMPLES = 1000000


print '>Loading training corpus'
with open("text8.txt", "r") as f:
    text = f.read()

#TODO: is the text_to_word_sequence step required or is the data already pre processed
filter_string = '!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n'
#TODO: tokenization cleanup
text = text_to_word_sequence(text, filters=filter_string, lower=True, split=" ")
#text = tokenize.sent_tokenize(text)
#print text
print 'No of words in corpus: {}'.format(len(text))
tokenizer = Tokenizer(num_words=VOCAB_SIZE, filters=filter_string, lower=True, split=" ", char_level=False)
tokenizer.fit_on_texts(text)
word_to_int = {}
temp_word_to_int = tokenizer.word_index
for k, v in temp_word_to_int.iteritems():    
    if(v > 0 and v <= VOCAB_SIZE):
        word_to_int[k] = v
        
#for k, v in word_to_int.iteritems(): 
#    print k,v
int_to_word = dict(zip(word_to_int.values(), word_to_int.keys()))
f = open('int_to_word_dict.txt', 'w')
simplejson.dump(int_to_word, f)
print 'No of unique words: {}'.format(len(word_to_int))
print '>Training corpus loaded'
#print word_to_int['ill']


print '>Generating training samples'
no_words = len(text)
x_train = []
y_train = []

samples = 0
count = 0
for i in range(no_words):
    if(samples > MAX_SAMPLES):
        break
    if(text[i] not in word_to_int):
        continue
    if(len(x_train) >= SAMPLES_SAVE): 
        samples += len(x_train)   
        print 'Total training samples saved: {}'.format(samples)
        f = open('Samples/samples_{}'.format(count), 'w')
        x_temp = []
        y_temp = []
        for x, y in zip(x_train, y_train):
            #print x
            x1 = [0 for _ in range(x-1)] + [1] + [0 for _ in range(VOCAB_SIZE - x)]
            y1 = [0 for _ in range(y-1)] + [1] + [0 for _ in range(VOCAB_SIZE - y)]
            x_temp.append(x1)
            y_temp.append(y1)
        simplejson.dump([x_temp, y_temp], f)
        f.close()
        count += 1 
        del x_train[:]
        del y_train[:]
        del x_temp[:]
        del y_temp[:]
    indices = [-x for x in range(1, CONTEXT_WINDOW+1)] + range(1, CONTEXT_WINDOW+1)
    for j in range(int(round(RANDOM_SAMPLES*CONTEXT_WINDOW*2, 0))):
        if((i >= CONTEXT_WINDOW) and (i+CONTEXT_WINDOW < no_words)):
            #negative numbers mean choose from left side context window
            #positive numbers mean choose from right side context window            
            index = choice(indices)
            #print indices
            #-1 for backward context window and +1 for forward context window
            output_word = i + index      
            #print output_word              
            if(text[output_word] in word_to_int):
                x_train.append(word_to_int[text[i]])
                y_train.append(word_to_int[text[output_word]])
                #remove samples word for next sampling
                indices.remove(index)
        else:
            #ignore input samples at the beginning and end of corpus whose context window is less than CONTEXT_WINDOW*2
            continue
#saving leftover samples to file
if(len(x_train) > 0): 
    samples += len(x_train)  
    print 'Total training samples saved: {}'.format(samples) 
    f = open('Samples/samples_{}'.format(count), 'w')
    x_temp = []
    y_temp = []
    for x, y in zip(x_train, y_train):
        #print x
        x1 = [0 for _ in range(x-1)] + [1] + [0 for _ in range(VOCAB_SIZE - x)]
        y1 = [0 for _ in range(y-1)] + [1] + [0 for _ in range(VOCAB_SIZE - y)]
        x_temp.append(x1)
        y_temp.append(y1)
    simplejson.dump([x_temp, y_temp], f)
    f.close()
    count += 1 
    del x_train[:]
    del y_train[:]
    del x_temp[:]
    del y_temp[:]
    
print 'No of training samples generated: {}'.format(samples)
print 'No of training files saved: {}'.format(count)
#print len(y_train)
#print zip(x_train, y_train)
#for k, v in word_to_int.iteritems():
#    print k, v
#print x_train[0]

#temp = one_hot(text, VOCAB_SIZE, filters=base_filter(), lower=True, split=" ")
#print temp
