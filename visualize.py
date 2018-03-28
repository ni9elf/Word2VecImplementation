import numpy as Math
import pylab as Plot
from tsne import tsne
import os
#os.environ['KERAS_BACKEND']='theano'
from keras.models import load_model
import simplejson

#glove_words = [line.strip() for line in open("glove.6B.50d.labels.txt")]
#target_words = [line.strip().lower() for line in open("4000-most-common-english-words-csv.csv")][1:200]

#rows = [glove_words.index(word) for word in target_words if word in glove_words]

NUM_WORDS = 2000

w2v = load_model('w2v.h5')
weights = w2v.get_weights()
word_vectors = weights[0]

f = open('int_to_word_dict.txt', 'r')
int_to_word = simplejson.load(f)

w2v_matrix = Math.asarray(word_vectors, dtype='float32')
reduced_matrix = tsne(w2v_matrix[:NUM_WORDS,], 2);
Plot.figure(figsize=(200, 200), dpi=100)
max_x = Math.amax(reduced_matrix, axis=0)[0]
max_y = Math.amax(reduced_matrix, axis=0)[1]
Plot.xlim((-max_x,max_x))
Plot.ylim((-max_y,max_y))
Plot.scatter(reduced_matrix[:, 0], reduced_matrix[:, 1], 20);

for k, v in int_to_word.iteritems(): 
    k = int(k)
    if(k > 0 and k < NUM_WORDS):
        target_word = v 
        x = reduced_matrix[k-1, 0]
        y = reduced_matrix[k-1, 1]
        Plot.annotate(target_word, (x,y))

Plot.savefig("word2vec.pdf", format='pdf');

'''
f = open('glove.6B.50d.txt')
i = 0
matrix = []
i = 0
words = []
for i in range in f:
    values = line.split()
    word = values[0]
    if(word in target_words):        
        words.append(values[0])
    #Glove file contains a word and then list of values which are the coeeficients corresponding to the k-dimensional embedding
        matrix.append(values[1:])
f.close()

glove_matrix = Math.asarray(matrix, dtype='float32')
#target_matrix = glove_matrix[rows,:]
reduced_matrix = tsne(glove_matrix, 2);
Plot.figure(figsize=(50, 50), dpi=100)
max_x = Math.amax(reduced_matrix, axis=0)[0]
max_y = Math.amax(reduced_matrix, axis=0)[1]
Plot.xlim((-max_x,max_x))
Plot.ylim((-max_y,max_y))

Plot.scatter(reduced_matrix[:, 0], reduced_matrix[:, 1], 20);

for i in range(0, len(words)):
    target_word = words[i]
    x = reduced_matrix[i, 0]
    y = reduced_matrix[i, 1]
    Plot.annotate(target_word, (x,y))

Plot.savefig("glove_2000.png");
'''
