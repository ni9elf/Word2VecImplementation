# Word2Vec in Keras

A Keras implementation of [word2vec](https://arxiv.org/abs/1301.3781), specifically the continuous Skip-gram model for computing continuous vector representations of words from very large data sets. The quality of the word vectors is measured in a word similarity task, with word2vec showing a large improvement in accuracy at a much lower computational cost. Further, word2vec performs at state-of-the-art accuracy for measuring  syntactic and semantic word similarities.

## Model architecture


## Prerequisites

## Usage


## Code Organization
To generate training and testing samples for the word2vec model use the `generate_train_samples.py` on a corpus. The model is implemented in Keras in the file `skip_gram.py`. The file `word2vec.py` trains and saves the model. Use `visualize.py` to create a t-SNE plot of the word embeddings.

## References


