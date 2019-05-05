#!/usr/bin/env python
# coding: utf-8
"""© 2019 Rajkumar Pujari All Rights Reserved

- Original Version

    Author: Rajkumar Pujari
    Last Modified: 03/12/2019

"""

from classifier import BinaryClassifier
from perceptron import Perceptron, AveragedPerceptron
from naive_bayes import NaiveBayes
from utils import read_data, build_vocab
import utils
from config import args

if __name__ == '__main__':
    filepath = '../data/given/'
    build_vocab(filepath, vocab_size=args.vocab_size)
    train_data, test_data = read_data(filepath)
    
    nb_classifier = NaiveBayes(args)
    nb_classifier.fit(train_data)
    acc, prec, rec, f1 = nb_classifier.evaluate(test_data)
    print('\nNaive Bayes Performance:')
    print('Accuracy: %.2f, Precision: %.2f, Recall: %.2f, F1: %.2f'%(acc, prec, rec, f1))
