#!/usr/bin/env python
# coding: utf-8
"""© 2019 Rajkumar Pujari All Rights Reserved

- Original Version

    Author: Rajkumar Pujari
    Last Modified: 03/12/2019

"""
import numpy as np
from classifier import BinaryClassifier
from utils import transform_data, get_feature_vectors, vocab
import math

class NaiveBayes(BinaryClassifier):
    
    def __init__(self, args):
        #TO DO: Initialize parameters here
#        print(args)
        self.lr = args.lr
        self.f_dim = args.f_dim
        self.num_iter = args.num_iter
        self.vocab_size = args.vocab_size
        self.bin_feats = args.bin_feats
    
        
    def fit(self, train_data):
        #TO DO: Learn the parameters from the training data
        pos = [0] * self.f_dim
        neg = [0] * self.f_dim
        pos_count = 0
        neg_count = 0
        x_vectors = get_feature_vectors(train_data[0], self.bin_feats)
        cminus = 0.0
        cplus = 0.0
        for i in train_data[1]:
            if i == -1:
                cminus += 1.0
            else:
                cplus += 1.0
#        print(len(train_data[1]))
#        print(cplus)
        self.cplus = cplus / len(train_data[1])
        self.cminus = cminus / len(train_data[1])
        
        j = 0
        for vector in x_vectors:
            if train_data[1][j] == 1:
#                pos = np.array(pos) + np.array(vector)
                pos = [x+y for x,y in zip(pos,vector)]
                pos_count += len(train_data[0][j].split())
#                print('positive')
            else:
#                neg = np.array(neg) + np.array(vector)
                neg = [x+y for x,y in zip(neg,vector)]
                neg_count += len(train_data[0][j].split())
#                print('negative')
            j += 1
#            print(pos)
#            print(neg)
        j = 0
        for num in pos:
            pos[j] = num / cplus
            j += 1
        j = 0
        for num in neg:
            neg[j] = num / cminus
            j += 1
        self.positive = pos
        self.negative = neg
#        print(pos)
#        print(neg)
#        self.cplus = pos_count
#        self.cminus = neg_count
#        print(vocab)
#        print(pos)
#        print(neg)

#        x_vectors = get_feature_vectors(train_data[0], self.bin_feats)
#        print(x_vectors)

        
#        print(len(train_data[0][0].strip().split()))
#        print(train_data[0][0])
#        print(transform_data(train_data[0][0]))
#        raise NotImplementedError

    def predict(self, test_x):
        #TO DO: Compute and return the output for the given test inputs
        ret_vector = []
        """ # ALternative method that was producing lower values
        x_vectors = get_feature_vectors(test_x, self.bin_feats)
#        print(len(x_vectors))
        for x in x_vectors:
            i=0
            countp = 0.0
            countn = 0.0
            for c in x:
#                print(self.positive[i])
#                print(x[i])
                if self.positive[i] > 0:
#                    print(countp)
#                    countp = countp * math.pow(self.positive[i], x[i])
#                    if self.positive[i] > 1:
#                        print('positive: ', self.positive[i])
#                    if math.pow(np.log(self.positive[i]),x[i]) < 0:
                    countp += math.pow(np.log(self.positive[i]),x[i])
#                    if countp > 0:
#                        print(countp)
                if self.negative[i] > 0:
#                    countn = countn * math.pow(self.negative[i], x[i])
#                    if self.negative[i] > 1:
#                        print('negative: ' ,self.negative[i])
#                    if math.pow(np.log(self.negative[i]),x[i]) < 0:
                    countn += math.pow(np.log(self.negative[i]),x[i])
#                    if countn > 0:
#                        print(countn)
                i+=1
#            print(countp)
#            print(countn)
            if countp > 0 or countn > 0:
                print(x_vectors.index(x))
            countp = (countp * self.cplus)
            countn = (countn * self.cminus)
            if abs(countp) < abs(countn):
                ret_vector.append(1)
#                print('positive wins!')
#                print("countp: ", countp)
#                print("countn: ", countn)
            else:
                ret_vector.append(-1)
#                print('negative wins!')
#                print("countp: ", countp)
#                print("countn: ", countn)

        """
#        print(test_x[0])
        for review in test_x:
            posproduct = 0.0#Decimal(1.0)
            negproduct = 0.0#Decimal(1.0)
#            print(len(review.split()))
            for word in review.split():
                if word in vocab:
                    index = list(vocab).index(word)
#                    print(self.positive[index])
                    if self.positive[index] > 0.0:
#                        print('pos: ' + str(posproduct))
                        posproduct = posproduct + math.log10(self.positive[index])
                    if self.negative[index] > 0.0:
#                        print('neg: ' + str(negproduct))
                        negproduct = negproduct + math.log10(self.negative[index])
#            print('pos: ' + str(posproduct))
#            print('neg: ' + str(negproduct))

            posproduct = (posproduct * self.cplus)
            negproduct = (negproduct * self.cminus)
#            print('pos: ' + str(posproduct))
#            print('neg: ' + str(negproduct))
            if posproduct > negproduct:
                ret_vector.append(1)
#                print('pos: ' + str(posproduct))
            else:
                ret_vector.append(-1)
#                print('neg: ' + str(negproduct))
#        print(self.cplus)
#        print(self.cminus)

    
        return ret_vector
