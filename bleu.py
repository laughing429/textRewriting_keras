#!/usr/bin/env python 
# -*- coding: utf-8 -*- 
# Create on: 2018-08-20
# Author: Lyu 
# Annotation:


class bleu(object):
    def __init__(self, y_true, y_pred):
        self.y_true = y_true
        self.y_pred = y_pred
        self.sess = K.get_session()
        pass

    def bp(self):
        r_len = K.shape(self.y_true).eval(session=self.sess)
        c_len = K.shape(self.y_pred).eval(session=self.sess)
        if c_len[0] > r_len[0]:
            return 1
        else:
            K.exp()