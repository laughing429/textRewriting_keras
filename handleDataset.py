#!/usr/bin/env python 
# -*- coding: utf-8 -*- 
# Create on: 2018-07-10
# Author: Lyu 
# Annotation:处理数据集

import codecs
from collections import Counter
import os
import numpy as np
from keras.utils import Sequence

class dataBase(object):
    """
    对数据做基本操作
    """
    pad = '<PAD>'
    unk = '<UNK>'
    eos = '<EOS>'
    go = '<GO>'
    special_str = [pad, unk, eos, go]

    def __init__(self, filename):
        root = os.path.dirname(__file__)
        file = os.path.join(root, 'data', filename)
        if not os.path.exists(file):
            ValueError('the data file is not exist.')

        self.corpus = []
        print('handle the {}'.format(filename))
        with codecs.open(file, 'r', 'utf8') as f:
            while True:
                line = f.readline()
                if not line:
                    break
                self.corpus.append(line)

        total = len(self.corpus)
        print("there are %d corpus in %s"%(total, filename))

        cnt = Counter()
        for line in self.corpus:
            cnt.update(line.strip().split())
        words = [w for w, c in cnt.most_common() if c >= 1]  # 把生僻词删除掉
        print("%s have %d words."%(filename, len(words)))
        self.word_index = {w: i for i, w in enumerate(words + self.special_str)}
        self.index_word = {i: w for i, w in enumerate(words + self.special_str)}

class transformSequence(Sequence):
    def __init__(self, source, target, batch_size, sequence_len, src_word_index, tgt_word_index):
        self.source = source
        self.target = target
        self.batch_size = batch_size
        self.sequence_len = sequence_len
        self.pad = '<PAD>'
        self.unk = '<UNK>'
        self.eos = '<EOS>'
        self.go = '<GO>'
        self.src_word_index = src_word_index
        self.tgt_word_index = tgt_word_index

    def __len__(self):
        return int(np.ceil(len(self.source)/1.0/self.batch_size))

    def __getitem__(self, idx):
        batch_src = self.source[idx*self.batch_size:(idx+1)*self.batch_size]
        batch_tgt = self.target[idx*self.batch_size:(idx+1)*self.batch_size]
        n = len(batch_src)

        # encoder input
        encoder_source_sequence = np.zeros((n, self.sequence_len), dtype=np.int32)
        for i, line in enumerate(batch_src):
            sentence = [w for w in line.strip().split()]
            if len(sentence) > self.sequence_len:
                sentence = sentence[:self.sequence_len]
            else:
                sentence.extend([self.pad] * (self.sequence_len - len(sentence)))
            encoder_source_sequence[i, :] = [self.src_word_index.get(w, self.src_word_index[self.unk]) for w in sentence]

        # decoder input and decoder_target
        decoder_input = np.zeros((n, self.sequence_len), dtype=np.int32)
        decoder_target_sequence = np.zeros((n, self.sequence_len, len(self.tgt_word_index)), dtype=np.int32)
        for i, line in enumerate(batch_tgt):
            sentence = [w for w in line.strip().split()]
            sentence.extend(self.eos)
            if len(sentence) > self.sequence_len:
                sentence = sentence[:self.sequence_len]
            else:
                sentence.extend([self.pad] * (self.sequence_len - len(sentence)))

            for j, word in enumerate(sentence):
                decoder_input[i, j] = self.tgt_word_index.get(word, self.tgt_word_index[self.unk])
                decoder_target_sequence[i, j, self.tgt_word_index.get(word, self.tgt_word_index[self.unk])] = 1

        # decoder_input first word change go_index
        decoder_input = np.insert(decoder_input, 0, self.tgt_word_index[self.go], axis=1)[:, :-1]

        return [encoder_source_sequence, decoder_input], decoder_target_sequence