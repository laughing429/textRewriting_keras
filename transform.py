#!/usr/bin/env python 
# -*- coding: utf-8 -*- 
# Create on: 2018-07-20
# Author: Lyu 
# Annotation: 转换这里有两种search方法，一种是greedy search，一种是beam search。

# todo: 1.replace_unk 2.beam_search

import os
import pickle
from keras.models import Model
import jieba
import numpy as np
import heapq


class beam(object):
    def __init__(self, beam_size, init_beam=None):
        if not init_beam:
            self.heap = list()
        else:
            self.heap = init_beam
            heapq.heapify(self.heap)
        self.beam_size = beam_size

    def add(self, prob, complete, prefix):
        heapq.heappush(self.heap, (prob, complete, prefix))
        if len(self.heap) > self.beam_size:
            heapq.heappop(self.heap) # remove the smallest item

    def __iter__(self):
        return iter(self.heap)

    def __len__(self):
        return len(self.heap)


class transformer(object):
    def __init__(self, model, text, count, max_sequence):
        self.text = text
        self.count = count
        self.max_sequence = max_sequence

        root = os.path.dirname(__file__)
        with open(os.path.join(root, 'conf', 'encoder_infer_config.model'), 'rb') as f:
            encoder_infer_config = pickle.load(f)
        with open(os.path.join(root, 'conf', 'decoder_infer_config.model'), 'rb') as f:
            decoder_infer_config = pickle.load(f)
        with open(os.path.join(root, 'conf', 'src_word_index.pkl'), 'rb') as f:
            src_word_index = pickle.load(f)
            # src_index_word = {index:word for word, index in src_word_index.items()}
        with open(os.path.join(root, 'conf', 'tgt_index_word.pkl'), 'rb') as f:
            tgt_index_word = pickle.load(f)
            tgt_word_index = {word: index for index, word in tgt_index_word.items()}

        self.encoder_infer = Model.from_config(encoder_infer_config)
        self.decoder_infer = Model.from_config(decoder_infer_config)

        self.encoder_infer.load_weights(os.path.join(root, 'checkpoints', model), by_name=True)
        self.decoder_infer.load_weights(os.path.join(root, 'checkpoints', model), by_name=True)

        query = jieba.lcut(self.text)
        src_unk_index = src_word_index['<UNK>']
        query_seq = [src_word_index.get(w, src_unk_index) for w in query]
        self.states_value = self.encoder_infer.predict(query_seq)
        self.tgt_go_index = tgt_word_index['<GO>']
        self.tgt_eos_index = tgt_word_index['<EOS>']

    def greedy_search(self):
        "when count == 1 ,beam search is equal to greedy search"
        pass

    @property
    def beam_search(self):
        result = []
        pre_beam = beam(self.count)
        pre_beam.add(1.0, False, [self.tgt_go_index])

        while True:
            curr_beam = beam(self.count)
            for pre_prob, complete, prefix in pre_beam:
                if complete:
                    curr_beam.add(pre_prob, True, prefix)
                else:
                    output_tokens, h, c = self.decoder_infer.predict([prefix]+self.states_value)
                    for suffix, next_prob in enumerate(output_tokens[0, -1, :]):
                        if suffix == self.tgt_eos_index:
                            curr_beam.add(pre_prob*next_prob, True, prefix)
                        else:
                            curr_beam.add(pre_prob*next_prob, False, prefix+[suffix])
                    self.states_value = [h, c]

            sorted_beam = sorted(curr_beam)
            best_prob, best_complete, best_sequence = sorted_beam[-1]
            if best_complete == True or len(best_sequence)-1 == self.max_sequence:
                result.append((best_sequence[1:], best_prob))
                sorted_beam.pop()

            if len(result) == 0: # achieve the target
                break

            pre_beam = beam(self.count, curr_beam)
        return result

def transform_chinese(text, model, max_sequence=10):
    root = os.path.dirname(__file__)
    # prepare
    with open(os.path.join(root, 'conf', 'encoder_infer_config.model'), 'rb') as f:
        encoder_infer_config = pickle.load(f)
    with open(os.path.join(root, 'conf', 'decoder_infer_config.model'), 'rb') as f:
        decoder_infer_config = pickle.load(f)
    with open(os.path.join(root, 'conf', 'src_word_index.pkl'), 'rb') as f:
        src_word_index = pickle.load(f)
        # src_index_word = {index:word for word, index in src_word_index.items()}
    with open(os.path.join(root, 'conf', 'tgt_index_word.pkl'), 'rb') as f:
        tgt_index_word = pickle.load(f)
        tgt_word_index = {word:index for index, word in tgt_index_word.items()}

    encoder_infer = Model.from_config(encoder_infer_config)
    decoder_infer = Model.from_config(decoder_infer_config)
    encoder_infer.load_weights(os.path.join(root, 'checkpoints', model), by_name=True)
    decoder_infer.load_weights(os.path.join(root, 'checkpoints', model), by_name=True)

    query = jieba.lcut(text)
    unk_index = src_word_index['<UNK>']
    query_seq = [src_word_index.get(w, unk_index) for w in query]

    states_value = encoder_infer.predict(query_seq)
    target_seq = np.zeros((1,1))
    target_seq[0, 0] = tgt_word_index['<GO>']

    decode_sentence = []
    while True:
        output_tokens, h, c = decoder_infer.predict([target_seq]+states_value)
        sampled_token_index = np.argmax(output_tokens[0, -1, :])
        sample_char = tgt_index_word[sampled_token_index]
        decode_sentence.append(sample_char)

        if sample_char == '<EOS>' or len(decode_sentence) >= max_sequence:
            break

        target_seq = np.zeros((1,1))
        target_seq[0,0] = sampled_token_index

        states_value = [h, c]
    return ' '.join(decode_sentence)


if __name__ == '__main__':
    res = transformer('zh2zh_weight_667-0.42.hdf5', '傻瓜脑子里总想着偷魔法 。', 2, 10)
    print(res.beam_search)
