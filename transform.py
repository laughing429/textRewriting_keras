#!/usr/bin/env python 
# -*- coding: utf-8 -*- 
# Create on: 2018-07-20
# Author: Lyu 
# Annotation:

import os
import pickle
from keras.models import Model
import jieba
import numpy as np

def transform_chinese(text, model, max_sequence=10, count=1):
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
        output_tokens, h, c = decoder_infer.predict_on_batch([target_seq]+states_value)
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
    res = transform_chinese('傻瓜脑子里总想着偷魔法 。', 'zh2zh_weight_667-0.42.hdf5')
    print(res)
