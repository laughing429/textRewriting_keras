#!/usr/bin/env python 
# -*- coding: utf-8 -*- 
# Create on: 2018-07-26
# Author: Lyu 
# Annotation:

import os
from keras.backend import tensorflow_backend as KTF
from keras.callbacks import  ModelCheckpoint, EarlyStopping, TensorBoard
from keras.models import load_model
import tensorflow as tf
from yaml import load
import pickle

from models import struct_model
from handleDataset import dataBase, transformSequence


def train(srcpath, tgtpath, batch_size, epochs, premodel=None):
    "use teacher forcing for training"
    conf_path = os.path.join(os.path.dirname(__file__), 'conf')
    with open(os.path.join(conf_path, 'config.yml'), 'r') as f:
        model_config = load(f)['model_config']
    sequence_len = model_config['sequence_len']
    latent_dim = model_config['latent_dim']
    embeding_size = model_config['embeding_size']

    # prepare
    src_text = dataBase(srcpath)
    source = src_text.corpus
    src_word_index = src_text.word_index
    src_vocab_size = len(src_word_index)

    tgt_text = dataBase(tgtpath)
    target = tgt_text.corpus
    tgt_word_index = tgt_text.word_index
    tgt_vocab_size = len(tgt_word_index)

    with open(os.path.join(conf_path, 'src_word_index.pkl'), 'wb') as f:
        pickle.dump(src_text.word_index, f)
    with open(os.path.join(conf_path, 'tgt_index_word.pkl'), 'wb') as f:
        pickle.dump(tgt_text.index_word, f)

    if premodel:
        if not os.path.exists(premodel):
            ValueError("the previous model doesn't found.")
        model = load_model(premodel)
    else:
        model, encoder_infer, decoder_infer = struct_model(src_vocab_size, tgt_vocab_size, embeding_size, latent_dim)

        # 模型序列化
        with open(os.path.join(conf_path, 'encoder_infer_config.model'), 'wb') as f:
            pickle.dump(encoder_infer.get_config(), f)

        with open(os.path.join(conf_path, 'decoder_infer_config.model'), 'wb') as f:
            pickle.dump(decoder_infer.get_config(), f)

    model_checkpoints_dir = os.path.join(os.path.dirname(__file__), 'checkpoints')
    # log_dir = os.path.join(os.path.dirname(__file__), 'logs')
    callbacks = [
        ModelCheckpoint(os.path.join(model_checkpoints_dir, 'zh2zh_weight_epoch:{epoch:02d}-loss:{loss:.2f}.hdf5'),
                        monitor='loss',
                        save_weights_only=True,
                        # save_best_only=True,
                        period=epochs//10
                        ),
        # EarlyStopping(monitor='loss', patience=10, mode='min', verbose=1),
        # TensorBoard(log_dir)
    ]

    if tf.test.is_gpu_available:
        config = tf.ConfigProto()
        config.gpu_options.allow_growth=True
        sess = tf.Session(config=config)
        KTF.set_session(sess)

    model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['acc'])
    # model.fit(
    #     x=[source, decoder_input_data],
    #     y=target,
    #     batch_size=batch_size,
    #     epochs=epochs,
    #     callbacks=callbacks,
    #     validation_split=validation)
    model.fit_generator(transformSequence(source, target, batch_size, sequence_len, src_word_index, tgt_word_index),
                        epochs=epochs,
                        callbacks=callbacks)


if __name__ == '__main__':
    # import argparse
    #
    # parse = argparse.ArgumentParser()
    # parse.add_argument('-srcpath', required=True)
    # parse.add_argument('-tgtpath', required=True)
    # parse.add_argument('-batch_size', type=int, required=True)
    # parse.add_argument('-epochs', type=int, required=True)
    # parse.add_argument('-validation', type=float)
    # parse.add_argument('-premodel')
    # opts = parse.parse_args()

    # train(opts.srcpath, opts.tgtpath, opts.batch_size, opts.epochs, opts.validation, opts.premodel)

    train('zh2zh/train_src.zh_cut', 'zh2zh/train_tgt.zh_cut', 64, 100000)
