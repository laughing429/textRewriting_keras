#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Create on: 2018-07-25
# Author: Lyu
# Annotation: base on a word-level model which rewrite text

from keras.layers import Input, LSTM, Dense, Embedding
from keras.models import Model

def struct_model(source_vocab_size, target_vocab_size, embeding_size, latent_dim):
    # encoder model
    encoder_inputs = Input(shape=(None,))
    encoder_embed_inputs = Embedding(input_dim=source_vocab_size, output_dim=embeding_size)(encoder_inputs)
    encoder = LSTM(latent_dim, return_state=True)
    _, state_h, state_c = encoder(encoder_embed_inputs)
    encoder_status = [state_h, state_c]

    # define decoder
    decoder_inputs = Input(shape=(None,))
    decoder_embed_inputs = Embedding(input_dim=target_vocab_size, output_dim=embeding_size)(decoder_inputs)
    decoder = LSTM(latent_dim, return_sequences=True, return_state=True)
    x, _, _ = decoder(decoder_embed_inputs, initial_state=encoder_status)
    decoder_dense = Dense(target_vocab_size, activation='softmax')
    decoder_outputs = decoder_dense(x)
    train_model = Model(inputs=[encoder_inputs, decoder_inputs], outputs=decoder_outputs)

    # encoder infer model
    encoder_infer = Model(encoder_inputs, encoder_status)
    # decoder infer model
    decoder_state_input_h = Input(shape=(latent_dim,))
    decoder_state_input_c = Input(shape=(latent_dim,))
    decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
    decoder_outputs, state_h, state_c = decoder(decoder_embed_inputs, initial_state=decoder_states_inputs)
    decoder_states = [state_h, state_c]
    decoder_outputs = decoder_dense(decoder_outputs)
    decoder_infer = Model(
        [decoder_inputs] + decoder_states_inputs,
        [decoder_outputs] + decoder_states
    )

    return train_model, encoder_infer, decoder_infer