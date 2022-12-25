import numpy as np
import random
import os, errno
import sys
from tqdm import trange

import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F


class lstm_encoder(nn.Module):
    ''' Encodes time-series sequence '''

    def __init__(self, input_dim, hidden_size, num_layers = 1):
        
        '''
        : param input_dim:     the number of features in the input X
        : param hidden_size:    the number of features in the hidden state h
        : param num_layers:     number of recurrent layers (i.e., 2 means there are
        :                       2 stacked LSTMs)
        '''
        
        super(lstm_encoder, self).__init__()
        self.input_dim = input_dim
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # define LSTM layer
        self.lstm = nn.LSTM(input_dim = input_dim, hidden_size = hidden_size,
                            num_layers = num_layers)

    def forward(self, x_input):
        
        '''
        : param x_input:               input of shape (seq_len, # in batch, input_dim)
        : return lstm_out, hidden:     lstm_out gives all the hidden states in the sequence;
        :                              hidden gives the hidden state and cell state for the last
        :                              element in the sequence 
        '''
        
        lstm_out, self.hidden = self.lstm(x_input.view(x_input.shape[0], x_input.shape[1], self.input_dim))
        
        return lstm_out, self.hidden     
    
    def init_hidden(self, batch_size):
        
        '''
        initialize hidden state
        : param batch_size:    x_input.shape[1]
        : return:              zeroed hidden state and cell state 
        '''
        
        return (torch.zeros(self.num_layers, batch_size, self.hidden_size),
                torch.zeros(self.num_layers, batch_size, self.hidden_size))


class lstm_decoder(nn.Module):
    ''' Decodes hidden state output by encoder '''
    
    def __init__(self, input_dim, hidden_size, num_layers = 1):

        '''
        : param input_dim:     the number of features in the input X
        : param hidden_size:    the number of features in the hidden state h
        : param num_layers:     number of recurrent layers (i.e., 2 means there are
        :                       2 stacked LSTMs)
        '''
        
        super(lstm_decoder, self).__init__()
        self.input_dim = input_dim
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(input_dim = input_dim, hidden_size = hidden_size,
                            num_layers = num_layers)
        self.linear = nn.Linear(hidden_size, input_dim)           

    def forward(self, x_input, encoder_hidden_states):
        
        '''        
        : param x_input:                    should be 2D (batch_size, input_dim)
        : param encoder_hidden_states:      hidden states
        : return output, hidden:            output gives all the hidden states in the sequence;
        :                                   hidden gives the hidden state and cell state for the last
        :                                   element in the sequence 
 
        '''
        
        lstm_out, self.hidden = self.lstm(x_input.unsqueeze(0), encoder_hidden_states)
        output = self.linear(lstm_out.squeeze(0))     
        
        return output, self.hidden

class LSTM_Seq2Seq(nn.Module):
    ''' train LSTM encoder-decoder and make predictions '''
    
    def __init__(self, input_dim, hidden_size, training_prediction = 'teacher_forcing', teacher_forcing_ratio = 1.0):

        '''
        : param input_dim:     the number of expected features in the input X
        : param hidden_size:    the number of features in the hidden state h
                : param training_prediction:       type of prediction to make during training ('recursive', 'teacher_forcing', or
        :                                  'mixed_teacher_forcing'); default is 'recursive'
        : param teacher_forcing_ratio:     float [0, 1) indicating how much teacher forcing to use when
        :                                  training_prediction = 'teacher_forcing.' For each batch in training, we generate a random
        :                                  number. If the random number is less than teacher_forcing_ratio, we use teacher forcing.
        :                                  Otherwise, we predict recursively. If teacher_forcing_ratio = 1, we train only using
        :                                  teacher forcing.
        '''

        super(LSTM_Seq2Seq, self).__init__()

        self.training_prediction = training_prediction
        self.teacher_forcing_ratio = teacher_forcing_ratio

        self.input_dim = input_dim
        self.hidden_size = hidden_size

        self.prenet_forward_encoder = lstm_encoder(input_dim = input_dim, hidden_size = hidden_size)
        self.prenet_backward_encoder = lstm_encoder(input_dim = input_dim, hidden_size = hidden_size)

        self.forward_decoder = lstm_decoder(input_dim = input_dim, hidden_size = hidden_size)
        self.backward_decoder = lstm_decoder(input_dim = input_dim, hidden_size = hidden_size)


    def forward(self, input_tensor, target_tensor, target_len):
        
        '''
        train lstm encoder-decoder
        
        : param input_tensor:              input data with shape (seq_len, # in batch, number features); PyTorch tensor    
        : param target_tensor:             target data with shape (seq_len, # in batch, number features); PyTorch tensor
        : param target_len:                number of values to predict 
        '''
        batch_size = input_tensor.size(1)

        # outputs tensor
        outputs = torch.zeros(target_len, batch_size, input_tensor.shape[2])

        # initialize hidden state
        encoder_hidden = self.encoder.init_hidden(batch_size)

        # encoder outputs
        encoder_output, encoder_hidden = self.encoder(input_tensor)

        # decoder with teacher forcing
        decoder_input = input_tensor[-1, :, :]   # shape: (batch_size, input_dim)
        decoder_hidden = encoder_hidden

        if self.training_prediction == 'recursive':
            # predict recursively
            for t in range(target_len): 
                decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden)
                outputs[t] = decoder_output
                decoder_input = decoder_output

        if self.training_prediction == 'teacher_forcing':
            # use teacher forcing
            if random.random() < self.teacher_forcing_ratio:
                for t in range(target_len): 
                    decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden)
                    outputs[t] = decoder_output
                    decoder_input = target_tensor[t, :, :]

            # predict recursively 
            else:
                for t in range(target_len): 
                    decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden)
                    outputs[t] = decoder_output
                    decoder_input = decoder_output

        if self.training_prediction == 'mixed_teacher_forcing':
            # predict using mixed teacher forcing
            for t in range(target_len):
                decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden)
                outputs[t] = decoder_output
                
                # predict with teacher forcing
                if random.random() < self.teacher_forcing_ratio:
                    decoder_input = target_tensor[t, :, :]
                
                # predict recursively 
                else:
                    decoder_input = decoder_output

        return outputs

    def predict(self, input_tensor, target_len):
        
        '''
        : param input_tensor:      input data (seq_len, input_dim); PyTorch tensor 
        : param target_len:        number of target values to predict 
        : return np_outputs:       np.array containing predicted values; prediction done recursively 
        '''

        # encode input_tensor
        input_tensor = input_tensor.unsqueeze(1)     # add in batch size of 1
        encoder_output, encoder_hidden = self.encoder(input_tensor)

        # initialize tensor for predictions
        outputs = torch.zeros(target_len, input_tensor.shape[2])

        # decode input_tensor
        decoder_input = input_tensor[-1, :, :]
        decoder_hidden = encoder_hidden
        
        for t in range(target_len):
            decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden)
            outputs[t] = decoder_output.squeeze(0)
            decoder_input = decoder_output
            
        np_outputs = outputs.detach().numpy()
        
        return np_outputs