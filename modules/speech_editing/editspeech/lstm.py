import random
import numpy as np
import sys
from tqdm import trange

import torch
import torch.nn as nn
import torch.nn.functional as F
from modules.speech_editing.commons.mel_encoder import MelEncoder


class lstm_encoder(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers = 1):
        super(lstm_encoder, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size = hidden_size, hidden_size = hidden_size,
                            num_layers = num_layers)

    # def forward(self, x_input):
    #     lstm_out, self.hidden = self.lstm(x_input.view(x_input.shape[0], x_input.shape[1], self.input_size))
    #     return lstm_out, self.hidden     
    def forward(self, x_input, encoder_hidden_states):
        lstm_out, self.hidden = self.lstm(x_input.unsqueeze(0), encoder_hidden_states)
        return lstm_out, self.hidden
    
    def init_hidden(self, batch_size, device):
        return (torch.zeros(self.num_layers, batch_size, self.hidden_size).to(device),
                torch.zeros(self.num_layers, batch_size, self.hidden_size).to(device))


class lstm_decoder(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_size, num_layers = 2):
        super(lstm_decoder, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size = in_dim, hidden_size = hidden_size,
                            num_layers = num_layers)
        self.linear = nn.Linear(hidden_size, out_dim)           

    def forward(self, x_input, encoder_hidden_states):
        lstm_out, self.hidden = self.lstm(x_input.unsqueeze(0), encoder_hidden_states)
        output = self.linear(lstm_out.squeeze(0))     
        return output, self.hidden

    def init_hidden(self, batch_size, device):
        return (torch.zeros(self.num_layers, batch_size, self.hidden_size).to(device),
                torch.zeros(self.num_layers, batch_size, self.hidden_size).to(device))

class LSTM_Seq2Seq(nn.Module):
    def __init__(self, prenet_hidden_size, hidden_size, output_dim, teacher_forcing_ratio=0.5):
        super(LSTM_Seq2Seq, self).__init__()

        self.output_dim = output_dim
        self.hidden_size = hidden_size
        self.teacher_forcing_ratio = teacher_forcing_ratio

        self.proj_in = nn.Linear(80, prenet_hidden_size)
        self.prenet = MelEncoder(input_dim=80, hidden_size=prenet_hidden_size)
        self.forward_encoder = lstm_encoder(input_size=80, hidden_size = hidden_size)
        self.backward_encoder = lstm_encoder(input_size=80, hidden_size = hidden_size)
        
        self.forward_decoder = lstm_decoder(in_dim = prenet_hidden_size, out_dim = output_dim, hidden_size = hidden_size)
        self.backward_decoder = lstm_decoder(in_dim = prenet_hidden_size, out_dim = output_dim, hidden_size = hidden_size)


    def forward(self, input_tensor, target_tensor, target_len, time_mel_masks, infer=False):
        
        '''
        train lstm encoder-decoder
        
        : param input_tensor:              input data with shape (seq_len, # in batch, number features); PyTorch tensor    
        : param target_tensor:             target data with shape (seq_len, # in batch, number features); PyTorch tensor
        : param target_len:                number of values to predict 
        '''
        batch_size = input_tensor.size(1)
        forward_target_tensor = target_tensor
        backward_target_tensor = torch.flip(target_tensor, dims=[0])
        
        # Add prenet output
        prenet_output = self.prenet(target_tensor.transpose(0, 1)*(1-time_mel_masks)).transpose(0, 1)
        input_tensor = input_tensor + prenet_output
        
        # input tensor
        forward_input_tensor = input_tensor
        backward_input_tensor = torch.flip(input_tensor, dims=[0])
        # outputs tensor
        forward_outputs = torch.zeros(target_len, batch_size, self.output_dim).to(input_tensor.device)
        backward_outputs = torch.zeros(target_len, batch_size, self.output_dim).to(input_tensor.device)

        # # initialize hidden state
        # forward_encoder_hidden = self.forward_encoder.init_hidden(batch_size, input_tensor.device)
        # backward_encoder_hidden = self.backward_encoder.init_hidden(batch_size, input_tensor.device)
        
        # # encoder outputs
        # _, forward_encoder_hidden = self.forward_encoder(forward_input_tensor)
        # _, backward_encoder_hidden = self.backward_encoder(backward_input_tensor)

        # decode with teacher forcing
        # forward_decoder_input = forward_input_tensor[0, :, :]   # shape: (seq_len, batch_size, hidden_size)
        # backward_decoder_input = backward_input_tensor[0, :, :]   # shape: (seq_len, batch_size, hidden_size)
        forward_decoder_hidden = self.forward_decoder.init_hidden(batch_size, input_tensor.device)
        backward_decoder_hidden = self.backward_decoder.init_hidden(batch_size, input_tensor.device)
        if not infer:
            # Training
            # Teacher forcing
            if random.random() < self.teacher_forcing_ratio:
                for t in range(target_len): 
                    forward_decoder_input = self.proj_in(forward_target_tensor[t, :, :])
                    backward_decoder_input = self.proj_in(backward_target_tensor[t, :, :])
                    forward_decoder_output, forward_decoder_hidden = self.forward_decoder(forward_decoder_input, forward_decoder_hidden)
                    backward_decoder_output, backward_decoder_hidden = self.backward_decoder(backward_decoder_input, backward_decoder_hidden)
                    forward_outputs[t] = forward_decoder_output
                    backward_outputs[t] = backward_decoder_output
            # Normal training
            else:
                for t in range(target_len):
                    forward_decoder_input = forward_input_tensor[t, :, :]
                    backward_decoder_input = backward_input_tensor[t, :, :]
                    forward_decoder_output, forward_decoder_hidden = self.forward_decoder(forward_decoder_input, forward_decoder_hidden)
                    backward_decoder_output, backward_decoder_hidden = self.backward_decoder(backward_decoder_input, backward_decoder_hidden)
                    forward_outputs[t] = forward_decoder_output
                    backward_outputs[t] = backward_decoder_output
        else:
            # Inference
            for t in range(target_len):
                forward_decoder_input = forward_input_tensor[t, :, :]
                backward_decoder_input = backward_input_tensor[t, :, :]
                forward_decoder_output, forward_decoder_hidden = self.forward_decoder(forward_decoder_input, forward_decoder_hidden)
                backward_decoder_output, backward_decoder_hidden = self.backward_decoder(backward_decoder_input, backward_decoder_hidden)
                forward_outputs[t] = forward_decoder_output
                backward_outputs[t] = backward_decoder_output
        
        backward_outputs = torch.flip(backward_outputs, dims=[0])
        return forward_outputs, backward_outputs

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