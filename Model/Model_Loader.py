#!/usr/bin/env python
# coding: utf-8

# # Imports

# In[1]:


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchtext
import pickle
from torchtext.legacy import data
from torch.utils.data import Dataset


# ### Classifier class

# Define a class for the LSTM classifier.

# In[2]:


class FilmClassifierLSTM(nn.Module):
    """ 
    Long-short term memory (LSTM) classifier
    Layers: Embedding -> LSTM -> fully connected
    
    Parameters:
        n_vocab: Number of words TEXT Field was trained on
        n_classes: Number of genres
        pad_index: Index of <pad> token
        unk_index: Index of <unk> token
        n_embedding: Size of the trained vectors, e.g if using 'glove.6B.100d', set to 100
        pretrained_embeddings: Vectors from pre-trained word embedding such as GloVe
        n_hidden: Number of hidden layers
        dropout: Dropout rate, e.g 0.2 = 20% dropout
        activation: Set as "softmax" or "sigmoid"
    
    Return on forward pass:
        output: Predicted probabilities for each class
        
    """
    
    def __init__(self, n_vocab, n_classes, pad_index, unk_index, n_embedding, pretrained_embeddings=None,
                 n_hidden=256, dropout=0.2, activation="sigmoid", bidirectional=True):
        super().__init__()
        
        self.bidirectional = bidirectional
        
        if bidirectional:
            # Use two layers for bidirectionality
            n_layers = 2
            # Double size of linear output
            linear_hidden = n_hidden * 2
        else:
            n_layers = 1
            linear_hidden = n_hidden
        
        # Create model layers
        self.dropout = nn.Dropout(dropout)
        self.embedding = nn.Embedding(n_vocab, n_embedding, padding_idx=pad_index) # Tell embedding not to learn <pad> embeddings
        self.lstm = nn.LSTM(n_embedding, n_hidden, num_layers=n_layers, dropout=dropout, bidirectional=bidirectional)
        self.linear = nn.Linear(linear_hidden, n_classes)
        
        # Set output activation function
        if activation == "softmax":
            self.activation = nn.Softmax(dim=1)
        else:
            # Sigmoid recommended for multi-label
            self.activation = nn.Sigmoid()
        
        if pretrained_embeddings != None:
            # Replace weights of embedding layer
            self.embedding.weight.data.copy_(pretrained_embeddings)
            # Set padding and unknown tokens to zero
            self.embedding.weight.data[pad_index] = torch.zeros(n_embedding)
            self.embedding.weight.data[unk_index] = torch.zeros(n_embedding)

    def forward(self, text, text_lengths):
        # Create word embedding, then apply drop out
        embedded = self.embedding(text)
        dropped_embedded = self.dropout(embedded)
        # Pack the embedding so that LSTM only processes non-embedded sequences
        packed_embedded = nn.utils.rnn.pack_padded_sequence(dropped_embedded, text_lengths.to('cpu'))
        # Return output of all hidden states in the sequence, hidden state of the last LSTM unit and cell state
        packed_output, (hidden, cell) = self.lstm(packed_embedded)
        # Unpack packed_output
        unpacked_output, output_lengths = nn.utils.rnn.pad_packed_sequence(packed_output)
        
        if self.bidirectional:
            # Find the final two hidden states and join them together
            top_two_hidden = torch.cat((hidden[-1], hidden[-2]), dim=1)
            # Apply dropout, pass through fully connected layer, then apply activation function
            output = self.activation(self.linear(self.dropout(top_two_hidden)))
        else:
            # Apply dropout to final hidden state, pass through fully connected layer, then apply activation function
            output = self.activation(self.linear(self.dropout(hidden[-1])))

        return output


# In[4]:


model_kwargs_file = 'model_kwargs.pickle'
model_kwargs = pickle.load(open(model_kwargs_file, 'rb'))


# In[14]:


# Create the classifier
lstm_model = FilmClassifierLSTM(**model_kwargs)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
lstm_model = lstm_model.to(device)
lstm_model.load_state_dict(torch.load("film_classifier_best.pt"))

