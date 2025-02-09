# -*- coding: utf-8 -*-

import os
import math
import sys
sys.path.append("../")
from typing import Optional

import torch
from torch import nn, Tensor
from torch.nn import functional as F
from torch.nn.modules import MultiheadAttention, Linear, Dropout, BatchNorm1d, TransformerEncoderLayer

from tcn import TemporalConvNet

#encoder_layer.self_attn.batch_first = True

    
class TransformerBatchNormEncoderLayer(nn.modules.Module):
    r"""This transformer encoder layer block is made up of self-attn and feedforward network.
    It differs from TransformerEncoderLayer in torch/nn/modules/transformer.py in that it replaces LayerNorm
    with BatchNorm.
    Args:
        d_model: the number of expected features in the input (required).
        nhead: the number of heads in the multiheadattention models (required).
        dim_feedforward: the dimension of the feedforward network model (default=2048).
        dropout: the dropout value (default=0.1).
        activation: the activation function of intermediate layer, relu or gelu (default=relu).
    """

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation="relu"):
        super(TransformerBatchNormEncoderLayer, self).__init__()
        self.self_attn = MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = Linear(d_model, dim_feedforward)
        self.dropout = Dropout(dropout)
        self.linear2 = Linear(dim_feedforward, d_model)

        self.norm1 = BatchNorm1d(d_model, eps=1e-5)  # normalizes each feature across batch samples and time steps
        self.norm2 = BatchNorm1d(d_model, eps=1e-5)
        self.dropout1 = Dropout(dropout)
        self.dropout2 = Dropout(dropout)

        self.activation = _get_activation_fn(activation)

    def __setstate__(self, state):
        if 'activation' not in state:
            state['activation'] = F.relu
        super(TransformerBatchNormEncoderLayer, self).__setstate__(state)

    def forward(self, src: Tensor, src_mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None) -> Tensor:
        r"""Pass the input through the encoder layer.
        Args:
            src: the sequence to the encoder layer (required).
            src_mask: the mask for the src sequence (optional).
            src_key_padding_mask: the mask for the src keys per batch (optional).
        Shape:
            see the docs in Transformer class.
        """
        src2 = self.self_attn(src, src, src, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)  # (seq_len, batch_size, d_model)
        src = src.permute(1, 2, 0)  # (batch_size, d_model, seq_len)
        # src = src.reshape([src.shape[0], -1])  # (batch_size, seq_length * d_model)
        src = self.norm1(src)
        src = src.permute(2, 0, 1)  # restore (seq_len, batch_size, d_model)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)  # (seq_len, batch_size, d_model)
        src = src.permute(1, 2, 0)  # (batch_size, d_model, seq_len)
        src = self.norm2(src)
        src = src.permute(2, 0, 1)  # restore (seq_len, batch_size, d_model)
        return src

class FixedPositionalEncoding(nn.Module):
    r"""Inject some information about the relative or absolute position of the tokens
        in the sequence. The positional encodings have the same dimension as
        the embeddings, so that the two can be summed. Here, we use sine and cosine
        functions of different frequencies.
    .. math::
        \text{PosEncoder}(pos, 2i) = sin(pos/10000^(2i/d_model))
        \text{PosEncoder}(pos, 2i+1) = cos(pos/10000^(2i/d_model))
        \text{where pos is the word position and i is the embed idx)
    Args:
        d_model: the embed dim (required).
        dropout: the dropout value (default=0.1).
        max_len: the max. length of the incoming sequence (default=1024).
    """

    def __init__(self, d_model, dropout=0.1, max_len=1024, scale_factor=1.0):
        super(FixedPositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)  # positional encoding
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = scale_factor * pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)  # this stores the variable in the state_dict (used for non-trainable variables)

    def forward(self, x:Tensor):
        r"""Inputs of forward function
        Args:
            x: the sequence fed to the positional encoder model (required).
        Shape:
            x: [sequence length, batch size, embed dim]
            output: [sequence length, batch size, embed dim]
        """

        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class LearnablePositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=1024):
        super(LearnablePositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        # Each position gets its own embedding
        # Since indices are always 0 ... max_len, we don't have to do a look-up
        self.pe = nn.Parameter(torch.empty(max_len, 1, d_model))  # requires_grad automatically set to True
        nn.init.uniform_(self.pe, -0.02, 0.02)

    def forward(self, x:Tensor):
        r"""Inputs of forward function
        Args:
            x: the sequence fed to the positional encoder model (required).
        Shape:
            x: [sequence length, batch size, embed dim]
            output: [sequence length, batch size, embed dim]
        """

        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

def get_pos_encoder(pos_encoding):
    if pos_encoding == "learnable":
        return LearnablePositionalEncoding
    elif pos_encoding == "fixed":
        return FixedPositionalEncoding

    raise NotImplementedError("pos_encoding should be 'learnable'/'fixed', not '{}'".format(pos_encoding))


def _get_activation_fn(activation):
    if activation == "relu":
        return F.relu
    elif activation == "gelu":
        return F.gelu
    raise ValueError("activation should be relu/gelu, not {}".format(activation))

    
class TCN_TRANS(nn.Module):
    def __init__(self, 
                 tcn_nfilters = [16, 16],
                 tcn_kernel_size = 6,
                 tcn_dropout = 0.2, 
                 trans_d_model = 64, 
                 trans_n_heads = 2, trans_num_layers = 2, 
                 trans_dim_feedforward = 128, 
                 shared_embed_dim = 64,
                 trans_dropout=0.2, 
                 trans_activation='relu', trans_norm='LayerNorm', 
                 trans_freeze=False):
        super(TCN_TRANS, self).__init__()

        self.tcn1 = TemporalConvNet(num_inputs = 1, num_channels = tcn_nfilters, 
                                    kernel_size = tcn_kernel_size, dropout=tcn_dropout)
        self.tcn2 = TemporalConvNet(num_inputs = 1, num_channels = tcn_nfilters, 
                                    kernel_size = tcn_kernel_size, dropout=tcn_dropout)
        self.tcn3 = TemporalConvNet(num_inputs = 1, num_channels = tcn_nfilters, 
                                    kernel_size = tcn_kernel_size, dropout=tcn_dropout)

        self.sig = nn.Sigmoid()

        feat_dim = tcn_nfilters[-1]
        self.project_1 = nn.Linear(feat_dim, trans_d_model)
        self.project_2 = nn.Linear(feat_dim, trans_d_model)
        self.project_3 = nn.Linear(feat_dim, trans_d_model)
        self.layernorm_1 = nn.LayerNorm(trans_d_model)
        self.layernorm_2 = nn.LayerNorm(trans_d_model)
        self.layernorm_3 = nn.LayerNorm(trans_d_model)
        
        self.d_model = trans_d_model
        # Transformer
        if trans_norm == 'LayerNorm':
            encoder_layer = TransformerEncoderLayer(trans_d_model, 
                                                    trans_n_heads, 
                                                    trans_dim_feedforward, 
                                                    trans_dropout*(1.0 - trans_freeze), 
                                                    activation=trans_activation)
        else:
            encoder_layer = TransformerBatchNormEncoderLayer(trans_d_model, 
                                                              trans_n_heads, 
                                                              trans_dim_feedforward, 
                                                              trans_dropout*(1.0 - trans_freeze), 
                                                              activation=trans_activation)

        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, trans_num_layers)

        self.project_embed = nn.Linear(trans_d_model, shared_embed_dim)
        
    def forward(self, x1:Tensor, x2:Tensor, x3:Tensor):
        """
        Args:
            X: (batch_size, seq_length, feat_dim) torch tensor (input)
        """
        x1 = x1.unsqueeze(1)
        x2 = x2.unsqueeze(1)
        x3 = x3.unsqueeze(1)
        
        x1 = self.sig(self.tcn1(x1))
        x2 = self.sig(self.tcn2(x2))
        x3 = self.sig(self.tcn3(x3))
     
        x1 = x1.permute(2, 0, 1)  
        x2 = x2.permute(2, 0, 1)  
        x3 = x3.permute(2, 0, 1)  
        
        seq_length = x1.shape[0]
        
        x1 = self.project_1(x1) * math.sqrt(self.d_model) 
        x2 = self.project_2(x2) * math.sqrt(self.d_model) 
        x3 = self.project_3(x3) * math.sqrt(self.d_model) 
        
        x1 = self.layernorm_1(x1.permute(1, 0, 2)).permute(1, 0, 2)  
        x2 = self.layernorm_2(x2.permute(1, 0, 2)).permute(1, 0, 2)  
        x3 = self.layernorm_3(x3.permute(1, 0, 2)).permute(1, 0, 2)  
        
        x = torch.cat((x1, x2, x3), dim= 0)
        x = self.transformer_encoder(x)  

        x1 = x[:seq_length,:,:]
        x2 = x[seq_length:seq_length*2,:,:]
        x3 = x[seq_length*2:,:,:]

        return x1,x2,x3

    
class SSL_MODEL(nn.Module):
    def __init__(self,
                  tcn_nfilters = [16, 16],
                  tcn_kernel_size = 6,
                  tcn_dropout = 0.2, 
                  trans_d_model = 128, 
                  trans_n_heads = 4, trans_num_layers = 1, 
                  trans_dim_feedforward = 128, 
                  shared_embed_dim = 64,
                  trans_dropout=0.2, 
                  trans_activation='relu', trans_norm='LayerNorm', 
                  trans_freeze=False,
                  ssl_embed_dim = 64, ssl_num_classes = 6, 
                  ssl_activation='relu', ssl_dropout = 0.1):
        super(SSL_MODEL, self).__init__()
        
        self.model_fusion =TCN_TRANS(
                  tcn_nfilters,
                  tcn_kernel_size,
                  tcn_dropout, 
                  trans_d_model, 
                  trans_n_heads, trans_num_layers, 
                  trans_dim_feedforward, 
                  shared_embed_dim,
                  trans_dropout, 
                  trans_activation, trans_norm, 
                  trans_freeze)

        self.act = _get_activation_fn(ssl_activation)
        self.dropout = nn.Dropout(ssl_dropout)
        
        # Modality-specific classification
        self.fc1 = nn.Linear(trans_d_model, ssl_embed_dim)
        self.bn1 = nn.BatchNorm1d(ssl_embed_dim)
        self.fc21 = nn.Linear(ssl_embed_dim, ssl_num_classes)

        self.fc2 = nn.Linear(trans_d_model, ssl_embed_dim)
        self.bn2 = nn.BatchNorm1d(ssl_embed_dim)
        self.fc22 = nn.Linear(ssl_embed_dim, ssl_num_classes)

        self.fc3 = nn.Linear(trans_d_model, ssl_embed_dim)
        self.bn3 = nn.BatchNorm1d(ssl_embed_dim)
        self.fc23 = nn.Linear(ssl_embed_dim, ssl_num_classes)
        
    def forward(self, x1, x2, x3):

        x1,x2,x3 = self.model_fusion(x1, x2, x3)
     
        x1 = x1.permute(1,2,0)  
        x2 = x2.permute(1,2,0)  
        x3 = x3.permute(1,2,0)  
        
        x1 = F.avg_pool1d(x1, kernel_size = x1.shape[-1], stride=1).squeeze(-1)
        x2 = F.avg_pool1d(x2, kernel_size = x2.shape[-1], stride=1).squeeze(-1)
        x3 = F.avg_pool1d(x3, kernel_size = x3.shape[-1], stride=1).squeeze(-1)
        
        x1 = self.fc1(x1)
        x1 = self.bn1(x1)
        x1 = self.act(x1)
        x1 = self.dropout(x1)
        x1 = self.fc21(x1)
        
        x2 = self.fc2(x2)
        x2 = self.bn2(x2)
        x2 = self.act(x2)
        x2 = self.dropout(x2)
        x2 = self.fc22(x2)
        
        x3 = self.fc3(x3)
        x3 = self.bn3(x3)
        x3 = self.act(x3)
        x3 = self.dropout(x3)
        x3 = self.fc23(x3)

        return x1, x2, x3
    
    






class SL_model(nn.Module):
    def __init__(self,
                 num_classes,
                 CUDA = False,
                 tcn_nfilters = [16, 16],
                 tcn_kernel_size = 6,
                 tcn_dropout = 0.2,
                 trans_d_model = 64,
                 trans_n_heads = 4, trans_num_layers = 1,
                 trans_dim_feedforward = 128,
                 shared_embed_dim = 64,
                 trans_dropout=0.2,
                 trans_activation='relu', trans_norm='LayerNorm',
                 trans_freeze=False,
                 sl_embed_dim1 = 64,
                 sl_embed_dim2 = 64,
                 sl_activation='relu', sl_dropout = 0.2):
        super(SL_model, self).__init__()
        

        self.act = _get_activation_fn(sl_activation)
        self.dropout = nn.Dropout(sl_dropout)
        
        self.model_fusion =TCN_TRANS(tcn_nfilters, tcn_kernel_size,
                                     tcn_dropout, trans_d_model,
                                     trans_n_heads, trans_num_layers,
                                     trans_dim_feedforward, shared_embed_dim,
                                     trans_dropout, trans_activation,
                                     trans_norm, trans_freeze)

        if CUDA:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self.model_fusion = self.model_fusion.to(device)
        
        
        
        self.fc1 = nn.Linear(trans_d_model*3, sl_embed_dim1)
        self.bn1 = nn.BatchNorm1d(sl_embed_dim1)
        self.fc_final = nn.Linear(sl_embed_dim1, num_classes)

                    

    def forward(self, x1, x2, x3):

        x1,x2,x3 = self.model_fusion(x1, x2, x3)
 
        x1 = x1.permute(1,2,0)
        x2 = x2.permute(1,2,0)
        x3 = x3.permute(1,2,0)
        #print('x1',x1.shape)
        #print('x2',x2.shape)
        #print('x3',x3.shape)
        
        x1 = F.avg_pool1d(x1, kernel_size = x1.shape[-1], stride=1).squeeze(-1)
        x2 = F.avg_pool1d(x2, kernel_size = x2.shape[-1], stride=1).squeeze(-1)
        x3 = F.avg_pool1d(x3, kernel_size = x3.shape[-1], stride=1).squeeze(-1)
        
        x = torch.cat((x1, x2, x3), dim= 1)
        #print('torch_embed', x.shape)
        x = self.fc1(x)
        #print('fc_embed', x.shape)
        #proj_dim = self.fc1(x)
        #proj_dim = self.bn1(x)
        x = self.bn1(x)
        #print('project_embed', x.shape)
        x = self.act(x)
        proj_dim  = self.act(x)
        x = self.dropout(x)
        x = self.fc_final(x)
            
        return proj_dim
        




class Den_Modified_Tran(nn.Module):
    """
        EMBEDDING_SIZE: It is the dimension of the embedding feature vector (i.e. 1024).
        CLASS_SIZE: Total number of classes in training and validation (i.e. 100),
                    it is compulsary if the ONLY_EMBEDDINGS is FALSE.
        PRETRAINED: TRUE or FALSE to flag the Imagenet pretrained model.
        ONLY_EMBEDDINGS: if TRUE, dimensions of both training/validation and test outputs are equal to EMBEDDING SIZE.
                         if FALSE, training/validation outputs = CLASS_SIZE while test outputs = EMBEDDING SIZE.
                         TRUE is necessary if custom penalty functions are utilized such as AAMP, LMCP
                         FALSE is necessary if standard softmax is preferred for training.
        L2_NORMED: The output of test pairs are normalized if it is TRUE. For AAMP, LMCP (Margin losses) it should be TRUE

    """
    def __init__(self, embedding_size, class_size=None, pretrained=True, only_embeddings=True,l2_normed=False):
        super(Den_Modified_Tran, self).__init__()
        self.only_embeddings=only_embeddings
        self.l2_normed=l2_normed
        input_size = 3
        len_size=240
        head_size = 64
        num_heads = 4
        #self.model = densenet161(pretrained=pretrained)
        #self.model = TimeSeriesModel(input_channels=5, embedding_size=embedding_size)
        #self.model=CNN_Model()
        self.model=SL_model(num_classes = 56,
                    CUDA = False,
                    tcn_nfilters = [16, 16],
                    tcn_kernel_size = 3,
                    tcn_dropout = 0.3,
                    trans_d_model = 128,
                    trans_n_heads = 4, trans_num_layers = 3,
                    trans_dim_feedforward = 128,
                    shared_embed_dim = 128,
                    trans_dropout=0.1,
                    trans_activation='relu',
                    trans_norm='LayerNorm',
                    trans_freeze=False,
                    sl_embed_dim1 = 64,
                    sl_embed_dim2 = 64,
                    sl_activation='relu',
                    sl_dropout = 0.1)
        

        self.model.classifier = nn.Sequential(
            #nn.BatchNorm1d(64),
            #nn.Dropout(p=0.1),
            nn.Linear(64, embedding_size),
            #nn.BatchNorm1d(embedding_size),
        )
        if not self.only_embeddings:
            self.final=nn.Linear(embedding_size,class_size)

        # Weight initialization
        for m in self.model.classifier:
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
            if isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x1,x2,x3, train=True):
        #features = self.model.features(x)
        #out = F.relu(features, inplace=True)
        #out = F.avg_pool2d(out, kernel_size=7, stride=1).view(features.size(0), -1)
        #print(x1.shape)
        out = self.model(x1,x2,x3)
        if train:
            if self.only_embeddings:
                return self.model.classifier(out)
            else:
                return self.final(self.model.classifier(out))
        else:
            if self.l2_normed:
                return l2_norm(self.model.classifier(out))
            else:
                return self.model.classifier(out)























import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F

import torch
import torch.nn as nn
import torch.nn.functional as F



class CNN_Model(nn.Module):
    def __init__(self):
        super(CNN_Model, self).__init__()

        # 187 x 1
        self.conv1 = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=2, stride=1, padding=0),
            nn.BatchNorm1d(32),
            nn.ReLU())

        # 186 x 32
        self.conv2 = nn.Sequential(
            nn.Conv1d(32, 64, kernel_size=2, stride=2, padding=0),
            nn.BatchNorm1d(64),
            nn.ReLU())

        # 93 x 64
        self.conv3 = nn.Sequential(
            nn.Conv1d(64, 128, kernel_size=2, stride=1, padding=0),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(4, stride=4))

        # 23 x 128
        self.conv4 = nn.Sequential(
            nn.Conv1d(128, 256, kernel_size=3, stride=1, padding=0),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.MaxPool1d(3, stride=3))

        # 7 x 256
        self.conv5 = nn.Sequential(
            nn.Conv1d(256, 64, kernel_size=3, stride=1, padding=0),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(3, stride=3))

        # 1 x 512
        self.projection = nn.Linear(576, 64)
        self.lastlayer = nn.Linear(64,6)


    def forward(self, x):
        # expected conv1d input : minibatch_size x num_channel x width
        #x = x.view(x.shape[0], 1,-1)
        x = x.reshape(x.shape[0], 1, -1)

        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        out = self.conv4(out)
        out = self.conv5(out)

        out_f = out.view(x.shape[0], out.size(1) * out.size(2))
        #print(out_f.shape)
        logit = self.projection(out_f)
        out = self.lastlayer(logit)

        return logit



def l2_norm(input):
    input_size = input.size()
    buffer = torch.pow(input, 2)
    normp = torch.sum(buffer, 1).add_(1e-10)
    norm = torch.sqrt(normp)
    _output = torch.div(input, norm.view(-1, 1).expand_as(input))
    output = _output.view(input_size)

    return output







'''

import torch
import torch.nn as nn

class CNN_Model(nn.Module):
    def __init__(self, num_classes=6, use_global_pooling=True, dropout_rate=0.5):
        super(CNN_Model, self).__init__()

        # Define Convolutional Blocks
        self.conv1 = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=2, stride=1, padding=0),
            nn.BatchNorm1d(32),
            nn.ReLU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(32, 64, kernel_size=2, stride=2, padding=0),
            nn.BatchNorm1d(64),
            nn.ReLU()
        )
        self.conv3 = nn.Sequential(
            nn.Conv1d(64, 128, kernel_size=2, stride=1, padding=0),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(4, stride=4)
        )
        self.conv4 = nn.Sequential(
            nn.Conv1d(128, 256, kernel_size=3, stride=1, padding=0),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.MaxPool1d(3, stride=3)
        )
        self.conv5 = nn.Sequential(
            nn.Conv1d(256, 64, kernel_size=3, stride=1, padding=0),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(3, stride=3)
        )

        # Optional global pooling for variable input lengths
        self.use_global_pooling = use_global_pooling
        if self.use_global_pooling:
            self.global_pool = nn.AdaptiveAvgPool1d(1)

        # Fully connected layers
        self.fc1 = nn.Sequential(
            nn.Linear(64 if use_global_pooling else 576, 64),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )
        self.last_layer = nn.Linear(64, num_classes)

    def forward(self, x):
        # Reshape input for Conv1D
        x = x.reshape(x.shape[0], 1, -1)

        # Pass through convolutional layers
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)

        if self.use_global_pooling:
            x = self.global_pool(x).squeeze(-1)  # Shape: [batch_size, channels]
        else:
            x = x.view(x.shape[0], -1)  # Flatten for dense layers

        # Fully connected layers
        x = self.fc1(x)
        x = self.last_layer(x)

        return x



#model = CNN_Model(num_classes=10, use_global_pooling=True)

model = CNN_Model(num_classes=6)  # Assuming this matches your updated code

#To freeze all layers except the final layer:



for param in model.parameters():
    param.requires_grad = False  # Freeze all parameters


import torch.nn as nn

model.last_layer = nn.Linear(64, num_new_classes)  # Replace with your target classes

for param in model.last_layer.parameters():
    param.requires_grad = True

# Example: Unfreeze the projection layer
for param in model.projection.parameters():
    param.requires_grad = True

optimizer = torch.optim.Adam(
    filter(lambda p: p.requires_grad, model.parameters()), lr=0.001
)
'''


if __name__ == '__main__':
    import torch.optim as optim
    import numpy as np
    # Loading the model into GPU
    device = torch.device("cpu")
    model = SL_model_valid(num_classes = 176,
                    tcn_nfilters = [16, 16],
                    tcn_kernel_size = 6,
                    tcn_dropout = 0.2,
                    trans_d_model = 128,
                    trans_n_heads = 4, trans_num_layers = 1,
                    trans_dim_feedforward = 128,
                    shared_embed_dim = 64,
                    trans_dropout=0.2,
                    trans_activation='relu',
                    trans_norm='LayerNorm',
                    trans_freeze=False,
                    sl_embed_dim1 = 64,
                    sl_embed_dim2 = 64,
                    sl_activation='relu',
                    sl_dropout = 0.2).to(device)

    # Initialize our optimizer and loss function
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    lossFn = nn.CrossEntropyLoss()
   
    import os
    import torch
    import torch
    # Load the state dictionary from the checkpoint
    file_path = "/Users/mohamedbenouis/Desktop/Alfredo_india/WER-SSL-main/Down/saved_model/SL/2024_12_09_16_20/best/best_model.pth"
    checkpoint = torch.load(file_path)
    state_dict = checkpoint['state_dict_down']
    # Get the model's current state_dict
    model_state_dict = model.state_dict()
    # Filter the state_dict to load only matching layers
    filtered_state_dict = {
    k: v for k, v in state_dict.items() if k in model_state_dict and model_state_dict[k].size() == v.size()}
    # Update the model's state_dict
    model_state_dict.update(filtered_state_dict)
    model.load_state_dict(model_state_dict)

    # Load the state dictionary from the .pth file
    #file_path = "/Users/mohamedbenouis/Desktop/Alfredo_india/WER-SSL-main/Down/saved_model/SL/2024_12_09_16_20/best/best_model.pth"
    #checkpoint = torch.load(file_path)  # Load the checkpoint file
    
    #model_dict = checkpoint['state_dict_down']  # Extract the specific state dictionary
    # Load the state dictionary into the model
    
    #model.load_state_dict(model_dict)


    x = torch.randn(64,240).to(device, dtype=torch.float)
    #print(model)
    pred = model(x,x,x)
