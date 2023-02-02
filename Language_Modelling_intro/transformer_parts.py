import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

class Seq_Embedding(nn.Module):
    """
    embeds characters of samples of sequence in given number of dimensions
    input is samples of sequences with shape --> (samples, max_seq_len)
    output has matrix with shape --> (samples, max_seq_len, emb_dim)
    """
    def __init__(self, vocab_size , dim):
        super().__init__()
        self.vocab_size= vocab_size
        self.dim = dim
        self.seq_e = nn.Embedding(self.vocab_size, self.dim)

    def forward(self, x):
        out = self.seq_e(x)
        return out

class Position_Embedding(nn.Module):
    """adds position embeddings to sequence embeddings
    input is embedded samples of sequences with shape --> (samples, max_seq_len, emb_dim)
    output has matrix with shape --> (samples, max_seq_len, emb_dim)
    """
    def __init__(self, dim, max_seq_len, n=10000):
        super().__init__()
        self.dim = dim # embedding dimensions
        self.max_seq = max_seq_len # max sequence length
        self.pe = torch.zeros(self.max_seq, self.dim) # initailize position embedding matrix
        self.n = n
        self.create_pe()

    def create_pe(self):
        for pos in range(0, self.max_seq):
            for i in range(0, self.dim, 2):
                self.pe[pos, i]= np.sin(pos/(self.n**(2*i/self.dim)))
                self.pe[pos, i + 1] = np.cos(pos/(self.n**(2*i/self.dim)))
        return self

    def visualize(self):
        fig, ax = plt.subplots(nrows=1, ncols= 3, figsize=(15,6))
        ax[0].plot(self.pe[0])
        ax[1].plot(self.pe[1])
        ax[2].plot(self.pe[2])

    def forward(self, x):
        out = x + self.pe
        return out

class Q_K_V(nn.Module):
    """
    layer that creates the query , key , value matrices
    input is embedded sequences of shape --> (n_samples, seq_len, emb_dim)
    output is tuple of query, key, value with shapes -->(n_samples,seq_len, emb_dim)

    """
    def __init__(self):
        super().__init__()

    def forward(self, x):
        # get dimensions of input
        n_samples , seq_len, emb_dim = x.size()
        # initialize linear layer
        self.linear_layer = nn.Linear(in_features=seq_len*emb_dim, out_features=seq_len*emb_dim)
        self.query = self.linear_layer(x.view(n_samples, seq_len*emb_dim)).view(n_samples, seq_len, emb_dim)
        self.key = self.linear_layer(x.view(n_samples, seq_len*emb_dim)).view(n_samples, seq_len, emb_dim)
        self.value = self.linear_layer(x.view(n_samples, seq_len*emb_dim)).view(n_samples, seq_len, emb_dim)
        return self.query, self.key, self.value


class MultiHeadAttention(nn.Module):
    """
    implements attention to the input
    input is 3 matrices of shape (#samples, max_seq_len, emb_dim)
    input is reshaped to 3 matrices of shape (#samples, heads, max_seq_len, query_size)
    output is a matrix with shape (#samples, seq_len, emb_dim)
    """
    def __init__(self, emb_dim, n_heads, mask=None):
        super().__init__()
        self.heads = n_heads
        self.emb_dim = emb_dim
        self.query_size = self.emb_dim//n_heads
        self.mask = mask

    def forward(self, q, k, v):
        n_samples, seq_len, emb_dim = q.size()
        assert self.emb_dim == emb_dim # check shape input matrix
        q= q.view(n_samples, self.heads, seq_len, self.query_size)
        # print(q.shape, 'q_shape after reshaping')
        v = v.view(n_samples, self.heads, seq_len, self.query_size)
        # print(v.shape, 'v_shape after reshaping')
        k= k.view(n_samples, self.heads, seq_len, self.query_size)
        # print(k.shape, 'k_shape after reshaping')
        wei = q @ k.transpose(-2, -1) # shape --> (n_samples, heads, seq_len, seq_len)
        # print(wei.shape, 'wei_shape')
        if self.mask is not None:
            wei = wei + self.mask
        wei = wei * (self.query_size**-0.5)
        wei = F.softmax(wei, dim= -1)# shape--> (n_samples, heads, seq_len, seq_len)
        a_s = wei @ v #attention_score matrix with shape-->(n_samples, heads, seq_len, query_size)
        # print(a_s.shape, 'a_s_shape before reshaping')
        out = a_s.view(n_samples, seq_len, self.emb_dim)
        # print(out.shape, 'a_s_shape after reshaping')
        return out

class Encoder(nn.Module):
    """
    unification of encoder which incoporates attention, batch norm , feed forwrd
    and batchnorm layers implemented with residual skip connections
    recieves input from seq and position embedding, a matrix with shape -->
    (n_samples(batch_size), seq_len, emb_dim)
    output is matrix of shape --> (n_samples(batch_size), seq_len, emb_dim)
    """
    def __init__(self, seq_len, emb_dim, n_heads, n_samples, mask=None):
        super().__init__()
        self.seq_len= seq_len
        self.emb_dim = emb_dim
        self.n_heads = n_heads
        self.batch_size = n_samples
        self.mask = mask
        self.q_k_v = Q_K_V() #creates query, key , value matrices
        self.attention = MultiHeadAttention(emb_dim=self.emb_dim, n_heads=self.n_heads, mask=mask) # calculates attention score
        self.batchNorm = nn.BatchNorm1d(num_features=self.emb_dim)# batch nomr layer
        self.feed_forward = nn.Linear(in_features=self.emb_dim*self.seq_len, out_features=self.emb_dim*self.seq_len)

    def forward(self, x):
        x_copy = torch.clone(x) # matrix with shape -->(n_samples, seq_len, emb_dim)
        q, k, v = self.q_k_v(x) # 3 matrices with shape --> (n_samples, seq_len, emb_dim)
        x_attention = self.attention(q,k,v) # matrix with shape -->(n_samples, seq_len, emb_dim)
        x = x_copy + x_attention # skip residual connection
        # batch norm layer , matrix has to be reshaped to (n_samples, emb_dim, seq_len)
        #in order to use pytorch batchNorm api
        x = self.batchNorm(x.view(self.batch_size, self.emb_dim, self.seq_len))
        # reshaping x back to shape -->(n_samples, seq_len, emb_dim)
        x = x.view(self.batch_size, self.seq_len, self.emb_dim)
        x_copy = torch.clone(x)
        # output line below has shape -->(n_samples, seq_len, emb_dim)
        x = self.feed_forward(x.view(self.batch_size, self.seq_len*self.emb_dim)).view(self.batch_size, self.seq_len, self.emb_dim)
        x = x + x_copy
#         fig, ax = plt.subplots(nrows=1, ncols= 2)
#         ax[0].imshow(x.detach().numpy()[0] > 0.99, cmap='gray', interpolation='nearest')
#         ax[0].set_title('before norm')
        x = self.batchNorm(x.view(self.batch_size, self.emb_dim, self.seq_len)) # another batch layer
#         ax[1].imshow(x.detach().numpy()[0] > 0.99, cmap='gray', interpolation='nearest')
#         ax[1].set_title('after norm')

        x = torch.tanh(x)
        out = x.view(self.batch_size, self.seq_len, self.emb_dim)

        return out

class Q_K_V_dec_enc(nn.Module):
    """
    this is similar to the Q_K_V class for the attention layer
    input is 2 matrices:
        1) encoder stack output with shape --> (n_samples(batch_size), seq_len, emb_dim)
            this is assigned to the key and value matrices
        2) target output with  shape --> (n_samples(batch_size), seq_len, emb_dim)
            this is assigned to the query matrix
    output is tuple of query, key, value with shapes -->(n_samples(batch_size),seq_len, emb_dim)
    """
    def __init__(self):
        super().__init__()

    def forward(self, e_out, t_out):
        batch_size , seq_len, emb_dim = t_out.size()
        # initialize linear layer
        linear_layer = nn.Linear(in_features=seq_len*emb_dim, out_features=seq_len*emb_dim)
        query = linear_layer(t_out.view(batch_size, seq_len*emb_dim)).view(batch_size, seq_len, emb_dim)
        key = linear_layer(e_out.view(batch_size, seq_len*emb_dim)).view(batch_size, seq_len, emb_dim)
        value = linear_layer(e_out.view(batch_size, seq_len*emb_dim)).view(batch_size, seq_len, emb_dim)
        return query, key, value

class Enc_Dec_Attention(nn.Module):
    """
    input is:
        q= q # query matrix with shape --> (n_samples(batch_size),seq_len, emb_dim)
        k= k # key matrix with shape --> (n_samples(batch_size),seq_len, emb_dim)
        v = v # value matrix with shape --> (n_samples(batch_size),seq_len, emb_dim)
    """
    def __init__(self, emb_dim, n_heads, mask=None):
        super().__init__()
        self.heads = n_heads
        self.emb_dim = emb_dim
        self.query_size = self.emb_dim//n_heads
        self.mask = mask

    def forward(self, q, k, v):
        n_samples, seq_len, emb_dim = q.size()
        assert self.emb_dim == emb_dim # check shape input matrix

        q= q.view(n_samples, self.heads, seq_len, self.query_size)
        # print(q.shape, 'q_shape after reshaping')
        v = v.view(n_samples, self.heads, seq_len, self.query_size)
        # print(v.shape, 'v_shape after reshaping')
        k= k.view(n_samples, self.heads, seq_len, self.query_size)
        # print(k.shape, 'k_shape after reshaping')
        wei = q @ k.transpose(-2, -1) # shape --> (n_samples, heads, seq_len, seq_len)
        # print(wei.shape, 'wei_shape')
        if self.mask is not None:
            wei = wei + self.mask
        wei = wei * (self.query_size**-0.5)
        wei = F.softmax(wei, dim= -1)# shape--> (n_samples, heads, seq_len, seq_len)
        a_s = wei @ v #attention_score matrix with shape-->(n_samples, heads, seq_len, query_size)
        # print(a_s.shape, 'a_s_shape before reshaping')
        out = a_s.view(n_samples, seq_len, self.emb_dim)
        # print(out.shape, 'a_s_shape after reshaping')
        return out

class Decoder(nn.Module):
    """
    unification of decoder which incoporates attention, enc_dec attention, batch norm,
    feed forwrd and batchnorm layers implemented with residual skip connections
    recieves input from seq and position embedding or other decoders and from encoding stack a matrix with shape -->
    (n_samples(batch_size), seq_len, emb_dim)
    output is matrix of shape --> (n_samples(batch_size), seq_len, emb_dim)
    """
    def __init__(self, seq_len, emb_dim, batch_size, mask=None, n_heads = 4):
        super().__init__()
        self.seq_len = seq_len
        self.emb_dim = emb_dim
        self.batch_size = batch_size
        self.mask = mask
        self.n_heads = n_heads
        if self.mask is None:
            tril = torch.tril(torch.ones((self.seq_len,self.seq_len)))
            mask = torch.zeros((self.seq_len, self.seq_len))
            mask = mask.masked_fill(tril ==0, float('-inf'))
            self.mask = mask
        self.q_k_v = Q_K_V()
        self.q_k_v_dec_enc = Q_K_V_dec_enc()
        self.attention = MultiHeadAttention(emb_dim=self.emb_dim, n_heads=self.n_heads, mask=self.mask)
        self.batchNorm = nn.BatchNorm1d(num_features=self.emb_dim)
        self.enc_dec_attention = Enc_Dec_Attention(emb_dim=self.emb_dim, n_heads=self.n_heads, mask=self.mask)
        self.feed_forward = nn.Linear(in_features=self.emb_dim*self.seq_len, out_features=self.emb_dim*self.seq_len)

    def forward(self, target, enc_out):
        target_copy = torch.clone(target) # matrix with shape -->(n_samples, seq_len, emb_dim)
        q, k, v = self.q_k_v(target) # 3 matrices with shape --> (n_samples, seq_len, emb_dim)
        target_attention = self.attention(q,k,v) # matrix with shape -->(n_samples, seq_len, emb_dim)
        target = target_copy + target_attention # skip residual connection
        # batch norm layer , matrix has to be reshaped to (n_samples, emb_dim, seq_len)
        #in order to use pytorch batchNorm api
        target = self.batchNorm(target.view(self.batch_size, self.emb_dim, self.seq_len))
        # reshaping x back to shape -->(n_samples, seq_len, emb_dim)
        target = target.view(self.batch_size, self.seq_len, self.emb_dim)
        target_copy = torch.clone(target)
        q,k,v = self.q_k_v_dec_enc(target, enc_out)
        target = self.enc_dec_attention(q,k,v)
        target = target + target_copy # skip residual connection
        target = self.batchNorm(target.view(self.batch_size, self.emb_dim, self.seq_len))
        # reshaping x back to shape -->(n_samples, seq_len, emb_dim)
        target = target.view(self.batch_size, self.seq_len, self.emb_dim)
        target_copy = torch.clone(target)
        # output of line below has shape -->(n_samples, seq_len, emb_dim)
        target = self.feed_forward(target.view(self.batch_size, self.seq_len*self.emb_dim)).view(self.batch_size, self.seq_len, self.emb_dim)
        target = target + target_copy # skip residual connection
        target = self.batchNorm(target.view(self.batch_size, self.emb_dim, self.seq_len)) # another batch layer
        target = torch.tanh(target)
        out = target.view(self.batch_size, self.seq_len, self.emb_dim)

        return out

class Encoder_stack(nn.Module):
    '''
    a stack of encoder layers connected sequentially
    takes input as embedded matrix of sequences with shape:
        (batch_size, seq_len, emb_dim)
    output is a matrix of shape --> (batch_size, seq_len, emb_dim)
    '''
    def __init__(self, n_encoders, seq_len, emb_dim, n_heads, batch_size, mask=None):
        super().__init__()
        self.enc = Encoder(seq_len=seq_len, emb_dim=emb_dim, n_heads=n_heads, n_samples= batch_size, mask =mask)
        self.encoders_stack = [self.enc]*n_encoders

    def forward(self, x):
        for enc in self.encoders_stack:
            x = enc(x)
        return x

class Decoder_stack(nn.Module):
    '''
    a stack of decoder layers connected sequentially
    takes input as embedded matrix of sequences with shape
        (batch_size, seq_len, emb_dim), and enc_stak output
    output is a matrix of shape --> (batch_size, seq_len, emb_dim)
    '''
    def __init__(self, n_decoders, seq_len, emb_dim, n_heads, batch_size, mask=None):
        super().__init__()
        self.dec = Decoder(seq_len=seq_len, emb_dim=emb_dim, n_heads = n_heads, batch_size=batch_size, mask= mask)
        self.decoders_stack = [self.dec]*n_decoders

    def forward(self, x, enc_output):
        for dec in self.decoders_stack:
            x = dec(x, enc_output)
        return x


# class Generate_Output(nn.Module):
#     """
#     generates output from input of decoding layers
#     """
#     def __init__(self, hidden_nuerons= 32, seq_len, emb_dim):
#         super().__init__()
#         self.linear1 = nn.Linear(in_features=seq_len*emb_dim)
