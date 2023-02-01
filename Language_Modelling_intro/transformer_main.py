import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from transformer_parts import Encoder_stack, Decoder_stack, Seq_Embedding, Position_Embedding

class Transformer(nn.Module):
    """
    unification of all the transformer parts
    for initalization , this class takes in:
        vocab_size--> number of unique characters
        emb_dim ---> number embedding dimensiions for the vocab_size
        seq_len --> length of the sequences
        enc_mask --> mask to be used in the encoding layers , mostly to mask out padded areas in sequences
                    default is None
        dec_mask --> mask to be used in decoding layers
    the forward takes as input 2 vectors (input sequences , target sequences )
    """
    def __init__(self, vocab_size, emb_dim, seq_len,batch_size=32, n_encoders=6, n_decoders=6, n_heads=8, enc_mask=None, dec_mask=None):
        super().__init__()
        self.emb = Seq_Embedding(vocab_size, emb_dim)
        self.pe = Position_Embedding(dim=emb_dim, max_seq_len=seq_len)
        self.enc_stack = Encoder_stack(n_encoders=n_encoders, seq_len=seq_len, emb_dim=emb_dim, n_heads=n_heads, batch_size= batch_size, mask= enc_mask)
        self.dec_stack = Decoder_stack(n_decoders=n_decoders, seq_len=seq_len, emb_dim=emb_dim, n_heads=n_heads, batch_size=batch_size, mask = dec_mask)

    def forward(self, input_seq, target_seq):
        # embed input and target sequences
        input_seq= self.emb(input_seq)
        target_seq = self.emb(target_seq)
        # add positional information
        input_seq = self.pe(input_seq)
        target_seq = self.pe(target_seq)
        # pass input and target sequences in the encoder and decoder stack
        enc_out = self.enc_stack(input_seq)
        dec_out = self.dec_stack(target_seq, enc_out)

        return dec_out

if __name__ == '__main__':
    vocab_size = 1000
    emb_dim = 40
    seq_len = 256
    batch_size = 32
    a = torch.randint(0, vocab_size, (batch_size, seq_len))
    b = torch.randint(0, vocab_size, (batch_size, seq_len))
    transformer = Transformer(vocab_size=vocab_size, emb_dim=emb_dim, seq_len=seq_len)
    print(transformer(a, b))
