from torch import Tensor
import torch
import torch.nn as nn
from torch.nn import Transformer
import math
import torch.nn.functional as F
from torch.nn.parameter import Parameter

# helper Module that adds positional encoding to the token embedding to introduce a notion of word order.
class PositionalEncoding(nn.Module):
    def __init__(self,
                 emb_size: int,
                 dropout: float,
                 maxlen: int = 500):
        super(PositionalEncoding, self).__init__()
        den = torch.exp(- torch.arange(0, emb_size, 2)* math.log(10000) / emb_size)
        pos = torch.arange(0, maxlen).reshape(maxlen, 1)
        pos_embedding = torch.zeros((maxlen, emb_size))
        pos_embedding[:, 0::2] = torch.sin(pos * den)
        pos_embedding[:, 1::2] = torch.cos(pos * den)
        pos_embedding = pos_embedding.unsqueeze(-2)

        self.dropout = nn.Dropout(dropout)
        self.register_buffer('pos_embedding', pos_embedding)

    def forward(self, token_embedding: Tensor):
        return self.dropout(token_embedding + self.pos_embedding[:token_embedding.size(0), :])

# helper Module to convert tensor of input indices into corresponding tensor of token embeddings
class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size: int, emb_size):
        super(TokenEmbedding, self).__init__()
        self.embedding = nn.Embedding(vocab_size, emb_size)
        self.emb_size = emb_size

    def forward(self, tokens: Tensor):
        return self.embedding(tokens.long()) * math.sqrt(self.emb_size)


class multi_head(nn.Module):
    def __init__(self, latent_dim, output_size):
        super(multi_head, self).__init__()
        self.fc1 = nn.Linear(latent_dim, 8)
        self.dropout1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(8, output_size)
        self.output_size = output_size

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)
        x = self.fc2(x)
        return x

# Seq2Seq Network
class Seq2SeqTransformer(nn.Module):
    def __init__(self,
                 num_encoder_layers: int,
                 num_decoder_layers: int,
                 emb_size: int,
                 nhead: int,
                 tgt_n_tasks: int,
                 src_size: int,
                 tgt_size: int,
                 dim_feedforward: int = 512,
                 dropout: float = 0.1):
        super(Seq2SeqTransformer, self).__init__()
        self.transformer = Transformer(d_model=emb_size,
                                       nhead=nhead,
                                       num_encoder_layers=num_encoder_layers,
                                       num_decoder_layers=num_decoder_layers,
                                       dim_feedforward=dim_feedforward,
                                       dropout=dropout)
        self.generator = nn.Linear(emb_size, tgt_n_tasks)
        # self.generator = nn.ModuleList([multi_head(emb_size, 1) for _ in range(tgt_n_tasks)])
        self.src_emb = nn.Linear(src_size, emb_size)
        self.tgt_emb = nn.Linear(tgt_size, emb_size)
        self.positional_encoding = PositionalEncoding(emb_size, dropout=dropout)

    def forward(self,
                src: Tensor,
                trg: Tensor,
                src_mask: Tensor,
                tgt_mask: Tensor,
                src_padding_mask: Tensor,
                tgt_padding_mask: Tensor,
                memory_key_padding_mask: Tensor):
        src_emb = self.positional_encoding(self.src_emb(src))
        tgt_emb = self.positional_encoding(self.tgt_emb(trg))
        outs = self.transformer(src_emb, tgt_emb, src_mask, tgt_mask, None,
                                src_padding_mask, tgt_padding_mask, memory_key_padding_mask)
        outs = self.generator(outs)
        # outs = [generator(outs) for generator in self.generator]
        # outs = torch.cat(outs, axis=1)
        return outs

    def encode(self, src: Tensor, src_mask: Tensor, src_key_padding_mask: Tensor=None):
        return self.transformer.encoder(self.positional_encoding(
                            self.src_emb(src)), src_mask, src_key_padding_mask=src_key_padding_mask)

    def decode(self, tgt: Tensor, memory: Tensor, tgt_mask: Tensor, tgt_key_padding_mask: Tensor=None):
        return self.transformer.decoder(self.positional_encoding(
                          self.tgt_emb(tgt)), memory,
                          tgt_mask, tgt_key_padding_mask=tgt_key_padding_mask)

    def decode_cat(self, tgt: Tensor, src: Tensor, memory: Tensor, tgt_mask: Tensor, tgt_key_padding_mask: Tensor=None):
        return self.transformer.decoder(self.positional_encoding(
                          self.tgt_emb(tgt)), torch.cat(memory, self.src_emb(src), axis=1),
                          tgt_mask, tgt_key_padding_mask=tgt_key_padding_mask)



class Seq2SeqTransformer_IDEC(nn.Module):
    def __init__(self, NUM_ENCODER_LAYERS, NUM_DECODER_LAYERS, EMB_SIZE, 
                 NHEAD, TGT_N_TASKS, SRC_SIZE, TGT_SIZE, FFN_HID_DIM, n_z, n_clusters, device):
        super(Seq2SeqTransformer_IDEC, self).__init__()
        self.alpha = 1.0
        self.n_clusters = n_clusters
        self.n_z = n_z
        self.transformer = Seq2SeqTransformer(NUM_ENCODER_LAYERS, NUM_DECODER_LAYERS, EMB_SIZE,
                                 NHEAD, TGT_N_TASKS, SRC_SIZE, TGT_SIZE, FFN_HID_DIM).to(device)
        # cluster layer
        self.cluster_layer = Parameter(torch.Tensor(self.n_clusters, self.n_z))
        self.best_model_state = torch.nan
        torch.nn.init.xavier_normal_(self.cluster_layer.data)
    
    def forward(self, z):
        # cluster
        q = 1.0 / (1.0 + torch.sum(
            torch.pow(z.unsqueeze(1) - self.cluster_layer, 2), 2) / self.alpha)
        q = q.pow((self.alpha + 1.0) / 2.0)
        q = (q.t() / torch.sum(q, 1)).t()
        return q



def target_distribution(q):
    weight = q**2 / q.sum(0)
    return (weight.t() / weight.sum(1)).t()


class Encoder(nn.Module):
    def __init__(self, input_size, latent_dim):
        super(Encoder, self).__init__()
        self.bn = nn.BatchNorm1d(input_size)
        self.fc1 = nn.Linear(input_size, 32)
        self.dropout1 = nn.Dropout(0.2)
        self.fc2 = nn.Linear(32, 32)
        self.dropout2 = nn.Dropout(0.2)
        self.z_mean = nn.Linear(32, latent_dim)
        self.z_log_var = nn.Linear(32, latent_dim)

    def forward(self, x):
        x = self.bn(x)
        x = F.tanh(self.fc1(x))
        x = self.dropout1(x)
        x = F.relu(self.fc2(x))
        x = self.dropout2(x)
        z_mean = self.z_mean(x)
        z_log_var = self.z_log_var(x)
        return z_mean, z_log_var

class Sampling(nn.Module):
    def __init__(self, latent_dim):
        super(Sampling, self).__init__()
        self.latent_dim = latent_dim

    def forward(self, arg):
        z_mean = arg[:, :self.latent_dim]
        z_log_var = arg[:, self.latent_dim:(2 * self.latent_dim)]
        epsilon = torch.randn_like(z_mean)
        return z_mean + torch.exp(z_log_var / 2) * epsilon

class Decoder(nn.Module):
    def __init__(self, latent_dim, output_size):
        super(Decoder, self).__init__()
        self.fc1 = nn.Linear(latent_dim, 8)
        self.dropout1 = nn.Dropout(0.2)
        self.fc2 = nn.Linear(8, 8)
        self.dropout2 = nn.Dropout(0.2)
        self.fc3 = nn.Linear(8, 1)
        self.output_size = output_size

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)
        x = F.relu(self.fc2(x))
        x = self.dropout2(x)
        x = self.fc3(x)
        return x

class VAE(nn.Module):
    def __init__(self, input_size, latent_dim, output_size, n_tasks=17):
        super(VAE, self).__init__()
        self.encoder = Encoder(input_size, latent_dim)
        self.sampling = Sampling(latent_dim)
        self.n_tasks = n_tasks
        self.decoders = nn.ModuleList([Decoder(latent_dim, output_size) for _ in range(self.n_tasks)])

    def forward(self, x):
        z_mean, z_log_var = self.encoder(x)
        z = self.sampling(torch.cat((z_mean, z_log_var), dim=1))
        ys = [decoder(z) for decoder in self.decoders]
        return torch.concat(ys, axis=1)

    def encoding(self, x):
        z_mean, z_log_var = self.encoder(x)
        z = self.sampling(torch.cat((z_mean, z_log_var), dim=1))
        return z

    def decoding(self, z):
        return [decoder(z) for decoder in self.decoders]