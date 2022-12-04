
import torch
import torch.nn as nn
import numpy as np
from vqvae.models.encoder import Encoder
from vqvae.models.quantizer import VectorQuantizer
from vqvae.models.decoder import Decoder


class VQVAE(nn.Module):
    def __init__(self, h_dim, res_h_dim, n_res_layers,
                 n_embeddings, embedding_dim, beta, x_in_depth, wh_reduction, decoder_activation, save_img_embedding_map=False):
        super(VQVAE, self).__init__()

        # encode image into continuous latent space
        self.encoder = Encoder(x_in_depth, h_dim, n_res_layers, res_h_dim, wh_reduction)
        self.pre_quantization_conv = nn.Conv2d(
            h_dim, embedding_dim, kernel_size=1, stride=1)
        # pass continuous latent vector through discretization bottleneck
        self.vector_quantization = VectorQuantizer(
            n_embeddings, embedding_dim, beta)
        # decode the discrete latent representation
        self.decoder = Decoder(embedding_dim, h_dim, n_res_layers, res_h_dim, x_in_depth, wh_reduction, decoder_activation=decoder_activation)

        if save_img_embedding_map:
            self.img_to_embedding_map = {i: [] for i in range(n_embeddings)}
        else:
            self.img_to_embedding_map = None

    def forward(self, x, verbose=False):

        z_e = self.encoder(x)

        z_e = self.pre_quantization_conv(z_e)
        embedding_loss, z_q, perplexity, _, _ = self.vector_quantization(
            z_e)
        x_hat = self.decoder(z_q)

        if verbose:
            print('original data shape:', x.shape)
            print('encoded data shape:', z_e.shape)
            print('recon data shape:', x_hat.shape)
            assert False

        return embedding_loss, x_hat, perplexity, z_q

    def encode_and_quantize(self, x, return_indices=False):
        z_e = self.encoder(x)
        z_e = self.pre_quantization_conv(z_e)
        embedding_loss, z_q, perplexity, _, min_encoding_indices = self.vector_quantization(z_e)

        if return_indices:
            return z_q, min_encoding_indices
        return z_q


    def indices_to_zq(self, min_encoding_indices):
        # min_encoding_indices has shape (bs, h, w, 1)

        shape = min_encoding_indices.shape[:-1]
        min_encoding_indices = min_encoding_indices.view(-1, 1)

        min_encodings = torch.zeros(min_encoding_indices.shape[0], self.vector_quantization.n_e).to('cuda')
        min_encodings.scatter_(1, min_encoding_indices, 1)

        z_q = torch.matmul(min_encodings, self.vector_quantization.embedding.weight).view(*shape, -1)
        return z_q.permute(0, 3, 1, 2).contiguous()

    def decode(self, z_q):
        x_hat = self.decoder(z_q)
        return x_hat
