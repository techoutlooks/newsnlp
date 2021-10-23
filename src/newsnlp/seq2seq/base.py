from torch import nn


class Encoder(nn.Module):

    def __init__(self, **kwargs):
        super().__init__()

    def forward(self, X, *args):
        raise NotImplementedError


class Decoder(nn.Module):

    def __init__(self, **kwargs):
        super().__init__()

    def init_state(self, enc_outputs, *args):
        raise NotImplementedError

    def forward(self, X, *args):
        raise NotImplementedError


class EncoderDecoder(nn.Module):

    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, encX, decX, *args):
        enc_outputs = self.encoder(*args)
        dec_state = self.decoder.init_state(enc_outputs)
        return self.decoder(decX, dec_state)

    def backward(self):
        raise NotImplementedError
