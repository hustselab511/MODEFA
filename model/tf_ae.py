import enum
import torch.nn as nn


class TFAutoencoderMode(enum.Enum):
    TRAIN1 = 1
    TRAIN2 = 2
    EVAL = 3


class TFAutoencoder(nn.Module):
    def __init__(self, input_dim, d_model, dim_feedforward, nhead, e_num_layers, d_num_layers):
        super(TFAutoencoder, self).__init__()

        self.input_proj = nn.Linear(input_dim, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, dropout=0.1, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=e_num_layers)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, dropout=0.1, batch_first=True
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=d_num_layers)

        self.output_proj = nn.Linear(d_model, input_dim)

    def forward(self, x, mode=TFAutoencoderMode.TRAIN1):
        if mode == TFAutoencoderMode.TRAIN1:
            x = self.input_proj(x)
            x = self.encoder(x)
            x = self.decoder(x, x)
            x = self.output_proj(x)

            return x

        if mode == TFAutoencoderMode.TRAIN2 or mode == TFAutoencoderMode.EVAL:
            x = self.input_proj(x)
            x = self.encoder(x)

            return x
