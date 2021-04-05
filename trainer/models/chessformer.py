import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

from einops import repeat

from models.transformer import Transformer
from loss import ChessformerLoss

class Chessformer(pl.LightningModule):

    def __init__(self, dim, depth, heads, mlp_dim, pool = 'cls', channels = 3, dim_head = 64, dropout = 0., emb_dropout = 0., representation="chessformer", save_every=200):

        super().__init__()

        if representation == "alphazero":
            self.num_classes = 4672
        elif representation == "flat":
            self.num_classes = 16384
        elif representation == "chessformer":
            self.num_classes = 4228 # including underpromotion

        self.batch_number = 0
        self.save_every = save_every

        self.embedding_dim = 15
        self.pos_embedding_dim = 15
        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'

        self.pos_embedding = nn.Parameter(torch.randn(1, 66, self.pos_embedding_dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, self.embedding_dim))

        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(self.embedding_dim + self.pos_embedding_dim, depth, heads, dim_head, mlp_dim, dropout)

        self.loss = ChessformerLoss(representation=representation)

        self.pool = pool
        self.to_latent = nn.Identity()

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(self.embedding_dim + self.pos_embedding_dim),
            nn.Linear(self.embedding_dim + self.pos_embedding_dim, self.num_classes + 1) # add value regression
        )

    def forward(self, x, mask=None):

        b, n, _ = x.shape

        cls_token = repeat(self.cls_token, '() n d -> b n d', b = b)
        x = torch.cat((cls_token, x), dim=1)
        pos_emb = repeat(self.pos_embedding, '() n d -> b n d', b = b)
        x = torch.cat((pos_emb, x), dim=2)
        x = self.dropout(x)

        x = self.transformer(x, mask)

        x = x.mean(dim = 1) if self.pool == 'mean' else x[:, 0]

        x = self.to_latent(x)
        if self.mlp_head is not None:
            x = self.mlp_head(x)

        return x

    def training_step(self, batch, batch_idx):
        # training_step defined the train loop. It is independent of forward
        board, result, moves = batch
        x = self.forward(board)

        loss = self.loss(x, result, moves, num_moves=self.num_classes)
        self.log('train_loss', loss)
        return loss

    '''def configure_optimizers(self):
        optimizer = AdamW(self.parameters(), lr=self.learning_rate)
        scheduler = InverseSquareRootLR(optimizer, self.lr_warmup_steps)
        return (
            [optimizer],
            [
                {
                    'scheduler': scheduler,
                    'interval': 'step',
                    'frequency': 1,
                    'reduce_on_plateau': False,
                    'monitor': 'val_loss',
                }
            ]
        )'''

    def save_to_onnx(self, filepath):
        random_batch = torch.zeros([4, 65, 15], dtype=torch.float).cuda()
        self.to_onnx(filepath, random_batch, export_params=True,
            opset_version=11, do_constant_folding=False,
            input_names=["input_board"], output_names=["output_moves"],
            dynamic_axes={'input_board': {0: 'sequence'}, 'output_moves': {0: 'sequence'}})

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer
