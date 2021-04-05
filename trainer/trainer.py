import os
import sys
import argparse
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
import pytorch_lightning as pl

from models.chessformer import Chessformer
from dataset import ChessIterableDataset
from checkpointer import ChessformerCheckpointer

def collate_boards(batch):
    return batch[0] # do nothing

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Chessformer trainer')
    parser.add_argument('--chessgames', type=str, required=True, help='Path of the directory containing chess games')
    parser.add_argument('--resume', type=str, required=False, default=None, help='Resume model from previous checkpoint')
    parser.add_argument('--checkpoint_dir', type=str, required=False, default="checkpoints/", help='Save checkpoints in this directory')
    parser.add_argument('--history', type=int, default=-1, help='History to use for training')
    args = parser.parse_args()

    dataset = ChessIterableDataset(args.chessgames, history=args.history, sample_size=256, sub_sampling=64)
    dataloader = DataLoader(dataset, shuffle=False, num_workers=1, collate_fn=collate_boards)

    model = None
    if args.resume is not None:
        model = Chessformer.load_from_checkpoint(args.resume)
    else:
        model = Chessformer(dim=1024, depth=24, heads=16, mlp_dim=4096).cuda()

    # Other types of models
    #model = Chessformer(dim=768, depth=12, heads=12, mlp_dim=3072).cuda() # ViT Base
    #model = Chessformer(dim=1024, depth=24, heads=16, mlp_dim=4096).cuda() # ViT Large
    #model = Chessformer(dim=1280, depth=32, heads=16, mlp_dim=5120).cuda() # ViT Huge

    checkpointer = ChessformerCheckpointer(100, args.checkpoint_dir)

    trainer = pl.Trainer(callbacks=[checkpointer],
                    max_epochs=20000, gpus=2, accelerator='dp', precision=16,
                    accumulate_grad_batches=1, gradient_clip_val=1.0)
    trainer.fit(model, dataloader)

