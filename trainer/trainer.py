import os
import sys
import argparse
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
import pytorch_lightning as pl

from models.chessformer import Chessformer
from dataset import ChessDataset, ChessIterableDataset
from checkpointer import ChessformerCheckpointer

def collate_iterate_boards(batch):
    return batch[0] # do nothing

def collate_boards(batch):
    boards_torch = [x[0] for x in batch]
    result_torch = [x[1] for x in batch]
    moves_vector_torch = [x[2] for x in batch]

    return torch.cat(boards_torch, 0), torch.cat(result_torch, 0), torch.cat(moves_vector_torch, 0)

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Chessformer trainer')
    parser.add_argument('--chessgames', type=str, required=True, help='Path of the directory containing chess games')
    parser.add_argument('--resume', type=str, required=False, default=None, help='Resume model from previous checkpoint')
    parser.add_argument('--checkpoint_dir', type=str, required=False, default="checkpoints/", help='Save checkpoints in this directory')
    parser.add_argument('--history', type=int, default=-1, help='History to use for training')
    parser.add_argument('--max_epochs', type=int, default=200, help='Maximum number of epochs')
    args = parser.parse_args()

    dataset = ChessDataset(args.chessgames, history=args.history, sub_sampling=32)
    dataloader = DataLoader(dataset, shuffle=True, num_workers=4, collate_fn=collate_boards, batch_size=8)

    model = None
    if args.resume is not None:
        model = Chessformer.load_from_checkpoint(args.resume)
    else:
        model = Chessformer(dim=1280, depth=24, heads=16, mlp_dim=5120).cuda()

    # Other types of models
    #model = Chessformer(dim=768, depth=12, heads=12, mlp_dim=3072).cuda() # ViT Base
    #model = Chessformer(dim=1024, depth=24, heads=16, mlp_dim=4096).cuda() # ViT Large
    #model = Chessformer(dim=1280, depth=32, heads=16, mlp_dim=5120).cuda() # ViT Huge

    checkpointer = ChessformerCheckpointer(100, args.checkpoint_dir)

    trainer = pl.Trainer(callbacks=[checkpointer],
                    max_epochs=args.max_epochs, gpus=3, accelerator='ddp', precision=16,
                    accumulate_grad_batches=8, gradient_clip_val=1.0)

    # Save initial model
    model.save_to_onnx(os.path.join(args.checkpoint_dir, "Chessformer_0_0.onnx"))

    # train the model
    trainer.fit(model, dataloader)

