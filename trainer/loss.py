import torch
import torch.nn as nn
import torch.nn.functional as F


class ChessformerLoss(nn.Module):

    def __init__(self, representation="chessformer"):
        super().__init__()

        self.representation = representation

        self.cross_entropy_loss = nn.BCEWithLogitsLoss()
        self.mse_loss = nn.MSELoss()

    def forward(self, output, target_value, target_moves, num_moves=4228):

        target_moves = target_moves.squeeze()
        target_value = target_value.squeeze()

        value_pred = output[:,-1]
        moves_pred = output[:, :num_moves]
        batch_size = moves_pred.shape[0]

        loss_moves = self.cross_entropy_loss(moves_pred, target_moves)
        loss_value = self.mse_loss(torch.tanh(value_pred), target_value)

        return loss_moves + loss_value




