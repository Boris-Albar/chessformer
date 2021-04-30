import torch
import torch.nn as nn
import torch.nn.functional as F


class ChessformerLoss(nn.Module):

    def __init__(self, representation="chessformer", balance_draw_ratio=None):
        super().__init__()

        self.representation = representation
        self.num_moves = 4228
        self.balance_draw_ratio = balance_draw_ratio

        self.cross_entropy_loss = nn.BCEWithLogitsLoss()
        self.mse_loss = nn.MSELoss()

    def forward(self, output, target_value, target_moves):

        target_moves = target_moves.squeeze()
        target_value = target_value.squeeze()

        value_pred = torch.tanh(output[:,-1])
        moves_pred = output[:, :self.num_moves]
        batch_size = moves_pred.shape[0]

        loss_value = None
        loss_moves = None

        if self.balance_draw_ratio is None:
            loss_value = self.mse_loss(value_pred, target_value)
            loss_moves = self.cross_entropy_loss(moves_pred, target_moves)
        else:
            draw_indices = (target_value == 0.0)
            notdraw_indices = (target_value != 0.0)

            #weight_draw = (1.0 / draw_indices.sum()) * self.balance_draw_ratio
            #weight_notdraw = (1.0 / notdraw_indices.sum()) * (1 - self.balance_draw_ratio)

            weight_draw = self.balance_draw_ratio
            weight_notdraw = (1 - self.balance_draw_ratio)

            loss_value_draw = torch.nan_to_num(self.mse_loss(value_pred[draw_indices], target_value[draw_indices]))
            loss_value_notdraw = torch.nan_to_num(self.mse_loss(value_pred[notdraw_indices], target_value[notdraw_indices]))

            loss_moves_draw = torch.nan_to_num(self.cross_entropy_loss(moves_pred[draw_indices], target_moves[draw_indices]))
            loss_moves_notdraw = torch.nan_to_num(self.cross_entropy_loss(moves_pred[notdraw_indices], target_moves[notdraw_indices]))

            loss_value = weight_draw * loss_value_draw + weight_notdraw * loss_value_notdraw
            loss_moves = weight_draw * loss_moves_draw + weight_notdraw * loss_moves_notdraw

        return loss_moves, loss_value




