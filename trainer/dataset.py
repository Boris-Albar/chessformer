import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, IterableDataset
import os, glob
import random

import chess

class ChessDataset(Dataset):
    def __init__(self, path, history=-1, sub_sampling=32):

        self.path = path
        self.history = history
        self.sub_sampling = sub_sampling

        self.file_list = sorted(glob.glob(self.path + '/*.txt'))[max(0, -self.history):]
        if self.history != -1:
            self.file_list = file_list[max(-len(file_list), -self.history):]

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):

        torch_boards = []
        result = []
        moves_vector = []

        choosen_game = self.file_list[idx]
        f = open(choosen_game)
        board_list = f.read().splitlines()
        match_result = board_list[-1]
        board_list = board_list[:-1]

        boards_indices = random.sample(list(range(len(board_list))), min(self.sub_sampling, len(board_list)))
        moves = np.load(choosen_game.replace("boards", "moves").replace(".txt", ".npz"))

        for idx in boards_indices:
            board = board_list[idx]
            board, repeat, fiftyrule = board.replace('"', '').split("--")
            board_chess = chess.Board(board)

            board_numpy = np.zeros((65, 15))

            piece_board = []
            for sq in chess.SQUARES:
                color = board_chess.color_at(sq)
                if color is not None:
                    if color == chess.WHITE:
                        board_numpy[int(sq), board_chess.piece_type_at(sq)] = 1.0 # white piece
                    else:
                        board_numpy[int(sq), 7 + board_chess.piece_type_at(sq)] = 1.0 # white piece

            # append side to move and castle rights, repeat moves
            turn = 0.0 if board_chess.turn == chess.WHITE else 1.0
            board_numpy[64, 0] = turn
            board_numpy[64, 1] = float(board_chess.has_kingside_castling_rights(chess.WHITE))
            board_numpy[64, 2] = float(board_chess.has_queenside_castling_rights(chess.WHITE))
            board_numpy[64, 3] = float(board_chess.has_kingside_castling_rights(chess.BLACK))
            board_numpy[64, 4] = float(board_chess.has_queenside_castling_rights(chess.BLACK))
            board_numpy[64, 5] = float(repeat) / 3.0
            board_numpy[64, 6] = float(fiftyrule) / 100.0

            torch_boards.append(board_numpy)

            if match_result == "WhiteWon":
                result.append(1.0)
            elif match_result == "BlackWon":
                result.append(-1.0)
            else:
                result.append(0.0)

            moves_array = moves[str(idx)]
            moves_array = moves_array / moves_array.sum() # normalize for probabilities
            moves_vector.append(moves_array)

        boards_torch = torch.Tensor(torch_boards).float()
        result_torch = torch.Tensor(result).float()
        moves_vector_torch = torch.Tensor(moves_vector).double()

        return boards_torch, result_torch, moves_vector_torch


class ChessIterableDataset(IterableDataset):

    def __init__(self, path, history=-1, sample_size=4096, sub_sampling=32):

        self.path = path
        self.history = history
        self.sample_size = sample_size
        self.sub_sampling = sub_sampling

    def __iter__(self):
        file_list = sorted(glob.glob(self.path + '/*.txt'))[max(0, -self.history):]
        if self.history != -1:
            file_list = file_list[max(-len(file_list), -self.history):]

        torch_boards = []
        result = []
        moves_vector = []

        while len(result) < self.sample_size:
            choosen_game = random.choice(file_list)
            f = open(choosen_game)
            board_list = f.read().splitlines()
            match_result = board_list[-1]
            board_list = board_list[:-1]

            boards_indices = random.sample(list(range(len(board_list))), min(min(self.sample_size - len(result), self.sub_sampling), len(board_list)))
            moves = np.load(choosen_game.replace("boards", "moves").replace(".txt", ".npz"))

            for idx in boards_indices:
                board = board_list[idx]
                board, repeat, fiftyrule = board.replace('"', '').split("--")
                board_chess = chess.Board(board)

                board_numpy = np.zeros((65, 15))

                piece_board = []
                for sq in chess.SQUARES:
                    color = board_chess.color_at(sq)
                    if color is not None:
                        if color == chess.WHITE:
                            board_numpy[int(sq), board_chess.piece_type_at(sq)] = 1.0 # white piece
                        else:
                            board_numpy[int(sq), 7 + board_chess.piece_type_at(sq)] = 1.0 # white piece

                # append side to move and castle rights, repeat moves
                turn = 0.0 if board_chess.turn == chess.WHITE else 1.0
                board_numpy[64, 0] = turn
                board_numpy[64, 1] = float(board_chess.has_kingside_castling_rights(chess.WHITE))
                board_numpy[64, 2] = float(board_chess.has_queenside_castling_rights(chess.WHITE))
                board_numpy[64, 3] = float(board_chess.has_kingside_castling_rights(chess.BLACK))
                board_numpy[64, 4] = float(board_chess.has_queenside_castling_rights(chess.BLACK))
                board_numpy[64, 5] = float(repeat) / 3.0
                board_numpy[64, 6] = float(fiftyrule) / 100.0

                torch_boards.append(board_numpy)

                if match_result == "WhiteWon":
                    result.append(1.0)
                elif match_result == "BlackWon":
                    result.append(-1.0)
                else:
                    result.append(0.0)

                moves_array = moves[str(idx)]
                moves_array = moves_array / moves_array.sum() # normalize for probabilities
                moves_vector.append(moves_array)

        boards_torch = torch.Tensor(torch_boards).float()
        result_torch = torch.Tensor(result).float()
        moves_vector_torch = torch.Tensor(moves_vector).double()

        yield boards_torch, result_torch, moves_vector_torch

