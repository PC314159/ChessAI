import math

import chess.pgn
import numpy
import torch

from src.model import FENAnalyzerModel

am = FENAnalyzerModel(num_filter_channels=256)
checkpoint = torch.load(
    "../output_models/2025/eval_move_models/v6/Chess_FEN_Analyzer_Model_Round_25.pth")
am.load_state_dict(checkpoint, strict=False)
am = am.cpu()
am.eval()


def move_to_grid(move: chess.Move | None) -> numpy.ndarray:
    if move is None:
        return numpy.full([8, 8], 0, dtype=int)
    fs = chess.SquareSet.from_square(move.from_square)
    fs = numpy.array(fs.tolist()).astype(int)
    ts = chess.SquareSet.from_square(move.to_square)
    ts = numpy.array(ts.tolist()).astype(int)
    return numpy.reshape(ts - fs, [8, 8])


def gamenode_to_data(game):
    board = game.board()
    data_list = []

    # COLORS = [WHITE, BLACK] = [True, False]
    # PIECE_TYPES = [PAWN, KNIGHT, BISHOP, ROOK, QUEEN, KING] = range(1, 7)
    for pc_type in chess.PIECE_TYPES:
        for color in chess.COLORS:
            pcs = board.pieces(piece_type=pc_type, color=color)
            pcs = numpy.array(pcs.tolist()).astype(bool)
            pcs = numpy.reshape(pcs, [8, 8])
            data_list.append(pcs)

    turn = board.turn
    turn_board = numpy.full([8, 8], turn).astype(bool)
    data_list.append(turn_board)

    castling_rights = board.castling_rights
    castling_board = numpy.full([8, 8], False)
    castling_board[0][0] = bool(castling_rights & chess.BB_A8)
    castling_board[0][7] = bool(castling_rights & chess.BB_H8)
    castling_board[7][0] = bool(castling_rights & chess.BB_A1)
    castling_board[7][7] = bool(castling_rights & chess.BB_H1)
    castling_board = castling_board.astype(bool)
    data_list.append(castling_board)

    ep_square = board.ep_square
    legal_en_passant = board.has_legal_en_passant()
    if legal_en_passant:
        ep_ss = chess.SquareSet(ep_square)
        ep_board = numpy.array(ep_ss.tolist()).astype(bool)
        ep_board = numpy.reshape(ep_board, [8, 8])
        data_list.append(ep_board)
    else:
        ep_board = numpy.full([8, 8], False, dtype=bool)
        data_list.append(ep_board)

    prev_move = game.move
    prev_prev_move = game.parent.move if game.parent is not None else None
    data_list.append(move_to_grid(prev_move))
    data_list.append(move_to_grid(prev_prev_move))

    arr = numpy.array([data_list])
    return torch.from_numpy(arr)


def analyze_position(gn):
    values = gamenode_to_data(gn)
    values = values.float().cpu()
    with (torch.no_grad()):
        eval_output, move_output = am(values)
    move_output = move_output.squeeze()
    return eval_output.item(), move_output.cpu().numpy()


class TreeNode:
    def __init__(self, param, gamenode: chess.pgn.GameNode, parent=None):
        self.children = []
        self.parent = parent
        self.param = param
        self.gamenode = gamenode
        self.turn = 1 if gamenode.board().turn else -1
        self.legal_moves = list(gamenode.board().legal_moves)
        self.available_moves = list(gamenode.board().legal_moves)
        self.visits = 0
        self.total_value = 0
        self.score = 0
        self.prob = [0]*(64*64)

        if self.gamenode.board().is_game_over():
            outcome = self.gamenode.board().outcome()
            if outcome.winner is True:
                self.score = 1
            elif outcome.winner is False:
                self.score = -1
            else:
                self.score = 0
        else:
            value, prob = analyze_position(self.gamenode)
            self.score = value
            self.prob = prob

    def step(self):
        self.select()

    def get_best(self):
        return max(self.children, key=lambda c: c.visits)

    def print_tree(self, indent=0):
        pass

    def select(self):
        if self.gamenode.board().is_game_over():
            outcome = self.gamenode.board().outcome()
            score = 0
            if outcome.winner is True:
                score = 1
            elif outcome.winner is False:
                score = -1
            self.backpropagate(score)
            return
        if len(self.available_moves) > 0:
            self.expand()
            return

        cmax = None
        cmax_sc = -2
        for c in self.children:
            m = c.gamenode.move
            mprob = self.prob[m.from_square * 64 + m.to_square]

            usc = self.param * mprob * math.sqrt(self.visits + 1 / (1 + c.visits))
            side_score = self.turn * c.score + usc
            if side_score > cmax_sc:
                cmax = c
                cmax_sc = side_score
        cmax.select()

    def expand(self):
        best_move = max(self.available_moves,
                        key=lambda m: self.prob[m.from_square * 64 + m.to_square])
        self.available_moves.remove(best_move)
        nn = TreeNode(self.param, self.gamenode.add_variation(best_move), self)
        self.children.append(nn)
        nn.rollout()

    def rollout(self):
        self.backpropagate(self.score)

    def backpropagate(self, result):
        self.visits += 1
        self.total_value += result
        self.score = self.total_value / self.visits
        if self.parent is not None:
            self.parent.backpropagate(result)

