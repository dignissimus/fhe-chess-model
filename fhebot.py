import numpy as np
import chess
import json
from collections import namedtuple

Evaluation = namedtuple("Evaluation", ["move", "score"])

MULTIPLIER = {chess.WHITE: 1, chess.BLACK: -1}


def serialise_position(board):
    # Strcture
    # Black bitboards then white bitboards
    # Pawn, Knight, Bishop, Rook, Queen, King
    # Rank then file (a1, b1, c1, ...)
    # Turn then castling (kqKQ)
    serialised_board = np.zeros(64 * 12)
    extra = np.zeros(1)

    for square in chess.SQUARES:
        file = chess.square_file(square)
        rank = chess.square_rank(square)
        piece = board.piece_at(square)
        if piece is None:
            continue

        offset = piece.color * 64 * 6
        offset += (piece.piece_type - 1) * 64
        offset += rank * 8
        offset += file
        serialised_board[offset] = 1

    if board.turn == chess.WHITE:
        extra[0] = 1

    return serialised_board.tolist() + extra.tolist()


def dot(xs, ys):
    return sum([x * y for x, y in zip(xs, ys)])


def evaluate(weights, position):
    return np.dot(weights, position)


def nextmove(weights, position, depth=2, rng=False, alpha=-999, beta=999):
    multiplier = MULTIPLIER[position.turn]
    best_move = None
    best_evaluation = -999
    if depth == 1:
        for move in position.legal_moves:
            position.push(move)
            evaluation = evaluate(weights, serialise_position(position)) * multiplier
            if rng:
                evaluation += np.random.normal(0, 2)
            position.pop()
            if evaluation > best_evaluation:
                best_evaluation = evaluation
                best_move = move
        return Evaluation(best_move, best_evaluation)

    for move in position.legal_moves:
        position.push(move)
        evaluation = nextmove(weights, position, depth - 1, rng, -beta, -alpha)
        position.pop()

        if -evaluation.score > best_evaluation:
            best_evaluation = -evaluation.score
            best_move = move
        if best_evaluation > alpha:
            alpha = best_evaluation
        if alpha > beta:
            return Evaluation(best_move, best_evaluation)

    return Evaluation(best_move, best_evaluation)


def main():
    with open("weights.json") as file:
        weights = np.array(json.load(file))

    board = chess.Board()
    for i in range(10):
        evaluation = nextmove(weights, board, 2, True)
        print(board.san(evaluation.move), evaluation.score)
        board.push(evaluation.move)


if __name__ == "__main__":
    main()
