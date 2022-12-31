import chess
import chess.engine
import fhebot
import json
from multiprocessing import Pool
from tqdm import tqdm
from sklearn.linear_model import LogisticRegression

ITERATIONS = 5
GAMES = 1000
GAME_LENGTH = 80


def stockfish_evaluate():
    pass


def generate_positions(weights):
    stockfish = chess.engine.SimpleEngine.popen_uci("stockfish")
    positions = []
    evaluations = []
    board = chess.Board()
    for i in range(GAME_LENGTH):
        board.push(fhebot.nextmove(weights, board, 2).move)
        positions.append(fhebot.serialise_position(board))
        info = stockfish.analyse(board, chess.engine.Limit(time=0.1))
        evaluations.append(info["score"].white().score(mate_score=1000))

    return positions, evaluations


def main():
    with open("initial-weights.json") as file:
        weights = json.load(file)

    positions = []
    evaluations = []

    for iteration in range(ITERATIONS):
        with Pool(8) as pool:
            for p, e in tqdm(
                pool.imap_unordered(generate_positions, [weights] * GAMES), total=GAMES
            ):
                positions += p
                evaluations += e

        model = LogisticRegression()
        model = model.fit(positions, [1 if e >= 0 else 0 for e in evaluations])
        print(
            f"Iteration {iteration + 1}",
            model.score(positions, [1 if e >= 0 else 0 for e in evaluations]),
        )
        weights = model.coef_[0]
        with open(f"weights-{iteration}.json", "w") as file:
            json.dump(weights.tolist(), file)


if __name__ == "__main__":
    main()
