import numpy as np
import chess
import chess.engine
import fhebot
import json
from multiprocessing import Pool
from tqdm import tqdm
from sklearn.linear_model import LogisticRegression

ITERATIONS = 10
GAMES = 100
GAME_LENGTH = 40


def stockfish_evaluate():
    pass


def generate_positions(data):
    weights, seed = data
    np.random.seed(seed)
    stockfish = chess.engine.SimpleEngine.popen_uci("stockfish")
    positions = []
    evaluations = []
    board = chess.Board()
    for i in range(GAME_LENGTH):
        evaluation = fhebot.nextmove(weights, board, 2, rng=True)
        if evaluation.move is None:
            break
        board.push(evaluation.move)
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
                pool.imap_unordered(generate_positions, [(weights, np.random.randint(0, 100000)) for _ in range(GAMES)]), total=GAMES
            ):
                positions += p
                evaluations += e

        model = LogisticRegression(max_iter=10000)
        model = model.fit(
            positions,
            [1 if e + np.random.normal(0, 0.001) > 0 else 0 for e in evaluations],
        )
        print(
            f"Iteration {iteration + 1}",
            model.score(positions, [1 if e > 0 else 0 for e in evaluations]),
        )

        weights = 15 * model.coef_[0] / np.max(np.abs(model.coef_[0]))

        with open(f"weights-{iteration}.json", "w") as file:
            json.dump(weights.tolist(), file)

        with open(f"positions.json", "w") as file:
            json.dump(positions, file)

        with open(f"evaluations.json", "w") as file:
            json.dump(evaluations, file)


if __name__ == "__main__":
    main()
