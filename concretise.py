import json
import numpy as np

with open("coefficients.json") as file:
    weights = np.array(json.load(file)[0])

magnitude = np.abs(weights)
sign = np.sign(weights)
scale = 8 / np.max(magnitude)

weights = np.round(scale * magnitude) * sign
json.dump(weights.tolist(), open("concrete-weights2.json", "w"))
