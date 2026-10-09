import os
import copy
import math
import random
import argparse
from datetime import timedelta

import numpy as np
import torch
import torch.nn as nn
from tqdm.auto import trange

import RNA
from amplify import VariableGenerator, solve, Poly, AmplifyAEClient

def parse_args():
    parser = argparse.ArgumentParser(
        description="FMQA for RNA inverse folding"
    )
    parser.add_argument(
        "--encoding",
        type=str,
        required=True,
        choices=["one-hot", "domain-wall", "binary", "unary"],
        help="Binary-integer encoding method",
    )
    parser.add_argument(
        "--base_allocation",
        type=str,
        required=True,
        help="Integer-to-nucleotide assignment, e.g. AUGC",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed",
    )
    parser.add_argument(
        "--target_structure",
        type=str,
        default="..((((((((.....)).))))))..",
        help="Target RNA secondary structure in dot-bracket notation",
    )
    return parser.parse_args()


RNA.cvar.pf_scale = 0.0
ensemble_defect_cache = {}


def black_box_ensemble_defect(binary_sequence, encoding, target_structure, base_allocation):
    key = (tuple(int(b) for b in binary_sequence), encoding, target_structure, base_allocation)
    if key in ensemble_defect_cache:
        return ensemble_defect_cache[key]

    integer_sequence = binary_to_integer(binary_sequence, encoding)
    if any(i < 0 or i > 3 for i in integer_sequence):
        ensemble_defect_cache[key] = 1.0
        return 1.0

    base_sequence = integer_to_base(integer_sequence, base_allocation)

    md = RNA.md()
    md.temperature = 37.0
    md.dangles = 2
    fc = RNA.fold_compound(base_sequence, md)

    try:
        _, mfe = fc.mfe()
        fc.exp_params_rescale(mfe)
        fc.pf()
        ed = float(fc.ensemble_defect(target_structure))
        ed = min(1.0, max(0.0, ed))
        fitness = ed
    except Exception:
        fitness = 1.0

    ensemble_defect_cache[key] = fitness
    return fitness

def integer_to_binary(integer_sequence, encoding):
    binary_sequence = []

    if encoding == "one-hot":
        for integer in integer_sequence:
            binary_sequence.extend([1 if i == integer else 0 for i in range(4)])

    elif encoding == "domain-wall":
        for integer in integer_sequence:
            binary_sequence.extend([1 if i < integer else 0 for i in range(3)])

    elif encoding == "unary":
        for integer in integer_sequence:
            bits = [1] * integer + [0] * (3 - integer)
            random.shuffle(bits)
            binary_sequence.extend(bits)

    elif encoding == "binary":
        for integer in integer_sequence:
            bits = format(integer, "02b")
            binary_sequence.extend([int(b) for b in bits[::-1]])

    else:
        raise ValueError(f"Unknown encoding: {encoding}")

    return binary_sequence


def binary_to_integer(binary_sequence, encoding):
    integer_sequence = []

    if encoding == "one-hot":
        for b in range(0, len(binary_sequence), 4):
            bit_seq = [int(binary_sequence[i]) for i in range(b, b + 4)]
            if bit_seq.count(1) != 1:
                integer_sequence.append(4)
            else:
                integer_sequence.append(bit_seq.index(1))

    elif encoding == "domain-wall":
        for b in range(0, len(binary_sequence), 3):
            bit_seq = [int(binary_sequence[i]) for i in range(b, b + 3)]
            count_ones = sum(bit_seq)
            if bit_seq != [1] * count_ones + [0] * (len(bit_seq) - count_ones):
                integer_sequence.append(4)
            else:
                integer_sequence.append(count_ones)

    elif encoding == "unary":
        for b in range(0, len(binary_sequence), 3):
            bit_seq = [int(binary_sequence[i]) for i in range(b, b + 3)]
            integer_sequence.append(sum(bit_seq))

    elif encoding == "binary":
        for b in range(0, len(binary_sequence), 2):
            bit_seq = [int(binary_sequence[i]) for i in range(b, b + 2)]
            bit_str = "".join(str(bb) for bb in bit_seq[::-1])
            integer_sequence.append(int(bit_str, 2))

    else:
        raise ValueError(f"Unknown encoding: {encoding}")

    return integer_sequence


def integer_to_base(integer_sequence, base_allocation):
    return "".join(base_allocation[integer] for integer in integer_sequence)


def mk_encoding_constraint(x_list, encoding, n_binary):
    encoding_constraint = 0

    if encoding == "one-hot":
        for b in range(0, n_binary, 4):
            encoding_constraint += (sum(x_list[i] for i in range(b, b + 4)) - 1) ** 2

    elif encoding == "domain-wall":
        for b in range(0, n_binary, 3):
            encoding_constraint += sum(x_list[i + 1] * (1 - x_list[i]) for i in range(b, b + 2))

    elif encoding in ["binary", "unary"]:
        encoding_constraint += 0

    else:
        raise ValueError(f"Unknown encoding: {encoding}")

    return encoding_constraint


def get_uniform_scale_for_linear(num_bits, input_var, labels_var):
    return math.sqrt(3.0 * labels_var / (2.0 * num_bits * input_var))


def get_uniform_scale_for_quad(num_edges, k, input_mean, input_var, labels_var):
    return (
        9.0 * labels_var
        / (2.0 * k * num_edges * input_var * (input_var + 2.0 * input_mean * input_mean))
    ) ** 0.25


def compute_init_scales(x_np: np.ndarray, y_np: np.ndarray, k: int):
    num_bits = x_np.shape[1]
    num_edges = num_bits * (num_bits - 1) / 2.0

    input_mean = float(x_np.mean())
    input_var = float(x_np.var())
    labels_var = float(y_np.var())

    if input_var == 0 or labels_var == 0 or num_edges == 0:
        return 0.01, 0.01

    scale_w = get_uniform_scale_for_linear(num_bits, input_var, labels_var)
    scale_v = get_uniform_scale_for_quad(num_edges, float(k), input_mean, input_var, labels_var)
    return scale_w, scale_v

class TorchFM(nn.Module):
    def __init__(self, d: int, k: int, scale_w: float = 0.1, scale_v: float = 0.1):
        super().__init__()
        self.d = d
        self.k = k
        self.w = nn.Parameter(torch.empty(d).uniform_(-scale_w, scale_w))
        self.v = nn.Parameter(torch.empty(d, k).uniform_(-scale_v, scale_v))
        self.w0 = nn.Parameter(torch.zeros(()))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out_linear = torch.matmul(x, self.w) + self.w0
        out_1 = torch.matmul(x, self.v).pow(2).sum(1)
        out_2 = torch.matmul(x.pow(2), self.v.pow(2)).sum(1)
        out_quadratic = 0.5 * (out_1 - out_2)
        return out_linear + out_quadratic

    def get_parameters(self):
        np_v = self.v.detach().cpu().numpy().copy()
        np_w = self.w.detach().cpu().numpy().copy()
        np_w0 = self.w0.detach().cpu().numpy().copy()
        return np_v, np_w, float(np_w0)


def build_optimizer(model: TorchFM, base_lr: float):
    return torch.optim.AdamW(
        [{"params": [model.v, model.w, model.w0]}],
        lr=base_lr,
        weight_decay=0.01,
    )


def train_fm(
    x_np: np.ndarray,
    y_np: np.ndarray,
    model: TorchFM,
    optimizer: torch.optim.Optimizer,
    epochs: int = 1000,
    patience: int | None = None,
) -> None:
    X = torch.from_numpy(x_np).float()
    Y = torch.from_numpy(y_np).float()

    loss_fn = nn.MSELoss()
    best_state = copy.deepcopy(model.state_dict())
    best_loss = float("inf")
    stall = 0

    for _ in range(epochs):
        model.train()
        optimizer.zero_grad()
        pred = model(X)
        loss = loss_fn(pred, Y)
        loss.backward()
        optimizer.step()

        cur_loss = loss.item()
        if cur_loss < best_loss - 1e-9:
            best_loss = cur_loss
            best_state = copy.deepcopy(model.state_dict())
            stall = 0
        else:
            stall += 1
            if patience is not None and stall >= patience:
                break

    model.load_state_dict(best_state)


def build_client(time_limit_ms: int):
    token = os.environ.get("AMPLIFY_TOKEN")
    if token is None:
        raise ValueError(
            "AMPLIFY_TOKEN is not set. Please set it as an environment variable."
        )

    client = AmplifyAEClient()
    client.token = token
    client.solver = "Qubo"
    client.parameters.time_limit_ms = timedelta(milliseconds=time_limit_ms)
    return client


def anneal(torch_model: TorchFM, encoding, penalty_coeff, client) -> np.ndarray:
    gen = VariableGenerator()
    x = gen.array("Binary", torch_model.d)

    v, w, w0 = torch_model.get_parameters()

    out_linear = w0 + (x * w).sum()
    out_1 = ((x[:, np.newaxis] * v).sum(axis=0) ** 2).sum()  # type: ignore
    out_2 = ((x[:, np.newaxis] * v) ** 2).sum()
    original_objective: Poly = out_linear + (out_1 - out_2) / 2

    vals = [abs(val) for val in original_objective.as_dict().values()]
    max_coeff = max(vals) if len(vals) > 0 else 1.0
    if max_coeff == 0:
        max_coeff = 1.0

    objective = 0
    for pair, coeff in original_objective.as_dict().items():
        if len(pair) == 0:
            objective += coeff / max_coeff
        elif len(pair) == 1:
            objective += x[pair[0]] * coeff / max_coeff
        else:
            objective += x[pair[0]] * x[pair[1]] * coeff / max_coeff

    encoding_constraint = mk_encoding_constraint(x, encoding, torch_model.d)
    qubo_model = objective + penalty_coeff * encoding_constraint

    result = solve(qubo_model, client)
    solution = x.evaluate(result.best.values)

    return np.array(solution, dtype=np.int8)


def main():
    args = parse_args()

    if len(args.base_allocation) != 4:
        raise ValueError("--base_allocation must be a string of length 4, e.g. AUGC")

    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    client = build_client(time_limit_ms=2000)

    target_structure = args.target_structure
    n = len(target_structure)

    temp_integer_sequence = [0 for _ in range(n)]
    temp_binary_sequence = integer_to_binary(temp_integer_sequence, args.encoding)
    n_binary = len(temp_binary_sequence)

    rng = np.random.default_rng(seed=args.seed)

    # initial random dataset
    initial_binary_sequence_list = []
    for _ in range(10):
        initial_integer_sequence = rng.integers(low=0, high=4, size=n, dtype=np.int8)
        initial_binary_sequence = integer_to_binary(initial_integer_sequence, args.encoding)
        initial_binary_sequence_list.append(initial_binary_sequence)

    initial_binary_sequence_array = np.array(initial_binary_sequence_list, dtype=np.int8)

    initial_energy_list = []
    for initial_binary_sequence in initial_binary_sequence_list:
        energy = black_box_ensemble_defect(
            initial_binary_sequence,
            args.encoding,
            target_structure,
            args.base_allocation,
        )
        initial_energy_list.append(energy)

    initial_energy_array = np.array(initial_energy_list, dtype=float)

    scale_w, scale_v = compute_init_scales(initial_binary_sequence_array, initial_energy_array, k=12)
    model = TorchFM(n_binary, k=12, scale_w=scale_w, scale_v=scale_v)

    best_solution = [float(np.min(initial_energy_array))]
    best_solution_sequence = [
        initial_binary_sequence_array[np.argmin(initial_energy_array)].tolist()
    ]

    binary_sequence_list = copy.deepcopy(initial_binary_sequence_array.tolist())
    energy_list = copy.deepcopy(initial_energy_array.tolist())

    x = initial_binary_sequence_array
    y = initial_energy_array

    optimizer = build_optimizer(model, base_lr=0.01)

    for i in trange(1000):
        train_fm(
            x_np=x,
            y_np=y,
            model=model,
            optimizer=optimizer,
            epochs=1000,
            patience=50,
        )

        binary_sequence = anneal(
            torch_model=model,
            encoding=args.encoding,
            penalty_coeff=2.0,
            client=client,
        )

        energy = black_box_ensemble_defect(
            binary_sequence,
            args.encoding,
            target_structure,
            args.base_allocation,
        )

        x = np.vstack((x, binary_sequence))
        y = np.append(y, energy)

        energy_list.append(float(energy))
        binary_sequence_list.append(binary_sequence.tolist())

        if energy < best_solution[-1]:
            best_solution.append(float(energy))
            best_solution_sequence.append(binary_sequence.tolist())
        else:
            best_solution.append(best_solution[-1])
            best_solution_sequence.append(best_solution_sequence[-1])

        print(f"FMQA cycle {i}: found y = {energy:.6f}; current best = {best_solution[-1]:.6f}")

    print("\n=== Finished ===")
    print(f"encoding           : {args.encoding}")
    print(f"base_allocation    : {args.base_allocation}")
    print(f"target_structure   : {args.target_structure}")
    print(f"best_value         : {best_solution[-1]:.6f}")

if __name__ == "__main__":
    main()