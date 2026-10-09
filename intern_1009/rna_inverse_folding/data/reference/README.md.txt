# FMQA for RNA Inverse Folding

This repository provides the implementation of the Factorization Machine with Quadratic-Optimization Annealing (FMQA) applied to the RNA inverse folding problem.

## Features
- Binary-integer encoding (one-hot, domain-wall, binary, unary)
- Nucleotide-to-integer assignment
- Ensemble defect-based objective function (ViennaRNA)
- FMQA optimization

## Requirements
- Python 3.12
- ViennaRNA 2.7.2
- amplify >= 1.4.1
- torch >= 2.0


## Setup

Set your Amplify token as an environment variable.

### Mac / Linux

```bash
export AMPLIFY_TOKEN=your_token_here
```


## Usage

Run the script with the required arguments:

```bash
python fmqa_rna.py --encoding one-hot --base_allocation AUGC
```


## Output

The script prints the optimization progress:

```text
FMQA cycle 0: found y = ...
FMQA cycle 1: found y = ...
...
=== Finished ===
encoding        : one-hot
base_allocation : AUGC
target_structure: ..((((((((.....)).))))))..
best_value      : 0.123456
```

## Citation

If you use this package in your work, please cite the following papers ([Shuta Kikuchi, and Shu Tanaka, 10.48550/arXiv.2602.16643](https://arxiv.org/abs/2602.16643))

```
Shuta Kikuchi, and Shu Tanaka, "Factorization Machine with Quadratic-Optimization Annealing for RNA Inverse Folding and Evaluation of Binary-Integer Encoding and Nucleotide Assignment," arXiv, arXiv:2602.16643, 2026, doi: 10.48550/arXiv.2602.16643.
```

```
@article{kikuchi2026factorization,
  title={Factorization Machine with Quadratic-Optimization Annealing for RNA Inverse Folding and Evaluation of Binary-Integer Encoding and Nucleotide Assignment},
  author={Kikuchi, Shuta and Tanaka, Shu},
  journal={arXiv preprint arXiv:2602.16643},
  year={2026},
  doi={10.48550/arXiv.2602.16643}
}
```
