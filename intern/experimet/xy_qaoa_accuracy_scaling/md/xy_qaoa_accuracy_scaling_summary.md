# Current OpenQARP FM-XY-QAOA: accuracy and qubit scaling

All entries are executed, noise-free OpenQARP/`qarpx` full-state simulations. The mixer is the current complete-graph within-block RXX/RYY mixer and the initial state is uniform over the One-Hot feasible subspace.

`p=2, 49 candidates` is not a continuous p=2 optimizer: it is the project's current fixed-seed random candidate policy. Thus it tests the current implementation as-is, not the best attainable p=2 performance.

## BB1_materials

| N | setting | P(opt) | expected gap | feasibility | runtime (s) |
| ---: | --- | ---: | ---: | ---: | ---: |
| 8 | p=1, 5x5 grid | 4.971492% | 3.231846 | 100.000000% | 0.004 |
| 8 | p=1, 7x7 grid | 1.319089% | 3.225933 | 100.000000% | 0.003 |
| 8 | p=2, 49 candidates | 13.233337% | 3.264326 | 100.000000% | 0.005 |
| 12 | p=1, 5x5 grid | 0.444124% | 4.823864 | 100.000000% | 0.019 |
| 12 | p=1, 7x7 grid | 0.817476% | 4.799963 | 100.000000% | 0.033 |
| 12 | p=2, 49 candidates | 13.132232% | 3.562235 | 100.000000% | 0.062 |
| 16 | p=1, 5x5 grid | 0.058068% | 7.641708 | 100.000000% | 0.472 |
| 16 | p=1, 7x7 grid | 0.058068% | 7.641708 | 100.000000% | 0.887 |
| 16 | p=2, 49 candidates | 5.697042% | 5.378191 | 100.000000% | 1.783 |
| 20 | p=1, 5x5 grid | 0.025117% | 8.825727 | 100.000000% | 15.911 |
| 20 | p=1, 7x7 grid | 0.054242% | 8.798265 | 100.000000% | 35.749 |
| 20 | p=2, 49 candidates | 1.681883% | 7.299086 | 100.000000% | 58.066 |

## BB2_random

| N | setting | P(opt) | expected gap | feasibility | runtime (s) |
| ---: | --- | ---: | ---: | ---: | ---: |
| 8 | p=1, 5x5 grid | 15.561866% | 1.408557 | 100.000000% | 0.006 |
| 8 | p=1, 7x7 grid | 15.561866% | 1.408557 | 100.000000% | 0.005 |
| 8 | p=2, 49 candidates | 12.186763% | 1.513892 | 100.000000% | 0.012 |
| 12 | p=1, 5x5 grid | 1.331734% | 1.713983 | 100.000000% | 0.039 |
| 12 | p=1, 7x7 grid | 1.331734% | 1.713983 | 100.000000% | 0.082 |
| 12 | p=2, 49 candidates | 2.367065% | 2.112984 | 100.000000% | 0.144 |
| 16 | p=1, 5x5 grid | 0.551174% | 5.015342 | 100.000000% | 0.860 |
| 16 | p=1, 7x7 grid | 0.450258% | 4.975297 | 100.000000% | 1.914 |
| 16 | p=2, 49 candidates | 0.213104% | 4.567216 | 100.000000% | 2.994 |
| 20 | p=1, 5x5 grid | 0.103141% | 6.344045 | 100.000000% | 18.883 |
| 20 | p=1, 7x7 grid | 0.103141% | 6.344045 | 100.000000% | 36.973 |
| 20 | p=2, 49 candidates | 0.180512% | 4.872895 | 100.000000% | 57.986 |

## Scope

The reported scaling is implementation/runtime scaling for this Mac and this full-state backend. It is not hardware scaling, a proof of quantum advantage, or a claim about other encodings. All comparisons at a fixed N use the same generated instance; cross-N objective magnitudes are not compared because the generated QUBO changes with N.

The full statevector is retained even though the XY mixer preserves the One-Hot subspace. Therefore the memory curve is `16 * 2^N` bytes for a complex128 statevector; the feasible-space dimension is only `4^(N/4)`.
