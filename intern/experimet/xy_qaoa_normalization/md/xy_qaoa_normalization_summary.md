# XY-QAOA QUBO-coefficient normalization sensitivity

Each QAOA cost Hamiltonian used either the raw QUBO, Q/max|Q_ij|, or Q/RMS(nonzero Q_ij). Every reported P(opt) and expected gap is evaluated afterwards using the same original, unnormalized QUBO. A positive global scaling does not change the exact discrete optimum; it can change this finite angle search's state distribution.

All entries are noise-free OpenQARP/`qarpx` full-state simulations with the current complete-graph within-block RXX/RYY mixer and One-Hot initial state. `p=2` means the present fixed-seed 49-candidate policy, not a continuously optimized p=2 circuit.

## BB1_materials

### p=1, 7x7 grid

| N | normalization | scale | P(opt) | raw expected gap | feasibility | runtime (s) |
| ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 8 | No normalization | 1.000000 | 1.319089% | 3.225933 | 100.000000% | 0.004 |
| 8 | max-abs(Q_ij) = 1 | 5.522577 | 0.686009% | 3.337004 | 100.000000% | 0.002 |
| 8 | RMS(nonzero Q_ij) = 1 | 2.456131 | 1.578742% | 3.223657 | 100.000000% | 0.002 |
| 12 | No normalization | 1.000000 | 0.817476% | 4.799963 | 100.000000% | 0.030 |
| 12 | max-abs(Q_ij) = 1 | 5.522577 | 0.730619% | 4.795129 | 100.000000% | 0.029 |
| 12 | RMS(nonzero Q_ij) = 1 | 1.574556 | 0.633977% | 4.796829 | 100.000000% | 0.029 |
| 16 | No normalization | 1.000000 | 0.058068% | 7.641708 | 100.000000% | 0.723 |
| 16 | max-abs(Q_ij) = 1 | 5.263160 | 0.365681% | 8.529386 | 100.000000% | 0.720 |
| 16 | RMS(nonzero Q_ij) = 1 | 1.720026 | 0.026659% | 8.381782 | 100.000000% | 0.725 |
| 20 | No normalization | 1.000000 | 0.054242% | 8.798265 | 100.000000% | 28.728 |
| 20 | max-abs(Q_ij) = 1 | 5.253386 | 0.057275% | 8.781698 | 100.000000% | 30.329 |
| 20 | RMS(nonzero Q_ij) = 1 | 1.242054 | 0.055075% | 8.793678 | 100.000000% | 22.865 |

### p=2, 49 candidates

| N | normalization | scale | P(opt) | raw expected gap | feasibility | runtime (s) |
| ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 8 | No normalization | 1.000000 | 13.233337% | 3.264326 | 100.000000% | 0.004 |
| 8 | max-abs(Q_ij) = 1 | 5.522577 | 10.823417% | 3.230757 | 100.000000% | 0.004 |
| 8 | RMS(nonzero Q_ij) = 1 | 2.456131 | 18.222292% | 3.127452 | 100.000000% | 0.004 |
| 12 | No normalization | 1.000000 | 13.132232% | 3.562235 | 100.000000% | 0.055 |
| 12 | max-abs(Q_ij) = 1 | 5.522577 | 7.148672% | 4.290246 | 100.000000% | 0.055 |
| 12 | RMS(nonzero Q_ij) = 1 | 1.574556 | 11.924448% | 3.596343 | 100.000000% | 0.055 |
| 16 | No normalization | 1.000000 | 5.697042% | 5.378191 | 100.000000% | 1.365 |
| 16 | max-abs(Q_ij) = 1 | 5.263160 | 2.762314% | 6.220395 | 100.000000% | 1.363 |
| 16 | RMS(nonzero Q_ij) = 1 | 1.720026 | 4.712048% | 5.361152 | 100.000000% | 1.340 |
| 20 | No normalization | 1.000000 | 1.681883% | 7.299086 | 100.000000% | 40.068 |
| 20 | max-abs(Q_ij) = 1 | 5.253386 | 0.250772% | 8.694763 | 100.000000% | 39.698 |
| 20 | RMS(nonzero Q_ij) = 1 | 1.242054 | 1.316466% | 7.409133 | 100.000000% | 46.963 |

## BB2_random

### p=1, 7x7 grid

| N | normalization | scale | P(opt) | raw expected gap | feasibility | runtime (s) |
| ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 8 | No normalization | 1.000000 | 15.561866% | 1.408557 | 100.000000% | 0.005 |
| 8 | max-abs(Q_ij) = 1 | 1.346657 | 15.768252% | 1.393761 | 100.000000% | 0.003 |
| 8 | RMS(nonzero Q_ij) = 1 | 0.672215 | 15.166568% | 1.436917 | 100.000000% | 0.003 |
| 12 | No normalization | 1.000000 | 1.331734% | 1.713983 | 100.000000% | 0.044 |
| 12 | max-abs(Q_ij) = 1 | 1.483434 | 1.286132% | 1.711688 | 100.000000% | 0.045 |
| 12 | RMS(nonzero Q_ij) = 1 | 0.688812 | 1.393963% | 1.720020 | 100.000000% | 0.044 |
| 16 | No normalization | 1.000000 | 0.450258% | 4.975297 | 100.000000% | 1.354 |
| 16 | max-abs(Q_ij) = 1 | 1.405712 | 0.490178% | 4.926179 | 100.000000% | 1.252 |
| 16 | RMS(nonzero Q_ij) = 1 | 0.634692 | 0.378325% | 5.074008 | 100.000000% | 1.201 |
| 20 | No normalization | 1.000000 | 0.103141% | 6.344045 | 100.000000% | 31.520 |
| 20 | max-abs(Q_ij) = 1 | 1.274081 | 0.091871% | 6.517442 | 100.000000% | 27.960 |
| 20 | RMS(nonzero Q_ij) = 1 | 0.619265 | 0.075449% | 6.253501 | 100.000000% | 32.935 |

### p=2, 49 candidates

| N | normalization | scale | P(opt) | raw expected gap | feasibility | runtime (s) |
| ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 8 | No normalization | 1.000000 | 12.186763% | 1.513892 | 100.000000% | 0.006 |
| 8 | max-abs(Q_ij) = 1 | 1.346657 | 13.163133% | 1.549777 | 100.000000% | 0.007 |
| 8 | RMS(nonzero Q_ij) = 1 | 0.672215 | 9.870094% | 1.673423 | 100.000000% | 0.007 |
| 12 | No normalization | 1.000000 | 2.367065% | 2.112984 | 100.000000% | 0.087 |
| 12 | max-abs(Q_ij) = 1 | 1.483434 | 2.330132% | 1.940981 | 100.000000% | 0.085 |
| 12 | RMS(nonzero Q_ij) = 1 | 0.688812 | 2.155810% | 2.109579 | 100.000000% | 0.086 |
| 16 | No normalization | 1.000000 | 0.213104% | 4.567216 | 100.000000% | 2.024 |
| 16 | max-abs(Q_ij) = 1 | 1.405712 | 0.135604% | 4.971372 | 100.000000% | 2.025 |
| 16 | RMS(nonzero Q_ij) = 1 | 0.634692 | 0.361310% | 4.076154 | 100.000000% | 2.101 |
| 20 | No normalization | 1.000000 | 0.180512% | 4.872895 | 100.000000% | 56.874 |
| 20 | max-abs(Q_ij) = 1 | 1.274081 | 0.116670% | 5.021622 | 100.000000% | 59.766 |
| 20 | RMS(nonzero Q_ij) = 1 | 0.619265 | 0.265898% | 5.035408 | 100.000000% | 65.090 |

## Interpretation boundary

Compare normalization choices only within the same problem and N. Objective magnitudes across N are not comparable because each N is a separately generated QUBO. This is a parameter-scale sensitivity result for the current classical state-vector implementation, not a hardware result or a general claim for other encodings.
