# Adaptive penalty validation

The heuristic penalty was compared with the exact minimum sufficient penalty over 50 random N=16 instances.

- Sufficient instances: 48/50
- Mean heuristic/exact ratio: 2.170
- Minimum ratio: 0.865
- Maximum ratio: 9.583
- Insufficient cases: seed 13 (exact=3.837, heuristic=3.320), seed 30 (exact=3.900, heuristic=3.450)

A ratio above 1 means that the heuristic is sufficient but conservative. This experiment validates sufficiency only for the tested instance family and scale; it does not prove a universal bound.
