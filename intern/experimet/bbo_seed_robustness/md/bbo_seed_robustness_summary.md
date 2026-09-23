# BBO seed robustness validation

Seeds: [11, 22, 33]; cycles: 100. Curves show mean ± sample SD.

| Problem | Method | Mean final best | SD | Optimum hits | Mean candidate failures |
|---|---|---:|---:|---:|---:|
| BB1_materials | FMQA | -15.773572 | 0.000000 | 3/3 | 0.00 |
| BB1_materials | Standard QAOA | -11.219724 | 3.946801 | 1/3 | 70.33 |
| BB1_materials | FM-XY-QAOA | -15.773572 | 0.000000 | 3/3 | 0.00 |
| BB2_random | FMQA | -5.809033 | 0.000000 | 3/3 | 0.00 |
| BB2_random | Standard QAOA | -5.044943 | 1.323443 | 2/3 | 55.67 |
| BB2_random | FM-XY-QAOA | -5.809033 | 0.000000 | 3/3 | 0.00 |

This is a preliminary three-seed check. Each seed changes the initial design and solver sampling stream; more seeds are required for inferential statistics.
