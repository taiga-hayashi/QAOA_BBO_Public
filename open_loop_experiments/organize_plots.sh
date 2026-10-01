#!/bin/bash
cd "$(dirname "$0")/open_loop"

for dir in 1_perfect_fm 2_lambda_tuning 3_fm_parity 4_fm_learning_curve 5_n200_test 6_xy_p_layers; do
    mkdir -p "$dir/png" "$dir/pdf" "$dir/md" "$dir/csv"
done
mkdir -p scripts

# scripts
mv py/*.py scripts/ 2>/dev/null || true

# 2_lambda_tuning
mv png/lambda_tuning*.png 2_lambda_tuning/png/ 2>/dev/null || true
mv pdf/lambda_tuning*.pdf 2_lambda_tuning/pdf/ 2>/dev/null || true
mv md/lambda_tuning*.md 2_lambda_tuning/md/ 2>/dev/null || true
# Move any existing ones that were already put directly in 2_lambda_tuning/
mv 2_lambda_tuning/*.png 2_lambda_tuning/png/ 2>/dev/null || true
mv 2_lambda_tuning/*.pdf 2_lambda_tuning/pdf/ 2>/dev/null || true
mv 2_lambda_tuning/*.md 2_lambda_tuning/md/ 2>/dev/null || true

# 3_fm_parity
mv png/fm_parity*.png 3_fm_parity/png/ 2>/dev/null || true
mv pdf/fm_parity*.pdf 3_fm_parity/pdf/ 2>/dev/null || true
mv 3_fm_parity/*.png 3_fm_parity/png/ 2>/dev/null || true
mv 3_fm_parity/*.pdf 3_fm_parity/pdf/ 2>/dev/null || true

# 4_fm_learning_curve
mv png/fm_learning_curve*.png 4_fm_learning_curve/png/ 2>/dev/null || true
mv pdf/fm_learning_curve*.pdf 4_fm_learning_curve/pdf/ 2>/dev/null || true
mv 4_fm_learning_curve/*.png 4_fm_learning_curve/png/ 2>/dev/null || true
mv 4_fm_learning_curve/*.pdf 4_fm_learning_curve/pdf/ 2>/dev/null || true

# 5_n200_test
mv md/*N200*.md 5_n200_test/md/ 2>/dev/null || true
mv md/*N200*.csv 5_n200_test/csv/ 2>/dev/null || true
mv 5_n200_test/*.md 5_n200_test/md/ 2>/dev/null || true
mv 5_n200_test/*.csv 5_n200_test/csv/ 2>/dev/null || true

# 6_xy_p_layers
mv png/xy_p_layers*.png 6_xy_p_layers/png/ 2>/dev/null || true
mv pdf/xy_p_layers*.pdf 6_xy_p_layers/pdf/ 2>/dev/null || true
mv 6_xy_p_layers/*.png 6_xy_p_layers/png/ 2>/dev/null || true
mv 6_xy_p_layers/*.pdf 6_xy_p_layers/pdf/ 2>/dev/null || true

# Clean up empty direct dirs
rmdir png pdf md py 2>/dev/null || true

echo "Existing plots organized into themed folders with subfolders by extension."
