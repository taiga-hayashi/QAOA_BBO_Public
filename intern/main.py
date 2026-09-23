"""
BBO-QAOA エントリポイント (ラッパー)
本体スクリプトは src/main.py に集約されています。
"""
import sys
import os

# プロジェクトルートと src ディレクトリを検索パスに追加
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(BASE_DIR, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)
if BASE_DIR not in sys.path:
    sys.path.insert(1, BASE_DIR)

if __name__ == "__main__":
    # Keep the original main-project execution unchanged.  The experiment
    # workflow is an extension of this entry point, not a separate project.
    if "--experiments" in sys.argv:
        sys.argv.remove("--experiments")
        from run_all_experiments import main as run_experiments
        run_experiments()
    else:
        from src.main import main
        main()
