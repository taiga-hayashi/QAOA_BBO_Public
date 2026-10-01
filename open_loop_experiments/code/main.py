import argparse
from plot_parity import run_parity
from plot_lambda_tuning import run_lambda_tuning
from plot_learning_curve import run_learning_curve
from run_n200_test import run_n200_test

if __name__ == "__main__":
    """Main runner script for all modularized open loop experiments."""
    parser = argparse.ArgumentParser(description="Run modularized open loop experiments with OpenQARP")
    parser.add_argument("--task", type=str, required=True, 
                        choices=["parity", "lambda_tuning", "learning_curve", "n200_test"], 
                        help="Task to run.")
    parser.add_argument("--N", type=int, default=18, help="Problem size N")
    parser.add_argument("--problem", type=str, default="bb1", help="Problem type (bb1, bb2)")
    
    args = parser.parse_args()
    
    if args.task == "parity":
        run_parity(args.N, args.problem)
    elif args.task == "lambda_tuning":
        run_lambda_tuning(args.N, args.problem)
    elif args.task == "learning_curve":
        run_learning_curve(args.N, args.problem)
    elif args.task == "n200_test":
        run_n200_test(args.N, args.problem)
