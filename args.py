import os
import argparse

import torch

from utils import get_hex_time


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-location",
        type=str,
        default=os.path.expanduser("/data"),
        help="The root directory for the datasets.",
    )
    ######## Task vector finetuning and evaluation ########
    parser.add_argument(
        "--eval-datasets",
        default=None,
        type=lambda x: x.split(","),
        help="Which datasets to use for evaluation. Split by comma, e.g. MNIST,EuroSAT ",
    )
    parser.add_argument(
        "--train-dataset",
        default=None,
        type=lambda x: x.split(","),
        help="Which dataset(s) to patch on.",
    )
    parser.add_argument(
        "--exp_name",
        type=str,
        default=None,
        help="Name of the experiment, for organization purposes only.",
    )
    parser.add_argument(
        "--results-db",
        type=str,
        default=None,
        help="Where to store the results, else does not store",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="ViT-B-32",
        help="The type of model (e.g. RN50, ViT-B-32).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=128,
    )
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate.")
    parser.add_argument("--wd", type=float, default=0.1, help="Weight decay")
    parser.add_argument("--ls", type=float, default=0.0, help="Label smoothing.")
    parser.add_argument(
        "--warmup_length",
        type=int,
        default=500,
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
    )
    parser.add_argument(
        "--load",
        type=lambda x: x.split(","),
        default=None,
        help="Optionally load _classifiers_, e.g. a zero shot classifier or probe or ensemble both.",
    )
    parser.add_argument(
        "--save",
        type=str,
        default="/checkpoints/ViT-B-32",
        help="Optionally save a _classifier_, e.g. a zero shot classifier or probe.",
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default=None,
        help="Directory for caching features and encoder",
    )

    ######## MAP config ########

    parser.add_argument(
        "--method",
        type=str,
        help="whether you want to run MAP (map), MAP with nested merging (nested), or MAP with Bayesian updates (bayesian)",
        default="nested",
    )

    parser.add_argument(
        "--pretrained-checkpoint",
        type=str,
        default="/checkpoints/ViT-B-32/zeroshot.pt",
        help="Directory for pretrained backbone checkpoint",
    )
    parser.add_argument(
        "--openclip-cachedir",
        type=str,
        default=".cache/open_clip",
        help="Directory for caching models from OpenCLIP",
    )
    parser.add_argument(
        "--results-path",
        type=str,
        default="nested_experiments4",
        help="Directory for storing results, including pareto front, evaluation results for scalings, etc",
    )
    parser.add_argument(
        "--num-task", type=int, default=2, help="Number of tasks to process"
    )
    parser.add_argument(
        "--model-name", type=str, default=2, help="Number of tasks to process"
    )
    parser.add_argument(
        "--point", type=str, default=2, help="Number of tasks to process"
    )
    parser.add_argument("--models", type=str)
    parser.add_argument(
        "--metric-type",
        type=str,
        default="accuracy",
        help="Metric to use for evaluation",
    )
    parser.add_argument(
        "--exp-id",
        type=str,
        default=get_hex_time(),
        help="Experiment ID for logging",
    )
    parser.add_argument(
        "--zeroshot-eval-datasets",
        nargs="+",
        help="A list of strings to process. Usage: --eval-datasets MNIST CIFAR10",
        default=["SUN397Val", "CarsVal", "DTDVal", "SVHNVal"],
    )
    parser.add_argument(
        "--zeroshot-merge-models",
        nargs="+",
        help="A list of strings to process. Usage: --eval-datasets MNIST CIFAR10",
        default=["SUN397Val", "CarsVal", "DTDVal", "SVHNVal"],
    )
    parser.add_argument(
        "--preference",
        type=str,
        help="A list of strings to process. Usage: --eval-datasets MNIST CIFAR10",
        default="example_preference.yaml",
    )

    parser.add_argument(
        "--method",
        type=str,
        help="whether you want to run MAP (map), MAP with nested merging (nested), or MAP with Bayesian updates (bayesian)",
        default="nested",
    )
    parser.add_argument(
        "--bayes-iter", type=int, help="Number of iterations for Bayesian updates"
    )
    parser.add_argument(
        "--bayes-update-pts",
        type=int,
        help="Number of points to update in Bayesian updates",
    )
    parser.add_argument(
        "--bayes-initial-pts",
        type=int,
        help="Number of initial points to sample in Bayesian updates",
    )
    # parser.add_argument('model', type=str, default="ViT-B-32")
    parsed_args = parser.parse_args()
    parsed_args.device = "cuda" if torch.cuda.is_available() else "cpu"

    if parsed_args.load is not None and len(parsed_args.load) == 1:
        parsed_args.load = parsed_args.load[0]
    return parsed_args
