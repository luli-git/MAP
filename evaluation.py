import os
import sys

import json
import tqdm
from concurrent.futures import ProcessPoolExecutor

import torch
import torch.nn as nn
from src import utils
from src.datasets.common import get_dataloader, maybe_dictionarize
from src.heads import get_classification_head
from src.modeling import ImageClassifier
from abc import ABC, abstractmethod

from src.datasets.registry import get_dataset
from merge import get_merged_model
from utils import read_config


class AbstractEvaluator(ABC):
    def __init__(self, model, args):
        """
        Initialize the AbstractEvaluator with an image encoder and argsuration arguments.
        :param image_encoder: The base encoder model to which task-specific vectors are applied.
        :param args: Configuration arguments including data locations, device, etc.
        """
        self.model = model
        self.args = args
        self.results = []

    @abstractmethod
    def eval_single_dataset(self, dataset_name):
        """
        Evaluates the image encoder model on a single dataset and returns metrics.

        Should return: {"metric_type": numeric_metric_result} e.g. {"accuracy": 0.95, "loss": 0.1}
        """
        pass

    @abstractmethod
    def evaluate(self, scaling):
        """
        Evaluates the image encoder on multiple datasets specified in the args along with scaling factors.

        Should return: a dictionary, whose key is the dataset name that the merged model is evaluated on,
        and the results for that dataset obtained from eval_single_dataset.
        results[dataset_name] = dataset_results e.g. {"MNIST": {"accuracy": 0.95, "loss": 0.1}, {"EuroSAT": {"accuracy": 0.85, "loss": 0.2}}
        """
        pass


class ZeroShotEvaluator(AbstractEvaluator):
    def __init__(self, image_encoder, args):
        """
        Initialize the EvaluationManager with an image encoder and argsuration arguments.
        :param image_encoder: The base encoder model to which task-specific vectors are applied.
        :param args: Configuration arguments including data locations, device, etc.
        """
        self.image_encoder = image_encoder
        self.args = args
        self.results = []

    def eval_single_dataset(self, dataset_name):
        """
        Evaluates the image encoder model on a single dataset and returns metrics.
        """
        classification_head = get_classification_head(self.args, dataset_name)
        model = ImageClassifier(self.image_encoder, classification_head)
        model.eval()

        dataset = get_dataset(
            dataset_name,
            model.val_preprocess,
            location=self.args.data_location,
            batch_size=self.args.batch_size,
        )
        dataloader = get_dataloader(dataset, is_train=False, args=self.args)
        device = self.args.device
        criterion = nn.CrossEntropyLoss()
        total_loss, correct, n = 0.0, 0.0, 0.0

        with torch.no_grad():
            for data in tqdm.tqdm(dataloader):
                data = maybe_dictionarize(data)
                x, y = data["images"].to(device), data["labels"].to(device)
                logits = utils.get_logits(x, model)
                pred = logits.argmax(dim=1, keepdim=True).to(device)
                correct += pred.eq(y.view_as(pred)).sum().item()
                total_loss += criterion(logits, y).item()
                n += y.size(0)

        top1 = correct / n
        average_loss = total_loss / len(dataloader)
        print(f"Done evaluating on {dataset_name}. Accuracy: {top1*100:.2f}%")
        return {"accuracy": top1, "loss": average_loss}

    def evaluate(self):
        """
        Evaluates the image encoder on multiple datasets specified in the args along with scaling factors.
        """

        results = {}
        for dataset_name in self.args.zeroshot_eval_datasets:
            dataset_results = self.eval_single_dataset(dataset_name)
            results[dataset_name] = dataset_results
        return results

    def save_results(self):
        """
        Save the results list as a JSON file.
        """
        if self.args.results_path:
            if not os.path.exists(os.path.dirname(self.args.results_path)):
                os.makedirs(os.path.dirname(self.args.results_path), exist_ok=True)
            with open(self.args.results_path, "w") as f:
                json.dump(self.results, f, indent=4)
            print(f"Results saved to {self.args.results_path}.")
        else:
            print("Results not saved (to do so, specify results_path in args).")

    def save_results_incrementally(self, index, result):
        """
        Updates the results file with the new result in the desired format.
        """
        names = "_".join(self.args.zeroshot_merge_models)
        results_file_path = os.path.join(
            self.args.results_path,
            self.args.exp_id,
        )
        if not os.path.exists(os.path.dirname(results_file_path)):
            os.makedirs(os.path.dirname(results_file_path), exist_ok=True)
        # Read existing results
        file_name = os.path.join(
            results_file_path,
            f"{names}scaling_results.json",
        )
        if os.path.exists(file_name):
            with open(file_name, "r") as f:
                try:
                    results = json.load(f)
                except json.JSONDecodeError:
                    results = {}
        else:
            results = {}

        # Update the results dictionary
        results[index] = result

        # Write the updated results back to the file
        with open(file_name, "w") as f:
            json.dump(results, f, indent=4)


def save_results_to_json(results, filename):
    with open(filename, "w") as f:
        json.dump(results, f, indent=4)
    print(f"Results have been saved to {filename}")


def evaluate_scalings(scalings, args):
    """
    Evaluate the image encoder model on multiple datasets with different scaling factors.
    :param scalings: List of scaling factors for each dimension.
    :param args: Configuration arguments including data locations, device, etc.
    """
    scaling_coefficients_results = {}
    finetune_checkpoints = [
        os.path.join(args.save, f"{data}/finetuned.pt")
        for data in args.zeroshot_merge_models
    ]
    if args.results_path:
        if not os.path.exists(os.path.join(args.results_path, args.exp_id)):
            os.makedirs(os.path.join(args.results_path, args.exp_id), exist_ok=True)
    for index, scaling in enumerate(scalings):
        image_encoder = get_merged_model(
            args.pretrained_checkpoint, finetune_checkpoints, scaling
        )
        evaluator = ZeroShotEvaluator(image_encoder, args)
        eval_results = evaluator.evaluate()
        models_info = {
            ds: scale for ds, scale in zip(args.zeroshot_merge_models, scaling)
        }
        scaling_coefficients_results[str(index)] = {
            "evals": eval_results,
            "models": models_info,
        }
        if args.results_path:
            evaluator.save_results_incrementally(
                str(index), {"evals": eval_results, "models": models_info}
            )

    else:
        print("Results not saved (to do so, specify results_path in args).")
    return scaling_coefficients_results


def evaluate_scalings_nested(scalings, args, merge_dict):
    preferences = read_config(args.preference)
    scaling_coefficients_results = {}

    if args.results_path:
        if not os.path.exists(
            os.path.join(
                args.results_path, args.exp_id, "_".join(args.zeroshot_merge_models)
            )
        ):
            os.makedirs(
                os.path.join(
                    args.results_path,
                    args.exp_id,
                    "_".join(args.zeroshot_merge_models),
                ),
                exist_ok=True,
            )
    for index, scaling in enumerate(scalings):
        image_encoder = get_merged_model(
            args.pretrained_checkpoint, args.finetuned_checkpoints, scaling
        )
        evaluator = ZeroShotEvaluator(image_encoder, args)
        eval_results = evaluator.evaluate()
        models_info = {
            ds: scale for ds, scale in zip(args.zeroshot_merge_models, scaling)
        }
        weighted_eval_results = {}
        used_preferences = []
        for dataset in args.zeroshot_merge_models:
            weighted_eval_results[dataset] = {}
            if dataset not in merge_dict:
                print("Not in merge_dict. Error.")
                break
            else:
                eval_datasets = merge_dict[dataset]
                preferences_lst = [preferences[ds] for ds in eval_datasets]
                used_preferences.extend((preferences_lst))
                normalized_preferences = [
                    p / sum(preferences_lst) for p in preferences_lst
                ]

                for metric_name in eval_results[eval_datasets[0]].keys():
                    metric = 0
                    for i, ds in enumerate(eval_datasets):
                        metric += (
                            normalized_preferences[i] * eval_results[ds][metric_name]
                        )
                    weighted_eval_results[dataset].update({metric_name: metric})

        if len(scalings) == 1:
            index = "final"

        if args.results_path:
            evaluator.save_results_incrementally(
                index,
                {
                    "evals": eval_results,
                    "raw_evals": eval_results,
                    "models": models_info,
                },
            )
        scaling_coefficients_results[index] = {
            "evals": weighted_eval_results,
            "raw_evals": eval_results,
            "models": models_info,
        }

    else:
        print("Results not saved (to do so, specify results_path in args).")
    if len(scalings) == 1:
        print("Calculating combined metric")
        nomarlized_combined_preference = [
            p / sum(used_preferences) for p in used_preferences
        ]
        combined_metric = 0
        for i, ds in enumerate(args.zeroshot_eval_datasets):
            combined_metric += (
                nomarlized_combined_preference[i]
                * scaling_coefficients_results["final"]["raw_evals"][ds][
                    args.metric_type.lower()
                ]
            )
        return combined_metric
    else:
        return scaling_coefficients_results
