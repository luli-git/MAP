import os
from evaluation import (
    evaluate_scalings_nested,
    evaluate_scalings,
    evaluate_scalings_single,
)
from args import parse_arguments
from merge import get_merged_model
from MAP import (
    find_pareto_front,
    fit_quadratic_functions,
    choose_activation,
    bayesian_pipeline,
    quadratic_model,
    sigmoid,
    get_scaling_metric_pairs,
)
import torch
import json
from utils import (
    prepare_metric,
    plot_pareto_front_curves,
    save_pf,
    get_scalings,
    get_scaling_from_pf,
    read_config,
    get_sample_n_bins,
    mean_plus_half_std,
    find_closest_datasets,
    load_json_data,
)
from task_vectors.src.task_vectors import TaskVector
from copy import deepcopy


class MAP:
    def __init__(self, args):
        self.args = args
        self.higherbetter = self.args.metric_type.lower() == "accuracy"

    def run_MAP(self, xy_labels=None, args=None):
        dimension = len(args.zeroshot_merge_models)
        self.scalings = get_scalings(
            dimension=dimension,
            total_points=int((dimension + 1)(dimension + 2)),
            prop_isotropic=0,
            prop_onehot=0,
        )
        scaling_coefficients_results = evaluate_scalings(self.scalings, args)
        activation = choose_activation(args.metric_type)
        params, error_encountered = fit_quadratic_functions(
            args,
            activation=activation,
            scaling_coefficients_results=scaling_coefficients_results,
        )
        if not error_encountered:
            res = find_pareto_front(
                params, higher=self.higherbetter, activation=activation, ngen=25
            )
            pareto_points = res["pareto_points"]
            pareto_decision_variables = res["pareto_decision_variables"]
            if args.results_path:
                save_pf(pareto_points, pareto_decision_variables, args)
        if (
            (len(pareto_points) > 2)
            and (len(args.zeroshot_merge_models) == 2)
            and args.results_path
        ):
            y = prepare_metric(scaling_coefficients_results, args.metric_type.lower())
            plot_pareto_front_curves(
                pareto_points=pareto_points,
                Y=y,
                xy_labels=xy_labels,
                path_to_save_fig=os.path.join(
                    args.results_path, args.exp_id, "pareto_front.png"
                ),
            )
        return pareto_points, pareto_decision_variables

    def run_MAP_Bayesian(
        self,
        sampling_schedule=None,
        r_dist_type="uniform",
        r_dist_info={"min": 0.2, "max": 0.8},
        temp=0.07,
        mean_std_tradeoff_in_aqc_func=0.5,
        std_percentage=20,
        std_sample=30,
        xy_labels=None,
        args=None,
    ):
        if sampling_schedule is None:
            if args is not None:
                sampling_schedule = [args.bayes_initial_pts] + [
                    args.bayes_update_pts
                ] * (args.bayes_iter - 1)
            else:
                raise ValueError(
                    "Need to provide sampling schedule either in args or as input to the function"
                )
        with torch.no_grad():
            dimension = len(args.zeroshot_merge_models)
            init_scalings = get_scalings(
                dimension=dimension,
                total_points=sampling_schedule[0],
                prop_isotropic=0,
                prop_onehot=0,
            )
            finetuned_checkpoints = [
                os.path.join(args.save, f"{data}/finetuned.pt")
                for data in args.zeroshot_merge_models
            ]
            task_vectors = [
                TaskVector(args.pretrained_checkpoint, ft).vector
                for ft in finetuned_checkpoints
            ]
            pretrained_model = torch.load(args.pretrained_checkpoint)
            activation = choose_activation(args.metric_type)
            sampled_n_bins = get_sample_n_bins(dimension)
            task_info = {}
            final_params = []
            for d in range(dimension):
                scaling_coefficients_results = evaluate_scalings_single(
                    init_scalings,
                    args.zeroshot_eval_datasets[d],
                    task_vectors,
                    pretrained_model,
                    args,
                    start=0,
                )
                ground_truth_target, scalings = get_scaling_metric_pairs(
                    scaling_coefficients_results, args.metric_type
                )
                task_info[f"prior_dist_{d}"] = None
                task_info[f"scalings_{d}"] = deepcopy(scalings)
                for i in range(len(sampling_schedule)):
                    params, error_encountered = fit_quadratic_functions(
                        args,
                        activation=activation,
                        scaling_coefficients_results=scaling_coefficients_results,
                        dimension=dimension,
                        single_dimension=True,
                    )
                    params = params[0]
                    if error_encountered:
                        raise ValueError(
                            f"Error encountered in fitting quadratic functions {error_encountered}"
                        )
                    updated_points, posterior_dist = bayesian_pipeline(
                        data=task_info[f"scalings_{d}"],
                        nb_task=1,
                        params=params,
                        sampled_n_bins=sampled_n_bins,
                        ground_truth_target=ground_truth_target,
                        acquisition_function=mean_plus_half_std,
                        quadratic_model_func=quadratic_model,
                        activation=sigmoid,
                        temperature=temp,
                        points_to_sample=(
                            sampling_schedule[i + 1]
                            if i + 1 < len(sampling_schedule)
                            else sampling_schedule[-1]
                        ),
                        prior_dist=task_info[f"prior_dist_{d}"],
                        lamba=mean_std_tradeoff_in_aqc_func,
                        std_percentage=std_percentage,
                        std_sample=std_sample,
                        r_dist_type=r_dist_type,
                        r_dist_info=r_dist_info,
                    )
                    scaling_coefficients_results_ = evaluate_scalings_single(
                        updated_points,
                        args.zeroshot_eval_datasets[d],
                        task_vectors,
                        pretrained_model,
                        args,
                        start=sampling_schedule[i],
                    )
                    scaling_coefficients_results.update(scaling_coefficients_results_)
                    task_info[f"prior_dist_{d}"] = posterior_dist
                    if i == len(sampling_schedule) - 1:
                        final_params.append(params)
            res = find_pareto_front(
                final_params, higher=self.higherbetter, activation=activation, ngen=25
            )
            pareto_points = res["pareto_points"]
            pareto_decision_variables = res["pareto_decision_variables"]
            if args.results_path:
                save_pf(pareto_points, pareto_decision_variables, args)
            if (
                (len(pareto_points) > 2)
                and (len(args.zeroshot_merge_models) == 2)
                and args.results_path
            ):
                y = prepare_metric(
                    scaling_coefficients_results, args.metric_type.lower()
                )
                plot_pareto_front_curves(
                    pareto_points=pareto_points,
                    Y=y,
                    xy_labels=xy_labels,
                    path_to_save_fig=os.path.join(
                        args.results_path, args.exp_id, "pareto_front.png"
                    ),
                )
            return pareto_points, pareto_decision_variables

    def run_MAP_nested(self, accuracy_dict, args=None):
        # This is a dictionary that will store the accuracy of the models
        dimension = 2  # Because nested merging is done on 2 models at a time
        self.scalings = get_scalings(
            dimension=dimension, total_points=20, prop_isotropic=0.2, prop_onehot=0
        )
        all_datasets = args.zeroshot_merge_models
        merge_dict = {}
        for ds in all_datasets:
            merge_dict[ds] = [ds]
        while len(all_datasets) > 1:
            datasets = find_closest_datasets(all_datasets, accuracy_dict)
            for dataset in datasets:
                all_datasets.remove(dataset)

            merged_model_name = "_".join(datasets)
            merge_dict[f"{merged_model_name}_{args.exp_id}"] = []
            for ds in datasets:
                if ds in merge_dict:
                    merge_dict[f"{merged_model_name}_{args.exp_id}"].extend(
                        merge_dict[ds]
                    )

            args.zeroshot_merge_models = datasets
            args.zeroshot_eval_datasets = merge_dict[
                f"{merged_model_name}_{args.exp_id}"
            ]

            args.results_path = os.path.join(
                args.results_path,
            )
            args.finetuned_checkpoints = [
                os.path.join(args.save, f"{data}/finetuned.pt") for data in datasets
            ]
            activation = choose_activation(args.metric_type)

            scaling_coefficients_results = evaluate_scalings_nested(
                self.scalings,
                args,
                merge_dict,
            )
            params, error_encountered = fit_quadratic_functions(
                args, activation=activation
            )

            if not error_encountered:
                res = find_pareto_front(
                    params, higher=self.higherbetter, activation=activation, ngen=25
                )
                pareto_points = res["pareto_points"]
                pareto_decision_variables = res["pareto_decision_variables"]
                if args.results_path:
                    save_pf(pareto_points, pareto_decision_variables, args)
                y = prepare_metric(
                    scaling_coefficients_results, args.metric_type.lower()
                )
                plot_pareto_front_curves(
                    pareto_points=pareto_points,
                    Y=y,
                    xy_labels=datasets,
                    path_to_save_fig=os.path.join(
                        args.results_path,
                        args.exp_id,
                        f"{merged_model_name}pareto_front.png",
                    ),
                )
                preferences = read_config(args.preference)
                preference_lst = []
                for i in range(len(datasets)):
                    datasets_names = datasets[i]
                    eval_sets = merge_dict[datasets_names]
                    preference = [preferences[ds] for ds in eval_sets]
                    preference_lst.append(sum(preference))

                scaling_coefficients = get_scaling_from_pf(
                    pareto_points,
                    pareto_decision_variables,
                    preference_lst,
                    higher=self.higherbetter,
                )
                get_merged_model(
                    args.pretrained_checkpoint,
                    args.finetuned_checkpoints,
                    scaling_coefficients,
                    save_merged_model_dir=os.path.join(
                        args.save, f"{merged_model_name}_{args.exp_id}"
                    ),
                )
                # merge the results as well
                combined_metric = evaluate_scalings_nested(
                    [scaling_coefficients],
                    args,
                    merge_dict,
                )
                all_datasets.append(f"{merged_model_name}_{args.exp_id}")
                accuracy_dict[f"{merged_model_name}_{args.exp_id}"] = combined_metric
                # save accuracy_dict
                with open(
                    os.path.join(args.results_path, args.exp_id, "accuracy_dict.json"),
                    "w",
                ) as f:
                    json.dump(accuracy_dict, f)


if __name__ == "__main__":
    args = parse_arguments()
    accuracy_dict = {
        "CarsVal": 78.87,
        "DTDVal": 76.6,
        "EuroSATVal": 98.78,
        "GTSRBVal": 99.96,
        "MNISTVal": 99.60,
        "RESISC45Val": 96.56,
        "SUN397Val": 74.86,
        "SVHNVal": 96.5,
    }  # This is a dictionary that will store the accuracy of the unmerged models
    # only used for zero-shot evaluation in nested MAP
    map = MAP(
        args
    )  # the default task is nested 4 task: SUN397+DTD+Cars+SVHN, please change the self.args for your needs
    if args.method.lower() == "map":
        map.run_MAP(args=args)
    elif args.method.lower() == "nested":
        map.run_MAP_nested(accuracy_dict, args=args)
    elif args.method.lower() == "bayesian":
        map.run_MAP_Bayesian(args=args)
    else:
        raise ValueError(
            f"Invalid method {args.method}, need to be one of [map, nested, bayesian]."
        )
