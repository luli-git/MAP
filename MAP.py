from utils import (
    load_json_data,
    prepare_scaling,
    prepare_metric,
    normalize_dict,
    sample_keys,
    sample_theta_uniformly,
    batch_hyperspherical_to_cartesian,
    get_middle_points,
    mean_plus_half_std,
    batch_cartesian_to_hyperspherical,
)
import numpy as np
from scipy.optimize import curve_fit
from deap import base, creator, tools, algorithms
from functools import partial
import os
import itertools
import json


def sigmoid(x):
    x = np.clip(x, -500, 500)
    return 1 / (1 + np.exp(-x))


def softplus(x):
    return np.log1p(
        np.exp(x)
    )  # np.log1p computes log(1 + x) more accurately for small x


def create_objective_function(params, activation):
    def objective(individual):
        return quadratic_model(individual, params, activation=activation)

    return objective


def quadratic_model(X, params, activation=lambda x: x):
    """
    X: array-like, shape (nb_task, nb_point)
    params: list of parameters for the quadratic model (quadratic + linear + constant)
    activation: activation function to apply to the output of the model, default is identity
    """
    n = len(X)  # Number of variables
    expected_num_params = (n + 1) * (n + 2) // 2  # Quadratic + Linear + Constant
    assert len(params) == expected_num_params, "Incorrect number of parameters"
    # Start building the quadratic model
    result = 0
    index = 0
    # Quadratic terms: x_i^2
    for i in range(n):
        result += params[index] * X[i] ** 2
        index += 1
    # Cross-product terms: x_i * x_j for all i < j
    for i in range(n):
        for j in range(i + 1, n):
            result += params[index] * X[i] * X[j]
            index += 1
    # Linear terms: x_i
    for i in range(n):
        result += params[index] * X[i]
        index += 1
    # Constant term
    result += params[index]
    return activation(result)


def fit_quadratic_from_json(X, y, dimension, activation=lambda x: x):
    try:
        initial_guess = np.ones(((dimension + 1) * (dimension + 2) // 2,))

        def model(X, *params):
            return quadratic_model(X, params, activation)

        params, _ = curve_fit(
            model, X.T, y.T, p0=initial_guess
        )  # Transpose to match curve_fit input
        return params
    except Exception as e:
        print(f"An error occurred: {e}")
        return None


def create_reference_points(nobj, p_values, scale_values):
    """
    Create combined reference points for NSGA-III optimization.

    Parameters:
    - nobj (int): Number of objectives.
    - p_values (list of int): List of 'p' values (partitions) for each reference point set.
    - scale_values (list of float): Scaling factors for each reference point set.

    Returns:
    - numpy.ndarray: Array of unique reference points.
    """
    # Generate reference points for each specified 'p' and scale
    ref_points = [
        tools.uniform_reference_points(nobj, p, s)
        for p, s in zip(p_values, scale_values)
    ]

    # Combine all reference points into one array
    ref_points = np.concatenate(ref_points, axis=0)

    # Remove duplicates from the combined array
    _, unique_indices = np.unique(ref_points, axis=0, return_index=True)
    ref_points = ref_points[unique_indices]

    return ref_points


def setup_nsga3(*objectives, higher=True, specific_dim=None):
    # Setup DEAP
    dimension = len(objectives)
    if specific_dim:
        dimension = specific_dim
    if higher:
        creator.create(
            "FitnessMulti", base.Fitness, weights=(1.0,) * dimension
        )  # weights can be adjusted if needed
    else:
        creator.create(
            "FitnessMulti", base.Fitness, weights=(-1.0,) * dimension
        )  # weights can be adjusted if needed
    creator.create("Individual", list, fitness=creator.FitnessMulti)

    toolbox = base.Toolbox()
    NDIM = dimension  # Number of decision variables
    BOUND_LOW, BOUND_UP = -3.0, 3.0

    toolbox.register("attr_float", np.random.uniform, BOUND_LOW, BOUND_UP)
    toolbox.register(
        "individual",
        tools.initRepeat,
        creator.Individual,
        toolbox.attr_float,
        n=dimension,
    )
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    toolbox.register("evaluate", lambda ind: tuple(obj(ind) for obj in objectives))
    toolbox.register(
        "mate", tools.cxSimulatedBinaryBounded, low=BOUND_LOW, up=BOUND_UP, eta=20.0
    )
    toolbox.register(
        "mutate",
        tools.mutPolynomialBounded,
        low=BOUND_LOW,
        up=BOUND_UP,
        eta=20.0,
        indpb=1.0 / NDIM,
    )

    P = [2, 1]  # List of 'p' values for each reference point set
    SCALES = [1, 0.5]  # Scaling factors for each set

    ref_points = create_reference_points(dimension, P, SCALES)
    toolbox.register("select", tools.selNSGA3, ref_points=ref_points)
    return toolbox


def find_pareto_front(
    params, higher=True, activation=sigmoid, path=None, ngen=150, specific_dim=None
):
    """
    Find the Pareto front using NSGA-II with the specified parameters.
    higher means if the metric we are interestd is higher the better or lower the better.
    params is a list of parameters for the objective functions.
    """
    objectives = [create_objective_function(p, activation) for p in params]
    setup_with_higher = partial(setup_nsga3, higher=higher, specific_dim=specific_dim)
    toolbox = setup_with_higher(*objectives)

    # Standard NSGA-II setup
    MU = 100
    NGEN = ngen
    pop = toolbox.population(n=MU)
    hof = tools.ParetoFront()
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean, axis=0)
    stats.register("std", np.std, axis=0)
    stats.register("min", np.min, axis=0)
    stats.register("max", np.max, axis=0)

    algorithms.eaMuPlusLambda(
        pop,
        toolbox,
        mu=MU,
        lambda_=MU,
        cxpb=0.7,
        mutpb=0.3,
        ngen=NGEN,
        stats=stats,
        halloffame=hof,
        verbose=True,
    )
    # Process results
    pareto_points = np.array([ind.fitness.values for ind in hof])
    pareto_decision_variables = np.array([ind for ind in hof])
    # dictionary of pareto points and decision variables
    res = {}
    res["pareto_points"] = pareto_points
    res["pareto_decision_variables"] = pareto_decision_variables
    return res


def fit_quadratic_functions(config, activation, scaling_coefficients_results=None):
    names = "_".join(config.zeroshot_merge_models)
    if scaling_coefficients_results is None:
        scaling_coefficients_results = load_json_data(
            os.path.join(
                config.results_path, config.exp_id, f"{names}scaling_results.json"
            )
        )
    else:  # json_data is provided
        assert isinstance(
            scaling_coefficients_results, dict
        ), "scaling_coefficients_results must be a dictionary containing the results."

    X, y = prepare_scaling(scaling_coefficients_results), prepare_metric(
        scaling_coefficients_results, "accuracy"
    )

    dimension = len(scaling_coefficients_results["0"]["models"])
    params, error_encountered = [], False

    for i in range(dimension):
        try:
            params2 = fit_quadratic_from_json(
                X, y[:, i], dimension, activation=activation
            )
            if params2 is not None:
                params.append(params2)
            else:
                print(f"Fit failed for {config.eval_datasets[i]}")
        except Exception as e:
            print(
                f"An unexpected error occurred in processing {config.eval_datasets[i]}: {e}"
            )
    # save the params
    # to list of list
    params = [list(p) for p in params]
    if not error_encountered:
        with open(
            os.path.join(config.results_path, config.exp_id, f"{names}_params.json"),
            "w",
        ) as f:
            json.dump(params, f, indent=4)

    return params, error_encountered


def read_params(path_):
    with open(path_, "r") as f:
        params = json.load(f)
    return params


def choose_activation(metric_type, activation=None):
    if activation is not None:
        return activation
    else:
        assert isinstance(metric_type, str), "metric_type must be a string"
        if metric_type.lower() == "accuracy":
            return sigmoid
        elif metric_type.lower() == "loss":
            return softplus
        else:
            raise ValueError(
                "metric_type must be 'accuracy' or 'loss', or specify your own activation."
            )


def r_distribution(number, distribution_type, dist_info):
    """
    input: number: int, number of points to sample
    distribution_type: str, distribution type to sample the points from the hyperspherical space.
    dist_info: dict, distribution information to sample the points from the hyperspherical space.
    """
    if distribution_type.lower() == "gaussian":
        assert "mean" in dist_info.keys(), "mean should be in dist_info"
        assert "std" in dist_info.keys(), "std should be in dist_info"
        rs = np.random.normal(dist_info["mean"], dist_info["std"], number)
    elif distribution_type.lower() == "uniform" or r_distribution == "heuristic":
        assert "min" in dist_info.keys(), "min should be in dist_info"
        assert "max" in dist_info.keys(), "max should be in dist_info"
        rs = np.random.uniform(dist_info["min"], dist_info["max"], number)
    else:
        raise ValueError(f"Not implemented distribution type {distribution_type}")
    rs = np.clip(rs, 0, 1.5)
    return rs


def bayesian_pipeline(
    data,
    nb_task,
    params,
    sampled_n_bins,
    ground_truth_target,
    acquisition_function=None,
    quadratic_model_func=quadratic_model,
    activation=sigmoid,
    temperature=0.07,
    points_to_sample=None,
    prior_dist=None,
    lamba=0.5,
    std_percentage=20,
    std_sample=30,
    r_distribution=r_distribution,
    r_dist_type=None,
    r_dist_info=None,
):
    """
    input variable:
    data: np.array of shape (nb_points, nb_tasks), each point is (c_1, ... , c_T), T is the number of tasks.
    sampled_n_bins: int, number of bins to sample for each dimension of hyper-spherical space along with phi dimensions.
    entropy_calc_bin: int, number of bins to calculate entropy for each bin.
    quadratic_model_func: function, model to calculate the scores.
            It should take the **data** and **params** and **activation_function** as input and return the scores.
            data: array-like, shape (nb_task, nb_point)
            params: list of parameters for the quadratic model (quadratic + linear + constant)
            activation_function: function, activation function to apply to the result, default is sigmoid.
    activation: function, activation function to apply to the result, default is sigmoid.
    temperature: float, temperature to normalize the entropy values in softmax.
    points_to_sample: int, number of points to sample from the entropy values.
    prior_dist: dict, prior distribution to sample the points from the hyperspherical space.
    r_mean: float, mean of the radius to sample from the hyperspherical space. default is 1.
    r_std: float, standard deviation of the radius to sample from the hyperspherical space. default is 0.07.
    """
    assert r_dist_type
    assert points_to_sample is not None, "points_to_sample is not implemented yet"
    assert r_dist_info is not None, "r_dist_info is not implemented yet"
    bin_score, bin_size = bayesian_compute_score_per_bin(
        data,
        nb_task,
        params,
        sampled_n_bins,
        ground_truth_target,
        acquisition_function=acquisition_function,
        quadratic_model_func=quadratic_model_func,
        activation=activation,
        lamba=lamba,
        std_percentage=std_percentage,
        std_sample=std_sample,
    )
    score_dist = normalize_dict(bin_score, temperature=temperature)
    if prior_dist is None:
        # # prior_dist by default is uniform distribution, using the same keys with score_dist
        # prior_dist = normalize_dict({key: 1 for key in score_dist.keys()})
        posterior_dist = score_dist
    else:
        assert set(prior_dist.keys()) == set(
            score_dist.keys()
        ), "The keys of prior_dist and score_dist should be the same"
        assert sum(prior_dist.values()) == 1, "The sum of prior_dist values should be 1"
        posterior_dist = score_dist
    realization_dict = sample_keys(posterior_dist, points_to_sample)
    list_cartesian_points = []
    for key, value in realization_dict.items():
        # print(f"Bin: {key}, Realizations: {value}")
        theta_s = sample_theta_uniformly(key, bin_size, value)
        rs = r_distribution(value, r_dist_type, r_dist_info)
        # clip the radius to be between 0 and 2

        list_cartesian_points.append(batch_hyperspherical_to_cartesian(theta_s, rs))
    cartesian_points = np.vstack(list_cartesian_points)
    return cartesian_points, posterior_dist


def bayesian_compute_score_per_bin(
    # Function to compute entropy for each bin
    data,
    nb_task,
    params,
    sampled_n_bins,
    ground_truth_target,
    acquisition_function=mean_plus_half_std,
    quadratic_model_func=None,
    activation=None,
    lamba=2.5,
    std_percentage=20,
    std_sample=30,
):
    """
    data: np.array of shape (nb_points, nb_tasks), each point is (c_1, ... , c_T), T is the number of tasks.
    sampled_n_bins: int, number of bins to sample for each dimension.
    ground_truth_target: np.array of shape (nb_points, 1), the ground truth target values.
    """
    assert (
        quadratic_model_func is not None
    ), "quadratic_model_func is not implemented yet"
    if activation is None:
        activation = lambda x: x
    assert len(data.shape) == 2, "Data should be 2D"
    assert (
        data.shape[1] == nb_task
    ), "Data should have the same number of tasks as the number of tasks"
    # Step 1: Transform the data to hyperspherical coordinates
    phi, r = batch_cartesian_to_hyperspherical(data)
    prediction = quadratic_model_func(data.T, params, activation=activation)
    losses = (prediction - ground_truth_target.reshape(-1)) ** 2
    # Step 2: Define the joint bins for the angular dimensions
    theta_dim = phi.shape[-1]  # Number of angular dimensions (n-1)
    bins = {i: np.linspace(0, np.pi / 2, sampled_n_bins + 1) for i in range(theta_dim)}
    bin_size = np.pi / 2 / sampled_n_bins
    bin_middle_points = {i: get_middle_points(k) for i, k in bins.items()}

    # Create joint bins using Cartesian product
    all_bins = list(
        itertools.product(*(bin_middle_points[i] for i in sorted(bin_middle_points)))
    )

    # Step 3: Calculate mean+0.5var for each bin
    bin_score = {}
    for bin_combination in all_bins:
        # Define the bin edges for this combination
        bin_edges = [
            (bin_combination[i] - bin_size / 2, bin_combination[i] + bin_size / 2)
            for i in range(theta_dim)
        ]

        # Find the points that fall within this bin
        mask = np.ones(len(phi), dtype=bool)
        for i in range(theta_dim):
            mask &= (phi[:, i] >= bin_edges[i][0]) & (phi[:, i] < bin_edges[i][1])

        # Calculate the mean and variance of the loss values for these points
        score = acquisition_function(
            mask,
            losses,
            lamba=lamba,
            std_percentage=std_percentage,
            std_sample=std_sample,
        )

        bin_score[bin_combination] = score

    return bin_score, bin_size


def get_scaling_metric_pairs(scaling_coefficients_eval, metric_type):
    metrics = []
    scalings = []
    for id, eval_and_model in scaling_coefficients_eval.items():
        metric = []
        scaling = []
        for dataset in eval_and_model["evals"].keys():
            metric.append(eval_and_model["evals"][dataset][metric_type])
            scaling.append(eval_and_model["models"][dataset])
        metrics.append(metric)
        scalings.append(scaling)
    return np.array(metrics), np.array(scalings)
