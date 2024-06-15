import json
import numpy as np
from scipy.optimize import curve_fit
from deap import base, creator, tools, algorithms
import matplotlib.pyplot as plt
import pandas as pd
import pickle
import time
import os
from datetime import datetime
from Sampling import Sampling
import random


def load_json_data(json_path):
    with open(json_path, "r") as file:
        data = json.load(file)
    return data


def save_json_data(json_path, data):
    with open(json_path, "w") as file:
        json.dump(data, file)


def save_pkl_data(pkl_path, data):
    with open(pkl_path, "wb") as file:
        pickle.dump(data, file, protocol=pickle.HIGHEST_PROTOCOL)


def load_pkl_data(pkl_path):
    with open(pkl_path, "rb") as file:
        data = pickle.load(file)
    return data


def save_pf(pareto_points, pareto_decision_variables, args):
    assert len(pareto_points) == len(
        pareto_decision_variables
    ), "Length mismatch between decision variables and Pareto points"
    out_dict = {
        str(list(pareto_decision_variables[i])): list(pareto_points[i])
        for i in range(len(pareto_points))
    }
    if not os.path.exists(os.path.join(args.results_path, args.exp_id)):
        os.makedirs(os.path.join(args.results_path, args.exp_id))
    models = "_".join(args.zeroshot_merge_models)
    save_json_data(
        os.path.join(
            args.results_path,
            args.exp_id,
            f"{models}pareto_front.json",
        ),
        out_dict,
    )
    return


def load_pf(loading_path):
    dict_ = load_json_data(loading_path)

    pareto_decision_variables = np.array(list(dict_.keys()))
    pareto_points = np.array(list(dict_.values()))
    return pareto_points, pareto_decision_variables


def prepare_scaling(json_data):
    prepared_data = []
    for index, details in json_data.items():
        model_vals = details["models"]
        row = []
        for key, val in model_vals.items():
            row.append(val)
        prepared_data.append(row)
    return np.array(prepared_data, dtype=float)


def prepare_metric(json_data, metric):
    prepared_data = []
    for index, details in json_data.items():
        evals = details["evals"]
        row = []
        for key, val in evals.items():
            row.append(evals[key][metric])
        prepared_data.append(row)
    return np.array(prepared_data, dtype=float)


def plot_pareto_front_curves(
    pareto_points,
    Y,
    xy_labels,
    pareto_front_df=None,
    higher=True,
    title=None,
    accu1=None,
    accu2=None,
    path_to_save_fig=None,
):

    plt.figure(figsize=(6, 4), dpi=300)

    plt.scatter(
        Y[:, 0], Y[:, 1], marker=".", label="Grid point data", color="#C4E4FF", s=80
    )
    if pareto_front_df is not None:
        plt.scatter(
            pareto_front_df.iloc[:, 0],
            pareto_front_df.iloc[:, 1],
            color="#FF8E8F",
            s=150,
            marker="*",
            label="Pareto Front (Grid search)",
            alpha=0.8,
        )

    if pareto_points.size > 0:
        plt.scatter(
            pareto_points[:, 0],
            pareto_points[:, 1],
            c="#B1AFFF",
            marker=">",
            s=10,
            alpha=0.4,
            label="Pareto Front (MAP, predicted)",
        )
        plt.title(title)
        plt.xlabel(f"{xy_labels[0]}")
        plt.ylabel(f"{xy_labels[1]}")
        plt.title(f"Pareto fronts from grid search and MAP")
        if higher:
            plt.legend(loc="lower left")
            print(
                f"Putting legend in lower left, if not visible, try upper right by setting higher = False."
            )
        else:
            plt.legend(loc="upper right")
    if accu1 is not None and accu2 is not None:
        plt.scatter(
            pareto_front_df.iloc[:, 0],
            pareto_front_df.iloc[:, 1],
            color="gray",
            marker="*",
            label="Empirical Pareto Front",
        )
        plt.axhline(
            y=accu2 / 100,
            color="#FDDE55",
            linestyle="-",
            label=f"Horizontal Line at y = {accu2}",
            linewidth=2,
        )
        plt.axvline(
            x=accu1 / 100,
            color="#FDDE55",
            linestyle="-",
            label=f"Vertical Line at x = {accu1}",
            linewidth=2,
        )
    if path_to_save_fig:
        plt.savefig(path_to_save_fig)


def get_scaling_from_pf(
    pareto_points, pareto_decision_variables, preference_lst, higher
):
    """
    Description:
        Get the scaling coefficients from the Pareto front
    """
    normalized_preference_lst = preference_lst / np.linalg.norm(preference_lst)
    assert len(pareto_points) == len(
        pareto_decision_variables
    ), "Length mismatch between decision variables and Pareto points"

    # weighted sum of the Pareto points
    weighted_sum = np.sum(pareto_points * normalized_preference_lst, axis=1)
    if higher:
        idx = np.argmax(weighted_sum)
    else:
        idx = np.argmin(weighted_sum)
    scaling_coefficients = pareto_decision_variables[idx]
    return scaling_coefficients


def get_hex_time(ms=False):
    """
    Description:
        get the current time in the format "DD/MM/YY HH:MM:SS" and convert it to a hexadecimal string
    """
    if ms:
        # Get current time with microseconds
        current_time = datetime.now().strftime("%d/%m/%y %H:%M:%S.%f")

        # Convert the time string to a datetime object
        dt_object = datetime.strptime(current_time, "%d/%m/%y %H:%M:%S.%f")

        # Convert the datetime object to a Unix timestamp with microseconds
        unix_time_with_microseconds = dt_object.timestamp()

        # Convert the Unix timestamp to a hexadecimal string, slicing off the '0x' and the 'L' at the end if it exists
        hex_time = hex(int(unix_time_with_microseconds * 1e6))[2:]

    else:
        current_time = time.strftime("%d/%m/%y %H:%M:%S", time.localtime())
        # convert the timestamp string to a Unix timestamp
        unix_time = int(time.mktime(time.strptime(current_time, "%d/%m/%y %H:%M:%S")))

        # convert the Unix timestamp to a hexadecimal string
        hex_time = hex(unix_time)[2:]

    return hex_time


def hex_to_time(hex_time, ms=False):
    """
    input:
        hex_time: str
    description:
        convert a hexadecimal string to a timestamp string in the format "DD/MM/YY HH:MM:SS"
    """
    # convert the hexadecimal string to a Unix timestamp
    if ms:
        # Convert the hexadecimal string to a Unix timestamp including microseconds
        unix_time_with_microseconds = (
            int(hex_time, 16) / 1e6
        )  # Divide by 1e6 to convert microseconds to seconds

        # Convert the Unix timestamp to a datetime object
        dt_object = datetime.fromtimestamp(unix_time_with_microseconds)

        # Format the datetime object to a string including microseconds
        time_str = dt_object.strftime("%d/%m/%y %H:%M:%S.%f")

    else:
        unix_time = int(hex_time, 16)

        # convert the Unix timestamp to a timestamp string in the format "DD/MM/YY HH:MM:SS"
        time_str = time.strftime("%d/%m/%y %H:%M:%S", time.localtime(unix_time))

    return time_str


def get_scalings(dimension, total_points, prop_isotropic, prop_onehot):
    sampler = Sampling(dimension=dimension)
    scalings = sampler.generate_combined_points(
        total_points=total_points,
        prop_isotropic=prop_isotropic,
        prop_onehot=prop_onehot,
    )
    return scalings


def read_config(file_path):
    try:
        with open(file_path, "r") as file:
            config_data = json.load(file)
        return config_data
    except FileNotFoundError:
        raise FileNotFoundError(f"Config file {file_path} not found.")
    except json.JSONDecodeError:
        raise ValueError("Error decoding JSON from the config file.")


def softmax(x, temperature=1.0):
    """Compute softmax values for each set of scores in x with temperature."""
    e_x = np.exp((x - np.max(x)) / temperature)  # shift values for numerical stability
    return e_x / e_x.sum(axis=0)


def normalize_dict(d, temperature=1.0):
    """
    input d: dictionary
    output: normalized dictionary by applying softmax to the values of the dictionary
    """
    """Normalize the dictionary by applying softmax across values of the dictionary."""
    keys = list(d.keys())
    values = np.array(list(d.values()))
    softmax_values = softmax(values, temperature)

    normalized_dict = {keys[i]: softmax_values[i] for i in range(len(keys))}
    return normalized_dict


def get_middle_points(bins):
    return (bins[:-1] + bins[1:]) / 2


def sample_keys(prob_dict, num_samples):
    """
    given a dictionary, sample keys according to their probabilities (values) and return the counts of each key
    e.g.
    probabilities = {'a': 0.5, 'b': 0.3, 'c': 0.2}
    num_samples = 10
    output = {'a': 5, 'b': 3, 'c': 2}
    """
    if not (0.999 <= sum(prob_dict.values()) <= 1.001):
        raise ValueError("The sum of probabilities must be approximately 1.")
    if not isinstance(num_samples, int) or num_samples < 0:
        raise ValueError("Number of samples must be a non-negative integer.")

    # Extract keys and their respective probabilities
    keys = list(prob_dict.keys())
    probabilities = list(prob_dict.values())

    # Sample keys according to their probabilities
    sampled_keys = random.choices(keys, weights=probabilities, k=num_samples)

    # Count the occurrences of each key in the sample
    result_dict = {key: 0 for key in keys}  # Initialize dictionary for result
    for key in sampled_keys:
        result_dict[key] += 1

    return result_dict


def batch_cartesian_to_hyperspherical(x):
    r = np.linalg.norm(x, axis=1)  # Compute the magnitude for each point
    squares = np.square(x)
    cumulative_sums = np.cumsum(squares[:, ::-1], axis=1)[:, ::-1]
    phi = np.arctan2(np.sqrt(cumulative_sums[:, 1:]), x[:, :-1])
    return phi, r


def mean_plus_half_std(mask, losses, lamba=0.5, std_percentage=20, std_sample=30):
    if np.any(mask):
        mean = np.mean(losses[mask])
        # randomly flip 20% of the points from True to False in the mask
        variance_list = []
        for k in range(std_sample):
            mask = flip_true_values(mask, percentage=std_percentage)
            if np.any(mask):
                variance_list.append(np.mean(losses[mask]))
        std = np.std(variance_list)
        score = mean + lamba * std
    else:
        score = -np.inf
    return score


def flip_true_values(array, percentage=20):
    """
    Flip a given percentage of True values to False in a numpy boolean array.

    Parameters:
    array (np.ndarray): A numpy array of boolean values.
    percentage (int): The percentage of True values to flip to False.

    Returns:
    np.ndarray: The modified array with some True values flipped to False.
    """
    # Validate input percentage
    if not 0 <= percentage <= 100:
        raise ValueError("Percentage must be between 0 and 100")

    # Find the indices where the array is True
    true_indices = np.where(array)[0]

    # Calculate the number of True values to flip
    num_to_flip = int((percentage / 100.0) * len(true_indices))

    # Randomly select the indices of the True values to flip
    indices_to_flip = np.random.choice(true_indices, num_to_flip, replace=False)

    # Set the selected True values to False
    array[indices_to_flip] = False

    return array


def sample_theta_uniformly(middles, delta, k):
    """
    input middles: [middle_1, middle_2, ..., middle_d] -> [low_1, up_1] x [low_2, up_2] x... x [low_d, up_d]
    delta: low_i = middle_i - delta/2, up_i = middle_i + delta/2
    k: number of samples
    output: np.array([[sample_1_dim_1, sample_1_dim_2, ..., sample_1_dim_d], ...])
    """
    n = len(middles)  # Number of dimensions
    middles = np.array(
        middles
    )  # Convert middles to a numpy array for vectorized operations

    # Calculate the low and high bounds for each dimension
    lows = middles - delta / 2
    highs = middles + delta / 2

    # Initialize an array to store the samples
    samples = np.empty((k, n))

    # Generate k samples within the specified bounds
    for i in range(n):
        samples[:, i] = np.random.uniform(low=lows[i], high=highs[i], size=k)

    return samples


def batch_hyperspherical_to_cartesian(phi, r):
    n_points, dim_phi = phi.shape
    n = dim_phi + 1
    sin_vals = np.sin(phi)
    cos_vals = np.cos(phi)
    sin_product = np.cumprod(sin_vals, axis=1)

    x = np.zeros((n_points, n))
    x[:, 0] = r * cos_vals[:, 0]
    x[:, 1:-1] = r[:, np.newaxis] * sin_product[:, :-1] * cos_vals[:, 1:]
    x[:, -1] = r * sin_product[:, -1]

    return x


def get_sample_n_bins(dim):
    if dim < 4:
        sampled_n_bins = 4
    elif dim <= 6:
        sampled_n_bins = 2
    else:
        sampled_n_bins = 2
        print("WE DONOT SUPPORT DIM > 6 YET! YOU CAN RUN IT BUT WILL GET BAD RESULTS!")

    return sampled_n_bins


def find_closest_datasets(all_datasets, accuracy_dict):
    closest_pair = None
    min_diff = float("inf")

    # Iterate through each pair of datasets
    for i in range(len(all_datasets)):
        for j in range(i + 1, len(all_datasets)):
            dataset1 = all_datasets[i]
            dataset2 = all_datasets[j]

            # Calculate the absolute difference in accuracy
            diff = abs(accuracy_dict[dataset1] - accuracy_dict[dataset2])

            # Update the closest pair if this pair has a smaller difference
            if diff < min_diff:
                min_diff = diff
                closest_pair = (dataset1, dataset2)

    return closest_pair
