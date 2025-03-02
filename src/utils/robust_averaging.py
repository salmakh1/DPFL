import logging

import torch
import numpy as np
import copy

from src.utils.train_utils import average_weights_defense


def knorm_average_weights(w, k):
    weights = list(w.values())
    # Calculate the average for each key
    w_avg = copy.deepcopy(weights[0])
    for key in w_avg.keys():
        values = [weight[key] for weight in weights]
        values = [value.cpu().numpy() for value in values]  # Convert to NumPy arrays
        sorted_values = sorted(values, key=lambda x: np.mean(x))
        k_values = sorted_values[:k]  # Exclude the k smallest and k largest values
        k_values = [torch.from_numpy(value) for value in k_values]  # Convert back to PyTorch tensors
        w_avg[key] = torch.mean(torch.stack(k_values), dim=0)
    return w_avg



def distance(a, b):
    # Convert the state_dict values to tensors
    weight1_tensors = [v for v in a.values()]
    weight2_tensors = [v for v in b.values()]

    # Compute the L2 distance for each corresponding tensor
    distances = [torch.norm(w1 - w2, p=2) for w1, w2 in zip(weight1_tensors, weight2_tensors)]

    # Calculate the overall L2 distance as the mean of individual tensor distances
    overall_distance = sum(distances) / len(distances)

    return overall_distance



def krum_average_weights(w, k):
    """
    Returns the parameter with the lowest score using Krum robust averaging.
    The score is defined as the sum of distances to its closest k vectors.

    param w: A dictionary of participant weights.
    param k: The number of nearest neighbors to consider.
    return: The parameter with the lowest score.
    """

    weights = list(w.values())

    num_weights = len(weights)

    # if num_weights <= k:
    #     logging.info(f"we are in the case of num_weights < k")
    #     return copy.deepcopy(weights[0])

    scores = np.zeros(num_weights)

    # Calculate the score for each weight vector
    for i in range(num_weights):
        distances = []
        for j in range(num_weights):
            if i != j:
                distances.append(distance(weights[i], weights[j]))
        distances.sort()
        logging.info(f"k {k}")
        k_closest_distances = distances[:k]
        logging.info(f"k_closest_distances {k_closest_distances}")
        scores[i] = sum(k_closest_distances)

    # Find the weight with the lowest score
    min_score_idx = np.argmin(scores)
    min_score_weight = weights[min_score_idx]
    logging.info(f"min_score_idx {min_score_idx}")

    return min_score_weight


def clipped_average_weights(w, clip_value):
    weights = list(w.values())
    # Clip each weight update to the specified range
    for i in range(len(weights)):
        for key in weights[i].keys():
            weights[i][key].clamp_(-clip_value, clip_value)

    # Calculate the average
    w_avg = copy.deepcopy(weights[0])
    for key in w_avg.keys():
        w_avg[key] = torch.mean(torch.stack([weight[key] for weight in weights]), dim=0)

    return w_avg


# def multi_krum_average_weights(w, num_models, num_k):
#
#     weights = list(w.values())
#     # Calculate distances between models
#     distances = torch.zeros(num_models, num_models)
#     for i in range(num_models):
#         for j in range(num_models):
#             distances[i][j] = torch.norm(torch.cat([weights[i][key] - weights[j][key] for key in weights[i].keys()]))
#
#     # Find the num_k models with the smallest sum of distances
#     min_distances_sum = torch.sum(torch.topk(distances, num_k, largest=False).values, dim=1)
#     selected_models = torch.argsort(min_distances_sum)[:num_k]
#
#     # Calculate the average using the selected models
#     w_avg = copy.deepcopy(weights[selected_models[0]])
#     for key in w_avg.keys():
#         w_avg[key] = torch.mean(torch.stack([weights[i][key] for i in selected_models]), dim=0)
#
#     return w_avg


def multi_krum_average_weights(w, k, m):
    """
    Returns the parameter with the lowest score using Multi-Krum robust averaging.
    The score is defined as the sum of distances to its closest k vectors.

    :param w: A dictionary of participant weights.
    :param k: The number of nearest neighbors to consider for Multi-Krum.
    :param m: The number of selected participants with the lowest scores.
    :return: The parameter with the lowest score.
    """

    weights = list(w.values())

    num_weights = len(weights)

    # if num_weights <= k:
    #     return copy.deepcopy(weights[0])

    scores = np.zeros(num_weights)

    # Calculate the score for each weight vector
    for i in range(num_weights):
        distances = []
        for j in range(num_weights):
            if i != j:
                distances.append(distance(weights[i], weights[j]))
        distances.sort()
        k_closest_distances = distances[:k]
        scores[i] = sum(k_closest_distances)

    logging.info(f"the scores are {scores}")
    # Find the indices of the lowest score participants
    min_score_indices = np.argsort(scores)[:m]
    logging.info(f"min_score_idx {min_score_indices}")
    logging.info(f" k is {k} and m is {m}")
    w = [weights[i] for i in range(m)]

    result = average_weights_defense(w)
    # Initialize the result as the first selected participant
    # result = weights[min_score_indices[0]]
    #
    # # Aggregate the selected participants' weights
    # for i in range(1, m):
    #     result += weights[min_score_indices[i]]
    #
    # result /= m

    return result

# def multi_krum_average_weights(w, k, m):
#     """
#     Returns the parameter with the lowest score using Multi-Krum robust averaging.
#     The score is defined as the sum of distances to its closest k vectors.
#
#     :param w: A dictionary of participant weights.
#     :param k: The number of nearest neighbors to consider for Multi-Krum.
#     :param m: The number of selected participants from the lowest score.
#     :return: The parameter with the lowest score.
#     """
#
#     weights = list(w.values())
#
#     num_weights = len(weights)
#     num_features = len(weights[0])
#
#     if num_weights <= k:
#         return copy.deepcopy(weights[0])
#
#     scores = np.zeros(num_weights)
#
#     # Calculate the score for each weight vector
#     for i in range(num_weights):
#         distances = []
#         for j in range(num_weights):
#             if i != j:
#                 distances.append(distance(weights[i], weights[j]))
#         distances.sort()
#         k_closest_distances = distances[:k]
#         scores[i] = sum(k_closest_distances)
#
#     # Find the indices of the lowest score participants
#     min_score_indices = np.argsort(scores)[:m]
#     min_score_weights = [weights[i] for i in min_score_indices]
#
#     # Compute the average of the selected participants
#     result = torch.zeros(num_features)
#     for weight in min_score_weights:
#         result += weight
#     result /= m
#
#     return result
