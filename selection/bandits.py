import copy
import logging
import random

from src.utils.train_utils import test, average_weights, test_silo_vehicle, test_silo_school, improved_average_weights, \
    weighted_average_weights
import numpy as np
from orderedset import OrderedSet


def silo_oracle(model, local_weights, clients, device, data_set, silo=None):
    # No need to deepcopy local_weights here
    weights = {client: local_weights[client].copy() for client in clients}

    new_weights = average_weights(weights)
    model.load_state_dict(new_weights)

    if silo:
        test_results = test_silo_school(model, device, data_set, test_client=False, client_id=False)
    else:
        test_results = test(model, device, data_set, test_client=False, client_id=False)

    reward = (10 - test_results['global_val_loss']) / 10
    return reward


def oracle(model, local_weights, clients, device, data_set, agg_base=False, data_weights=None, task=None, indd=None):
    # No need to deepcopy local_weights here
    # logging.info(f"clients {clients}")
    weights = {client: copy.deepcopy(local_weights[client]) for client in clients}

    if data_weights:
        # data = {}
        l = [data_weights[i] for i in clients]
        # data[clients[0]] = l

        new_weights = average_weights(weights, data_weights=l)
    else:
        new_weights = average_weights(weights)

    if agg_base:
        model.base.load_state_dict(new_weights)
    else:
        model.load_state_dict(new_weights)

    test_results = test(model, device, data_set, test_client=False, client_id=False, task=task, indd=indd)

    reward = (10 - test_results['global_val_loss']) / 10
    # logging.info(f"loss for the set {weights.keys()} is {test_results['global_val_loss']} and reward is {reward}")

    return reward


def optimized_rgl(model, local_weights, active_clients, device, data_set, sampling=False, rep=1, agg_base=False):
    active_clients_indexes = active_clients
    n = len(active_clients_indexes)
    xi = [active_clients_indexes[0]]

    # Precompute initial_reward and f_y0
    r11 = oracle(model, local_weights, xi, device, data_set, agg_base=agg_base)
    x0 = copy.deepcopy(xi)
    r_0 = r11
    decided_set = x0
    decided_reward = r_0
    for _ in range(rep):
        xi = [active_clients_indexes[0]]
        yi = active_clients_indexes.copy()
        r11 = oracle(model, local_weights, xi, device, data_set, agg_base=agg_base)
        r12 = oracle(model, local_weights, yi, device, data_set, agg_base=agg_base)

        # yi_ = yi[1:]
        # random.shuffle(yi_)
        # yi = [yi[0]] + yi_
        for i in range(1, n):

            s = set(xi)
            s.add(active_clients_indexes[i])
            r21 = oracle(model, local_weights, list(s), device, data_set, agg_base=agg_base)
            ai = (r21 - r11)

            s = set(yi)
            s = s - {active_clients_indexes[i]}
            r22 = oracle(model, local_weights, list(s), device, data_set, agg_base=agg_base)
            bi = (r22 - r12)

            ai_prime = max(ai, 0)
            bi_prime = max(bi, 0)

            if ai_prime == bi_prime:
                if ai_prime == 0:
                    p = 0
            else:
                p = ai_prime / (ai_prime + bi_prime)
            # T = 0.1
            # p = np.exp(ai / T) / (np.exp(ai / T) + np.exp(bi / T))
            # logging.info(f"p is {p}")

            if np.random.uniform(0, 1) < p:
                xi.append(active_clients_indexes[i])
                add = True
                r11 = r21
                r12 = r12
            else:
                yi.remove(active_clients_indexes[i])
                add = False
                r11 = r11
                r12 = r22

            if add:
                r_star = r21
            else:
                r_star = r22

        decided_set = xi

    return decided_set


def optimized_rgl_p(model, local_weights, active_clients, device, data_set, sampling=False, p_=None, agg_base=True,
                    selected=None, data_weights=None, cardinality=None, t=None, task=None, indd=None):
    k = 0
    total = 0
    temp_model = copy.deepcopy(model)
    if p_ is None:
        p = [0]
    logging.info(f" the probability is {p_}")
    active_clients_indexes = active_clients
    n = len(active_clients_indexes)
    if cardinality is not None:
        break_point = int(cardinality)
    xi = [active_clients_indexes[0]]

    if t == 2:
        rep = 1
    else:
        rep = 1
    # if t == 5 and cardinality is not None:
    #     cardinality = cardinality - 1
    selected[active_clients_indexes[0]] += 1
    r11 = oracle(temp_model, local_weights, xi, device, data_set, agg_base=agg_base, data_weights=data_weights,
                 task=task, indd=indd)

    picked_sets = {}
    for r in range(rep):
        point = False
        xi = [active_clients_indexes[0]]
        yi = active_clients_indexes.copy()
        # yi_ = yi[1:]
        # random.shuffle(yi_)
        # yi = [yi[0]] + yi_
        z = active_clients_indexes.copy()
        z_ = z[1:]
        random.shuffle(z_)
        z = [z[0]] + z_

        r11 = oracle(temp_model, local_weights, xi, device, data_set, agg_base=agg_base, data_weights=data_weights,
                     task=task, indd=indd)

        r12 = oracle(temp_model, local_weights, yi, device, data_set, agg_base=agg_base, data_weights=data_weights,
                     task=task, indd=indd)

        for i in range(1, n):
            s = OrderedSet(xi)
            s.add(z[i])
            r21 = oracle(temp_model, local_weights, list(s), device, data_set, agg_base=agg_base,
                         data_weights=data_weights, task=task, indd=indd)
            # logging.info(f"r21 {r21}")
            ai = (r21 - r11)

            s = OrderedSet(yi)
            s = s - {z[i]}
            if len(list(s)) == 0:
                return xi, selected
            r22 = oracle(temp_model, local_weights, list(s), device, data_set, agg_base=agg_base,
                         data_weights=data_weights, task=task, indd=indd)
            bi = (r22 - r12)

            ai_prime = max(ai, 0)
            bi_prime = max(bi, 0)
            # logging.info(f"ai {ai} and bi {bi}")
            # logging.info(f"ai_p {ai_prime} and bi_p {bi_prime}")

            if ai_prime == bi_prime:
                if ai_prime == 0:
                    p = 1
            else:
                p = ai_prime / (ai_prime + bi_prime)
                if p != 0 and p != 1:
                    k += 1
                # logging.info(f"else, and p is {p}")

            total += 1
            if np.random.uniform(0, 1) < p:
                xi.append(z[i])
                add = True
                r11 = r21
                r12 = r12
            else:
                yi.remove(z[i])
                add = False
                r11 = r11
                r12 = r22

            if add:
                r_star = r21
            else:
                r_star = r22

            if cardinality is not None:
                if len(xi) == break_point:
                    point = True
                    logging.info(f"X breakpoint is {break_point}")
                    for k in xi:
                        if k != z[0]:
                            selected[k] += 1
                    reward = oracle(temp_model, local_weights, list(xi), device, data_set, agg_base=agg_base,
                                    data_weights=data_weights, task=task, indd=indd)
                    # logging.info(f"reward {reward} and r_star {r_star}")
                    picked_sets[reward] = xi
                    break

        if len(xi) < break_point or point == False:
            reward = oracle(temp_model, local_weights, list(xi), device, data_set, agg_base=agg_base,
                            data_weights=data_weights, task=task, indd=indd)
            picked_sets[reward] = xi

        logging.info(f"this is repetition {r} and picked_sets are {picked_sets}")
    # sorted_sets = dict(sorted(picked_sets.items()))
    best_reward = max(picked_sets, key=lambda l: l)
    xi = picked_sets[best_reward]
    # logging.info(f"client {xi[0]} K is {k} and total is {k / total} ")
    return xi, selected


def constrained_greedy(model, local_weights, active_clients, device, data_set, agg_base=True,
                       selected=None, cardinality=None):
    temp_model = copy.deepcopy(model)
    active_clients_indexes = active_clients
    n = len(active_clients_indexes)
    xi = [active_clients_indexes[0]]
    yi = active_clients_indexes.copy()
    yi = set(yi)
    yi.remove(active_clients_indexes[0])
    # yi = eliminate main clinet from yi
    k = int(cardinality)
    logging.info(f" START for client {xi}")
    for j in range(1, k):
        rewards = {}
        for c in yi:
            s = set(xi)
            r0 = oracle(temp_model, local_weights, list(s), device, data_set, agg_base=agg_base)
            s.add(c)
            r1 = oracle(temp_model, local_weights, list(s), device, data_set, agg_base=agg_base)
            rewards[c] = r1 - r0

        sorted_rewards = sorted(rewards.items(), key=lambda x: x[1], reverse=True)
        rewards_dict = dict(sorted_rewards)
        logging.info(f" rewards {rewards_dict}")
        best_k_elements = list(rewards_dict.keys())[:k + 1]
        logging.info(f" best elements are {best_k_elements[:(k - len(xi) + 1)]}")
        m = random.randint(0, (k - len(xi) + 1))
        # m = 0
        logging.info(f" m is {m}")
        xi.append(best_k_elements[m])
        yi.remove(best_k_elements[m])
        logging.info(f"xi is {xi} and yi is {yi}")
        selected[best_k_elements[m]] += 1

    return xi, selected


def non_monotone_constrained_greedy(model, local_weights, active_clients, device, data_set, agg_base=True,
                                    selected=None, cardinality=None, data_weights=None):
    temp_model = copy.deepcopy(model)
    active_clients_indexes = active_clients
    n = len(active_clients_indexes)
    xi = [active_clients_indexes[0]]
    yi = active_clients_indexes.copy()
    yi = set(yi)
    yi.remove(active_clients_indexes[0])
    # yi = eliminate main clinet from yi
    k = int(cardinality)
    logging.info(f" START for client {xi}")
    for j in range(1, k):
        rewards = {}
        for c in yi:
            s = OrderedSet(xi)
            r0 = oracle(temp_model, local_weights, list(s), device, data_set, agg_base=agg_base,
                        data_weights=data_weights)
            s.add(c)
            r1 = oracle(temp_model, local_weights, list(s), device, data_set, agg_base=agg_base,
                        data_weights=data_weights)
            rewards[c] = r1 - r0

        sorted_rewards = sorted(rewards.items(), key=lambda x: x[1], reverse=True)
        rewards_dict = dict(sorted_rewards)
        # logging.info(f" rewards {rewards_dict}")
        best_element_reward = list(rewards_dict.values())[0]
        best_element = list(rewards_dict.keys())[0]
        logging.info(f"best element is {best_element} and its reward is {best_element_reward}")
        if best_element_reward >= 0:
            xi.append(best_element)
            yi.remove(best_element)
            logging.info(f"xi is {xi} and yi is {yi}")
            selected[best_element] += 1
        else:
            print("else")
            return xi, selected

        # best_k_elements = list(rewards_dict.keys())[:k + 1]
        # logging.info(f" best elements are {best_k_elements[:(k - len(xi) + 1)]}")
        # m = random.randint(0, (k - len(xi) + 1))
        # m = 0
        # logging.info(f" m is {m}")

    return xi, selected


def straightforward_selection(model, local_weights, active_clients, device, data_set, agg_base=True,
                              selected=None):
    temp_model = copy.deepcopy(model)
    active_clients_indexes = active_clients
    n = len(active_clients_indexes)
    xi = [active_clients_indexes[0]]
    yi = active_clients_indexes.copy()
    yi = set(yi)
    # yi.remove(active_clients_indexes[0])
    # yi = eliminate main clinet from yi
    k = int(n / 2)
    logging.info(f" START for client {xi}")
    rewards = {}
    for c in yi:
        s = {c}
        r1 = oracle(temp_model, local_weights, list(s), device, data_set, agg_base=agg_base)
        rewards[c] = r1

    sorted_rewards = sorted(rewards.items(), key=lambda x: x[1], reverse=True)
    rewards_dict = dict(sorted_rewards)
    # logging.info(f" rewards {rewards_dict}")
    best_k_elements = list(rewards_dict.keys())[:k]
    logging.info(f" best elements are {best_k_elements[:k]}")

    for m in range(len(best_k_elements)):
        selected[best_k_elements[m]] += 1

    return best_k_elements, selected
