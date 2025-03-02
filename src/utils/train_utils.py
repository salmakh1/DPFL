import itertools
import logging
import os
import random
import statistics
import time
from collections import OrderedDict
from hydra.utils import instantiate
import wandb
from omegaconf import OmegaConf

import datetime
import numpy as np
import torch
import copy
from random import sample
from torch import nn
from torch.autograd import Variable
from sklearn.metrics import f1_score, accuracy_score, jaccard_score
import torch.nn.functional as F

from src.data.sent140_utils import process_x, repackage_hidden, process_y, get_word_emb_arr


def average_weights(w, neighbors=None, data_weights=None):
    """
    Returns the average of the weights.
    """

    if neighbors is None:
        neighbors = []
    if not neighbors:
        weights = list(w.values())
    else:
        weights = [w[c] for c in neighbors]

    if data_weights is not None:
        # logging.info(f"dataweight {data_weights}")
        # data_weights = [item for sublist in list(data_weights.values()) for item in
        #                 (sublist if isinstance(sublist, list) else [sublist])]
        data_weights = [x / sum(data_weights) for x in data_weights]
        # logging.info(f"data weights {data_weights} and sum is {sum(data_weights)}")

    if data_weights is None:
        w_avg = copy.deepcopy(weights[0])
        for key in w_avg.keys():
            for i in range(1, len(weights)):
                w_avg[key] += weights[i][key]
            w_avg[key] = torch.div(w_avg[key], len(weights))
    else:
        w_avg = copy.deepcopy(weights[0])
        for key in w_avg.keys():
            for i in range(0, len(weights)):
                if i == 0:
                    w_avg[key] = weights[i][key] * data_weights[i]
                else:
                    w_avg[key] += weights[i][key] * data_weights[i]
            # w_avg[key] = torch.div(w_avg[key], len(weights))
    return w_avg


def average_weights_defense(weights):
    """
    Returns the average of the weights.
    """

    w_avg = copy.deepcopy(weights[0])
    for key in w_avg.keys():
        for i in range(1, len(weights)):
            w_avg[key] += weights[i][key]
        w_avg[key] = torch.div(w_avg[key], len(weights))

    return w_avg


# def weighted_average_weights(w, neighbors=[], alpha=0.9):
#     """
#     Returns the average of the weights.
#     """
#
#     if not neighbors:
#         weights = list(w.values())
#     else:
#         weights = [w[c] for c in neighbors]
#
#     if len(weights) == 1:
#         alpha = 1
#
#     w_avg = copy.deepcopy(weights[0])
#     for key in w_avg.keys():
#         w_avg[key] = alpha * weights[0][key]
#
#     for key in w_avg.keys():
#         for i in range(1, len(weights)):
#             w_avg[key] += (1 - alpha) * torch.div(weights[i][key], len(weights) - 1)
#         # w_avg[key] = torch.div(w_avg[key], len(weights) - 1)
#
#     return w_avg


def weighted_average_weights(w, neighbors=[], alphas=None):
    """
    Returns the average of the weights.
    """

    logging.info(f"alphas {alphas}")
    if not neighbors:
        weights = list(w.values())
        if alphas:
            alphas = list(alphas.values())
    else:
        weights = [w[c] for c in neighbors]
        if alphas:
            alphas = [alphas[c] for c in neighbors]

    logging.info(f"the len of alphas is {len(alphas)} and the len of benign is {len(neighbors)}")
    # if len(weights) == 1:
    #     alpha = 1

    # alphas = torch.tensor(alphas, dtype=torch.float32)
    # alphas = alphas * 0.5
    # softmax_alphas = F.softmax(alphas, dim=0)
    # logging.info(f"softmax alphas are {softmax_alphas}")

    # tau = (torch.std(alphas) / torch.mean(alphas)) * 0.5
    # logging.info(f" tau is equal to {tau}")

    alphas = torch.tensor(alphas)
    # if all(alphas) == 0:
    #     alphas = [1 for _ in alphas]
    s = sum(alphas)
    alphas_ = [i / s for i in alphas]
    logging.info(f"alphas are {alphas_}")

    w_avg = copy.deepcopy(weights[0])
    for key in w_avg.keys():
        # w_avg[key] += softmax_alphas[0] * weights[0][key]
        w_avg[key] = torch.tensor(alphas_[0]) * weights[0][key]

    for key in w_avg.keys():
        for i in range(1, len(weights)):
            # w_avg[key] += softmax_alphas[i] * weights[i][key]
            w_avg[key] += torch.tensor(alphas_[0]) * weights[i][key]

        # w_avg[key] = torch.div(w_avg[key], len(weights) - 1)

    return w_avg


def improved_average_weights(prev_avg_weights, new_weight, n, add):
    if new_weight is None:
        return prev_avg_weights

    if add:
        for key in prev_avg_weights.keys():
            prev_avg_weights[key] = n * prev_avg_weights[key] + new_weight[key]
            prev_avg_weights[key] = torch.div(prev_avg_weights[key], n + 1)
    else:
        for key in prev_avg_weights.keys():
            prev_avg_weights[key] = n * prev_avg_weights[key] + new_weight[key]
            prev_avg_weights[key] = torch.div(prev_avg_weights[key], n - 1)

    return prev_avg_weights


def average_weights_frank(w, links):
    """
    Returns the average of the weights.
    """
    # weight = {}
    # for c in neighbors:
    #     weight[c] = copy.deepcopy(w[c])

    # logging.info(f"the size of the neighbors are {len(neighbors)} and weights are {len(weight)}")
    weights = list(w.values())

    # logging.info(f"the links are {links}")
    # logging.info(f"w_avg is {copy.deepcopy(weights[0])}")

    w_avg = copy.deepcopy(weights[0])
    for key in w_avg.keys():
        for i in range(0, len(weights)):
            if i == 0:
                # logging.info(f"torch.tensor(links[i-1]) {torch.tensor(links[i-1])}")
                w_avg[key] = torch.tensor(links[i]) * weights[i][key]
            else:
                w_avg[key] += torch.tensor(links[i]) * weights[i][key]

    # logging.info(f"after averaging {w_avg}")
    return w_avg


# def weighted_average_weights(w):
#     """
#     Returns the average of the weights.
#     """
#     weights = list(w.values())
#     w_avg = copy.deepcopy(weights[0])
#     for key in w_avg.keys():
#         for i in range(1, len(weights)):
#             w_avg[key] += weights[i][key]
#         w_avg[key] = torch.div(w_avg[key], len(weights))
#
#     return w_avg


#
def initialize_weights(m):
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_uniform_(m.weight.data, nonlinearity='relu')
        if m.bias is not None:
            nn.init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight.data, 1)
        nn.init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.Linear):
        nn.init.kaiming_uniform_(m.weight.data)
        nn.init.constant_(m.bias.data, 0)


def set_seed(seed=42):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # When running on the CuDNN backend, two further options must be set
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # Set a fixed value for the hash seed
    os.environ["PYTHONHASHSEED"] = str(seed)
    print(f"Random seed set as {seed}")


def return_knn_pred(l2_distance, template_labels, n_classes, knn_k):
    pred = torch.zeros((l2_distance.shape[0], n_classes)).cuda()
    # logging.info(f" the knn value is {knn_k} and size of l2 is {l2_distance.size(1)}")
    if knn_k > l2_distance.size(1):
        knn_k = l2_distance.size(1)
    values, indices = torch.topk(l2_distance, k=knn_k, dim=1, largest=True, sorted=True)
    for sample_id in range(l2_distance.shape[0]):
        topk_labels = template_labels[indices[sample_id]]
        for k_id in range(knn_k):
            pred[sample_id, topk_labels[k_id]] += torch.exp(values[sample_id, k_id])
    return torch.softmax(pred, dim=1)


def obtain_feature_label_pair(model, train_dataloader, num_extract=5000, task=None, indd=None, emb=None):
    model.eval()
    model.cuda()
    features, labels = [], []
    if task == "NLP_sent":
        hidden_train = model.init_hidden(4)
        with torch.no_grad():
            for batch_idx, (x, target) in enumerate(train_dataloader):
                data, label = process_x(x, indd), process_y(target, indd)
                data, label = data.cuda(), label.cuda()
                input = torch.transpose(data, 0, 1)
                # logging.info(f"size of input is {input.size()}")
                batch_size = input.size(1)  # Get the batch size from the input tensor
                encoder = torch.tensor(emb)
                emb_ = torch.zeros((25, batch_size, 300))
                # logging.info(f"size of emb is {emb_.size()}")
                for i in range(25):
                    for j in range(batch_size):
                        emb_[i, j, :] = encoder[input[i, j], :]
                emb_ = emb_.cuda()
                emb_ = emb_.view(300, batch_size, 25)
                hidden_train = repackage_hidden(hidden_train)
                if batch_size != hidden_train[0].size(1):
                    hidden_train = (hidden_train[0][:, :batch_size, :], hidden_train[1][:, :batch_size, :])
                rep, hidden_train = model.base(emb_, hidden_train)
                last_batch_size = rep.size(1)
                if last_batch_size != batch_size:
                    # Handle the size difference, perhaps by discarding or padding the last batch
                    rep = rep[:, :batch_size, :]
                    label = label[:batch_size]
                features.append(rep)
                labels.append(label)
    else:
        with torch.no_grad():
            for batch_idx, (x, target) in enumerate(train_dataloader):
                x, target = x.cuda(), target.to(dtype=torch.int64).cuda()
                rep = model.base(x)
                features.append(rep)
                labels.append(target)

    logging.info(f"torch {[x.size() for x in features]}")
    features = torch.cat(features, dim=0)
    labels = torch.cat(labels, dim=0)
    min_num = num_extract if num_extract < len(labels) else len(labels)
    random_seq = torch.randperm(len(labels))[:min_num]
    return features[random_seq], labels[random_seq]


def KNN_test(model, device, data_set, test_client=False, client_id=False, feature_label_pair=None, n_classes=10,
             knn_k=10, interpolation=0., task=None,indd= None):
    model.eval()  # tells net to do evaluating
    criterion = torch.nn.CrossEntropyLoss()
    batch_acc = []
    batch_loss = []
    targets = []
    # logging.info(f" feature label pair is  {feature_label_pair}")
    template_features, template_labels = feature_label_pair
    num_features = template_features.shape[0]
    if task == "NLP_sent":
        model.eval()
        batch_size = 4
        hidden_test = model.init_hidden(batch_size)
        for batch_idx, (images, label) in enumerate(data_set):
            data, labels = process_x(images, indd), process_y(label, indd)
            data, labels = data.to(device), labels.to(device)
            # labels = labels.float()
            with torch.no_grad():
                hidden_test = repackage_hidden(hidden_test)
                outputs, hidden_test = model(data, hidden_test)
                test_loss = criterion(outputs.t(), torch.max(labels, 1)[1])
                val_running_loss = test_loss.detach().item()
                model_pred = torch.softmax(outputs, dim=1)
                feature, _ = model.base(data, hidden_test)
                if len(feature.size()) < 2:
                    feature = feature.unsqueeze(0)
                feature = feature.unsqueeze(1).repeat(1, num_features, 1)
                template = template_features.unsqueeze(0).repeat(feature.shape[0], 1, 1)
                l2_distance = -torch.pow((feature - template), 2)
                l2_distance = torch.sum(l2_distance, dim=-1)
                knn_pred = return_knn_pred(l2_distance, template_labels, n_classes, knn_k)

                final_pred = interpolation * knn_pred + (1.0 - interpolation) * model_pred

                _, predicted = final_pred.max(1)
                total = labels.size(0)
                _, lab = labels.max(1)
                correct = predicted.eq(lab).sum().item()
                acc = 100. * correct / total
                batch_acc.append(acc)
                batch_loss.append(val_running_loss)
    else:
        for batch_idx, (images, label) in enumerate(data_set):
            if task == "NLP":
                label = label.float()
            data, target = images.to(device), label.to(device)
            targets.append(target.tolist())
            with torch.no_grad():
                output = model(data)
                test_loss = criterion(output, target)
                model_pred = torch.softmax(output, dim=1)
                val_running_loss = test_loss.detach().item()
                feature = model.base(data)
                if len(feature.size()) < 2:
                    feature = feature.unsqueeze(0)
                feature = feature.unsqueeze(1).repeat(1, num_features, 1)
                template = template_features.unsqueeze(0).repeat(feature.shape[0], 1, 1)
                l2_distance = -torch.pow((feature - template), 2)
                l2_distance = torch.sum(l2_distance, dim=-1)
                knn_pred = return_knn_pred(l2_distance, template_labels, n_classes, knn_k)

                final_pred = interpolation * knn_pred + (1.0 - interpolation) * model_pred

                _, predicted = final_pred.max(1)
                total = target.size(0)
                if task == "NLP":
                    _, lab = target.max(1)
                    correct = predicted.eq(lab).sum().item()
                else:
                    correct = predicted.eq(target).sum().item()
                # correct = predicted.eq(target).sum().item()
                acc = 100. * correct / total

                batch_acc.append(acc)
                batch_loss.append(val_running_loss)

    acc = sum(batch_acc) / len(batch_acc)
    loss = sum(batch_loss) / len(batch_loss)

    if test_client:
        test_results = {'clientId': client_id, 'val_acc': acc, 'val_loss': loss}
    else:
        test_results = {'global_val_acc': acc, 'global_val_loss': loss}

    # logging.info(f"the labels are {set(itertools.chain(*targets))}")
    return test_results


def test(model, device, data_set, test_client=False, client_id=False, topk=False, task=None, indd=None):
    model.eval()  # tells net to do evaluating
    criterion = torch.nn.CrossEntropyLoss()
    batch_acc = []
    batch_loss = []
    targets = []
    # logging.info(f" the number of validation  datapoints are {len(data_set.dataset)}")
    if task == "NLP_sent":
        model.eval()
        batch_size = 4
        hidden_test = model.init_hidden(batch_size)
        for batch_idx, (images, label) in enumerate(data_set):
            data, labels = process_x(images, indd), process_y(label, indd)
            data, labels = data.to(device), labels.to(device)
            # labels = labels.float()
            with torch.no_grad():
                hidden_test = repackage_hidden(hidden_test)
                outputs, hidden_test = model(data, hidden_test)
                test_loss = criterion(outputs.t(), torch.max(labels, 1)[1])
                val_running_loss = test_loss.detach().item()
                _, predicted = torch.max(outputs.t(), 1)
                _, lab = labels.max(1)
                correct = predicted.eq(lab).sum().item()
                total = labels.size(0)
                acc = 100. * correct / total
                batch_acc.append(acc)
                batch_loss.append(val_running_loss)
    else:
        for batch_idx, (images, label) in enumerate(data_set):
            data, target = images.to(device), label.to(device)
            if task == "NLP":
                target = target.float()
            targets.append(target.tolist())
            with torch.no_grad():
                output = model(data)
                test_loss = criterion(output, target)

                val_running_loss = test_loss.detach().item()
                if topk:
                    accu = accuracy(output, target, topk=(1, 5))
                    acc = accu[1].item()
                else:
                    _, predicted = output.max(1)
                    if task == "NLP":
                        _, lab = target.max(1)
                        correct = predicted.eq(lab).sum().item()
                    else:
                        correct = predicted.eq(target).sum().item()
                    total = target.size(0)
                    # correct = predicted.eq(target).sum().item()
                    acc = 100. * correct / total
                batch_acc.append(acc)
                batch_loss.append(val_running_loss)

    acc = sum(batch_acc) / len(batch_acc)
    loss = sum(batch_loss) / len(batch_loss)
    # unique_labels = set(itertools.chain(*targets))  # Get unique labels here
    # logging.info(f"unique validation labels are {unique_labels}")
    if test_client:
        test_results = {'clientId': client_id, 'val_acc': acc, 'val_loss': loss}
    else:
        test_results = {'global_val_acc': acc, 'global_val_loss': loss}

    # logging.info(f"the labels are {set(itertools.chain(*targets))}")
    return test_results


def compute_std(test_acc):
    # test_values = list(test_acc.values())
    logging.info(f"the accuracies are {test_acc} and avg is  {sum(test_acc) / len(test_acc)}")
    return statistics.stdev(test_acc)


def random_selection(clients, num_selected_clients, t, seed):
    np.random.seed(seed + t)
    # active_clients_idx = list(np.random.choice(num_clients, selected_clients, replace=False))
    active_clients_idx = sample(clients, num_selected_clients)
    return active_clients_idx


def top_k(logits, y, k: int = 1):
    """
    logits : (bs, n_labels)
    y : (bs,)
    """
    labels_dim = 1
    assert 1 <= k <= logits.size(labels_dim)
    k_labels = torch.topk(input=logits, k=k, dim=labels_dim, largest=True, sorted=True)[1]

    logging.info(f"k_labels {k_labels}")
    # True (#0) if `expected label` in k_labels, False (0) if not
    a = ~torch.prod(input=torch.abs(y.unsqueeze(labels_dim) - k_labels), dim=labels_dim).to(torch.bool)

    logging.info(f"a {a}")

    # These two approaches are equivalent
    # if False:
    y_pred = torch.empty_like(y)
    for i in range(y.size(0)):
        if a[i]:
            y_pred[i] = y[i]
        else:
            y_pred[i] = k_labels[i][0]

    f1 = f1_score(y_pred, y, average='weighted') * 100
    # acc = sum(correct)/len(correct)*100
    acc = accuracy_score(y_pred, y) * 100

    iou = jaccard_score(y, y_pred, average="weighted") * 100

    return acc, f1, iou, y_pred


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.reshape(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k)

        return res


def test_silo_vehicle(model, device, data_set, test_client=False, client_id=False, topk=False):
    model.eval()  # tells net to do evaluating
    batch_acc = []
    batch_loss = []
    for batch_idx, (images, label) in enumerate(data_set):
        data, target = images.to(device), label.to(device)
        with torch.no_grad():
            output = model(data)
            test_loss = hinge_loss(model, images, target)
            val_running_loss = test_loss.detach().item()
            predicted = torch.sign(output[:, -1]).long()
            total = target.size(0)
            correct = predicted.eq(target.long()).sum().item()
            acc = 100. * correct / total
            batch_acc.append(acc)
            batch_loss.append(val_running_loss)

    acc = sum(batch_acc) / len(batch_acc)
    loss = sum(batch_loss) / len(batch_loss)

    if test_client:
        test_results = {'clientId': client_id, 'val_acc': acc, 'val_loss': loss}
    else:
        test_results = {'global_val_acc': acc, 'global_val_loss': loss}

    return test_results


def test_silo_school(model, device, data_set, test_client=False, client_id=False, topk=False):
    model.eval()  # tells net to do evaluating
    batch_loss = []
    for batch_idx, (images, label) in enumerate(data_set):
        data, target = images.to(device), label.to(device)
        with torch.no_grad():
            test_loss = l2_loss(model, (data, target))
            val_running_loss = test_loss.detach().item()
            batch_loss.append(val_running_loss)

    loss = sum(batch_loss) / len(batch_loss)

    if test_client:
        test_results = {'clientId': client_id, 'val_loss': loss}
    else:
        test_results = {'global_val_loss': loss}

    return test_results


def hinge_loss(model, input, target, reg=0.1):
    inputs, targets = input, target
    preds = model(inputs).squeeze()
    losses = nn.functional.relu(1.0 - targets * preds) + 0.5 * reg * torch.norm(model.linear.weight) ** 2
    return torch.mean(losses)


def l2_loss(model, batch):
    inputs, targets = batch
    # Scalar output's last dimension needs to be squeezed.
    preds = model(torch.tensor(inputs))
    per_example_loss = 0.5 * (preds - targets) ** 2
    return torch.mean(per_example_loss)


def subtract_weights(w1, w2, device):
    """
    Returns the average of the weights.
    """
    sum = torch.tensor(0., requires_grad=True).to(device)  # 1. Set requires_grad=True
    for t1, t2 in zip(w1, w2):
        # logging.info(f"######################## T1 {t1}")
        sum += torch.norm(torch.subtract(t1, t2)) ** 2
    return sum


def saving_model(client, path):
    client_dir = path
    logging.info(f"client dir is {client_dir}")
    os.makedirs(client_dir, exist_ok=True)
    model_path = os.path.join(client_dir, 'model.pth')
    torch.save(client.model.state_dict(), model_path)


def run_wandb(algorithm_name, cfg, hydra_cfg, note=None):
    lr = str(
        cfg.client.learning_rate)

    if str(
            hydra_cfg["split"]) == "dirichlet":
        missing = str(cfg.split.alpha)
    elif str(hydra_cfg["split"]) == "random_split":
        missing = str(0)
    else:
        missing = str(
            cfg.split.missing_classes)

    run_name = "Clt" + "_" + str(
        cfg.num_clients) + "BY" + str(cfg.m) + "_Act_" + str(
        cfg.available_clients) + "_Rds_" + str(
        cfg.rounds) + "_" + "Data" + "_" + str(
        hydra_cfg["datamodule"]) + "_LclS_" + str(
        cfg.client.local_steps) + algorithm_name + "_split_" + str(
        hydra_cfg["split"]) + '_' + str(missing) + "_Seed_" + str(
        cfg.datamodule.seed) + "_lr_" + lr \
        # + \+ "_Opt_" + str(hydra_cfg["optim"])

    time_stamp = datetime.datetime.fromtimestamp(time.time()).strftime('%H%M%S')
    logging.info(f"time_stamp is {time_stamp}")
    print(wandb.config)
    print(cfg.logger)

    wandb_run = instantiate(cfg.logger, id=run_name + "_TS_" + time_stamp,
                            settings=wandb.Settings(start_method='fork'))
    if note:
        wandb.run.notes = note
        wandb.group = str(hydra_cfg["datamodule"])
    cfg.client.client_id = 0
    wandb.config = OmegaConf.to_container(
        cfg, resolve=True, throw_on_missing=True
    )
    if algorithm_name == "_Constrainedrgl" or algorithm_name == "_C":
        return time_stamp


def make_collaboration_symmetric(clients_to_collaborate):
    # Create a copy to modify the dictionary without affecting the iteration
    updated_collaborations = clients_to_collaborate.copy()

    for client_id, collaborators in clients_to_collaborate.items():
        for collaborator in collaborators:
            if collaborator not in updated_collaborations:
                # If the collaborator is not a key, initialize it with an empty list
                updated_collaborations[collaborator] = []
            if client_id not in updated_collaborations[collaborator]:
                # Make the relationship symmetric
                updated_collaborations[collaborator].append(client_id)

    return updated_collaborations


def remove_asymmetric_collaborations(clients_to_collaborate):
    # Create a copy to modify the dictionary without affecting the original data
    updated_collaborations = clients_to_collaborate.copy()

    # Collect keys to be removed after iteration to avoid modifying during iteration
    to_remove = []

    for client_id, collaborators in clients_to_collaborate.items():
        for collaborator in collaborators:
            # Check if the current collaborator has the client_id in their list
            if collaborator not in clients_to_collaborate or client_id not in clients_to_collaborate[collaborator]:
                to_remove.append((client_id, collaborator))

    # Remove asymmetric collaborations
    for client_id, collaborator in to_remove:
        if collaborator in updated_collaborations[client_id]:
            updated_collaborations[client_id].remove(collaborator)

    # Optionally, remove any client IDs that now have empty collaborator lists
    for client_id in list(updated_collaborations):
        if not updated_collaborations[client_id]:
            del updated_collaborations[client_id]

    return updated_collaborations


def testing(active_clients, best_agg_val_acc, best_agg_test_acc, global_test_set, device, benign_clients=None,
            malicious_clients=False):
    global_test_acc = []
    global_test_loss = []
    local_val_agg_acc = []
    local_val_agg_loss = []
    local_test_agg_acc = []
    local_test_agg_loss = []
    for client in active_clients:
        logging.info(f"global test..")
        global_test_results = test(client.model, device, global_test_set)
        global_test_acc.append(global_test_results["global_val_acc"])
        global_test_loss.append(global_test_results['global_val_loss'])

        logging.info(f"validation after agg ..")
        val_test_results = client.validation()
        local_val_agg_acc.append(val_test_results["global_val_acc"])
        local_val_agg_loss.append(val_test_results["global_val_loss"])
        if val_test_results["global_val_acc"] > best_agg_val_acc[client.client_id]:
            if malicious_clients:
                if client.client_id in benign_clients:
                    best_agg_val_acc[client.client_id] = val_test_results["global_val_acc"]
            else:
                best_agg_val_acc[client.client_id] = val_test_results["global_val_acc"]

        logging.info(f"local agg test..")
        local_test_agg_results = client.validation(test_=True)
        local_test_agg_acc.append(local_test_agg_results["global_val_acc"])
        local_test_agg_loss.append(local_test_agg_results["global_val_loss"])
        if local_test_agg_results["global_val_acc"] > best_agg_test_acc[client.client_id]:
            if malicious_clients:
                if client.client_id in benign_clients:
                    best_agg_test_acc[client.client_id] = local_test_agg_results["global_val_acc"]
            else:
                best_agg_test_acc[client.client_id] = local_test_agg_results["global_val_acc"]

    return global_test_acc, global_test_loss, local_val_agg_acc, local_val_agg_loss, best_agg_val_acc, \
           local_test_agg_acc, local_test_agg_loss, best_agg_test_acc


def testing_(active_clients, best_agg_val_acc, best_agg_test_acc, device, benign_clients=None,
             malicious_clients=False):
    global_test_acc = []
    global_test_loss = []
    local_val_agg_acc = []
    local_val_agg_loss = []
    local_test_agg_acc = []
    local_test_agg_loss = []
    for client in active_clients:
        logging.info(f"validation after agg ..")
        val_test_results = client.validation()
        if malicious_clients:
            if client.client_id in benign_clients:
                local_val_agg_acc.append(val_test_results["global_val_acc"])
                local_val_agg_loss.append(val_test_results["global_val_loss"])
        else:
            local_val_agg_acc.append(val_test_results["global_val_acc"])
            local_val_agg_loss.append(val_test_results["global_val_loss"])
        if val_test_results["global_val_acc"] > best_agg_val_acc[client.client_id]:
            if malicious_clients:
                if client.client_id in benign_clients:
                    best_agg_val_acc[client.client_id] = val_test_results["global_val_acc"]
            else:
                best_agg_val_acc[client.client_id] = val_test_results["global_val_acc"]

        logging.info(f"local agg test..")
        local_test_agg_results = client.validation(test_=True)
        if malicious_clients:
            if client.client_id in benign_clients:
                local_test_agg_acc.append(local_test_agg_results["global_val_acc"])
                local_test_agg_loss.append(local_test_agg_results["global_val_loss"])
        else:
            local_test_agg_acc.append(local_test_agg_results["global_val_acc"])
            local_test_agg_loss.append(local_test_agg_results["global_val_loss"])
        if local_test_agg_results["global_val_acc"] > best_agg_test_acc[client.client_id]:
            if malicious_clients:
                if client.client_id in benign_clients:
                    best_agg_test_acc[client.client_id] = local_test_agg_results["global_val_acc"]
            else:
                best_agg_test_acc[client.client_id] = local_test_agg_results["global_val_acc"]

    return local_val_agg_acc, local_val_agg_loss, best_agg_val_acc, \
           local_test_agg_acc, local_test_agg_loss, best_agg_test_acc


# def wandb_log(best_val_acc, best_test_acc, best_agg_val_acc, best_agg_test_acc, train_loss_batch, train_acc_batch,
#               local_val_test_acc,
#               local_val_test_loss, local_val_agg_acc, local_val_agg_loss, local_test_agg_acc, local_test_agg_loss,
#               local_test_acc, local_test_loss, global_test_acc,
#               global_test_loss, global_local_test_ac=None, global_local_test_loss=None, t):
#     wandb.log({'cumulative_round/val_acc': sum(best_val_acc) / len(best_val_acc),
#                'cumulative_round/test_acc': sum(best_test_acc) / len(best_test_acc),
#                'cumulative_round/val_agg_acc': sum(best_agg_val_acc) / len(best_agg_val_acc),
#                'cumulative_round/test_agg_acc': sum(best_agg_test_acc) / len(best_agg_test_acc),
#                },
#               step=t)
#
#     wandb.log({'round/train_loss': sum(train_loss_batch) / len(train_loss_batch),
#                'round/train_accuracy': sum(train_acc_batch) / len(train_acc_batch),
#                'round/local_val_test_acc': sum(local_val_test_acc) / len(local_val_test_acc),
#                'round/local_val_test_loss': sum(local_val_test_loss) / len(local_val_test_loss),
#                'round/local_val_aggregation_acc': sum(local_val_agg_acc) / len(local_val_agg_acc),
#                'round/local_val_aggregation_loss': sum(local_val_agg_loss) / len(local_val_agg_loss),
#                'round/local_test_aggregation_acc': sum(local_test_agg_acc) / len(local_test_agg_acc),
#                'round/local_test_aggregation_loss': sum(local_test_agg_loss) / len(local_test_agg_loss),
#                'round/local_test_acc': sum(local_test_acc) / len(local_test_acc),
#                'round/local_test_loss': sum(local_test_loss) / len(local_test_loss),
#                'round/global_test_acc': sum(global_test_acc) / len(global_test_acc),
#                'round/global_test_loss': sum(global_test_loss) / len(global_test_loss),
#                # 'round/global_local_test_acc': sum(global_local_test_acc) / len(global_local_test_acc),
#                # 'round/global_local_test_loss': sum(global_local_test_loss) / len(global_local_test_loss),
#                },
#               step=t)


def wandb_log(best_val_acc, best_test_acc, best_agg_val_acc, best_agg_test_acc, train_loss_batch, train_acc_batch,
              local_val_test_acc,
              local_val_test_loss, local_val_agg_acc, local_val_agg_loss, local_test_agg_acc, local_test_agg_loss,
              local_test_acc, local_test_loss, t):
    wandb.log({'cumulative_round/val_acc': sum(best_val_acc) / len(best_val_acc),
               'cumulative_round/test_acc': sum(best_test_acc) / len(best_test_acc),
               'cumulative_round/val_agg_acc': sum(best_agg_val_acc) / len(best_agg_val_acc),
               'cumulative_round/test_agg_acc': sum(best_agg_test_acc) / len(best_agg_test_acc),
               },
              step=t)

    wandb.log({'round/train_loss': sum(train_loss_batch) / len(train_loss_batch),
               'round/train_accuracy': sum(train_acc_batch) / len(train_acc_batch),
               'round/local_val_test_acc': sum(local_val_test_acc) / len(local_val_test_acc),
               'round/local_val_test_loss': sum(local_val_test_loss) / len(local_val_test_loss),
               'round/local_val_aggregation_acc': sum(local_val_agg_acc) / len(local_val_agg_acc),
               'round/local_val_aggregation_loss': sum(local_val_agg_loss) / len(local_val_agg_loss),
               'round/local_test_aggregation_acc': sum(local_test_agg_acc) / len(local_test_agg_acc),
               'round/local_test_aggregation_loss': sum(local_test_agg_loss) / len(local_test_agg_loss),
               'round/local_test_acc': sum(local_test_acc) / len(local_test_acc),
               'round/local_test_loss': sum(local_test_loss) / len(local_test_loss),
               },
              step=t)
