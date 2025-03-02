import copy
import itertools
import os
import time
from typing import Optional

import torch
import logging

from src.data.sent140_utils import process_x, process_y, repackage_hidden
from src.utils.train_utils import test

log = logging.getLogger(__name__)


####a
class Client(object):

    def __init__(self, client_id, local_steps, task, learning_rate, batch_size, topk, device, train_loaders, model,
                 val_loaders: Optional = None, test_loaders: Optional = None, indd: Optional = None):

        self.client_id = client_id
        self.device = device
        self.local_steps = local_steps
        self.task = task
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.topk = topk
        self.model = model

        # train
        self.train_set = train_loaders


        # s = time.time()
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.learning_rate, momentum=0.9,
                                         weight_decay=0.0001)
        # e = time.time()
        # logging.info(f"e-s  {e - s}")
        self.criterion = torch.nn.CrossEntropyLoss()

        self.val_set = val_loaders

        self.test_set = test_loaders

        self.indd = indd

    def train(self, t=None, last_round=False):
        for param_group in self.optimizer.param_groups:
            logging.info(f"learning rate is {param_group['lr']} for client {self.client_id}")

        self.model.train()
        ac = []
        los = []
        if not last_round:
            local_steps = self.local_steps
            if self.task == "NLP_sent":
                for completed_steps in range(local_steps):
                    hidden_train = self.model.init_hidden(self.batch_size)
                    batch_acc = []
                    batch_loss = []
                    for batch_idx, (images, label) in enumerate(self.train_set):
                        data, labels = process_x(images, self.indd), process_y(label, self.indd)
                        data, labels = data.to(self.device), labels.to(self.device)
                        # labels = labels.float()
                        self.optimizer.zero_grad()
                        hidden_train = repackage_hidden(hidden_train)
                        outputs, hidden_train = self.model(data, hidden_train)
                        loss = self.criterion(outputs.t(), torch.max(labels, 1)[1])
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 20.0)
                        self.optimizer.step()

                        run_loss = loss.item()
                        _, predicted = torch.max(outputs.t(), 1)
                        _, lab = labels.max(1)
                        correct = predicted.eq(lab).sum().item()
                        total = labels.size(0)
                        acc = 100. * correct / total
                        batch_acc.append(acc)
                        batch_loss.append(run_loss)

                    ac.append(sum(batch_acc) / len(batch_acc))
                    los.append(sum(batch_loss) / len(batch_loss))
            else:
                for completed_steps in range(local_steps):
                    batch_acc = []
                    batch_loss = []
                    targets = []
                    for batch_idx, (images, label) in enumerate(self.train_set):
                        if self.task == "NLP":
                            label = label.float()
                        inputs, labels = images.to(self.device), label.to(self.device)

                        targets.append(labels.tolist())
                        # zero the parameter gradients
                        self.optimizer.zero_grad()
                        outputs = self.model(inputs)

                        predicted_probabilities, predicted_labels = torch.max(outputs, dim=1)

                        loss = self.criterion(outputs, labels)
                        loss.backward()
                        # gradient clipping
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 20)
                        self.optimizer.step()

                        run_loss = loss.item()
                        _, predicted = outputs.max(1)

                        if self.task == "NLP":
                            _, lab = labels.max(1)
                            correct = predicted.eq(lab).sum().item()
                        else:
                            correct = predicted.eq(labels).sum().item()
                        # correct = predicted.eq(labels).sum().item()

                        total = labels.size(0)
                        acc = 100. * correct / total
                        batch_acc.append(acc)
                        batch_loss.append(run_loss)

                    ac.append(sum(batch_acc) / len(batch_acc))
                    los.append(sum(batch_loss) / len(batch_loss))
                    flattened_targets = list(itertools.chain.from_iterable(targets))
                    unique_labels = set(flattened_targets)
                    # logging.info(f"the labels for client {self.client_id} are {unique_labels}")
        else:
            logging.info(f"last round in {last_round}")
            for completed_steps in range(self.local_steps * 2):
                batch_acc = []
                batch_loss = []
                b = True
                for batch_idx, (images, label) in enumerate(self.train_set):
                    inputs, labels = images.to(self.device), label.to(self.device)
                    # zero the parameter gradients
                    self.optimizer.zero_grad()

                    for param in self.model.base.parameters():
                        param.requires_grad = True
                    for param in self.model.classifier.parameters():
                        param.requires_grad = False

                    outputs = self.model(inputs)
                    loss = self.criterion(outputs, labels)
                    loss.backward()
                    self.optimizer.step()

                    run_loss = loss.item()
                    _, predicted = outputs.max(1)
                    total = labels.size(0)
                    correct = predicted.eq(labels).sum().item()
                    acc = 100. * correct / total
                    batch_acc.append(acc)
                    batch_loss.append(run_loss)

                ac.append(sum(batch_acc) / len(batch_acc))
                los.append(sum(batch_loss) / len(batch_loss))

        acc = sum(ac) / len(ac)
        run_loss = sum(los) / len(los)

        weights = self.model.state_dict()

        logging.info(f"client {self.client_id} finished with loss {run_loss} and acc {acc}")

        results = {'clientId': self.client_id, 'update_weight': weights, 'train_acc': acc,
                   'train_loss': run_loss}

        # logging.info(f"the targets in the train are {set(itertools.chain(*targets))}")

        return results

    def validation(self, test_=False):
        if test_:

            test_results = test(self.model, self.device, self.test_set, test_client=False, client_id=self.client_id,
                                topk=self.topk, task=self.task, indd=self.indd)
        else:

            test_results = test(self.model, self.device, self.val_set, test_client=False, client_id=self.client_id,
                                topk=self.topk, task=self.task, indd=self.indd)

        return test_results
