import csv
import math
import os
import random
from itertools import chain

from omegaconf import OmegaConf
import logging
import torch

from fl_client.client_celoss import Client
from selection.bandits import optimized_rgl, optimized_rgl_p, constrained_greedy, straightforward_selection, \
    non_monotone_constrained_greedy
from src.data.sent140_utils import get_word_emb_arr
from src.utils.attacks import label_flip_attack, gradient_boost_attack, random_update_attack, FlippedLabelsDataset, \
    same_model_attack, sign_flip_model_attack

log = logging.getLogger(__name__)
from hydra.utils import instantiate
import copy
import time
from src.utils.train_utils import average_weights, test, random_selection, initialize_weights, set_seed, run_wandb, \
    saving_model, wandb_log, weighted_average_weights, testing, testing_, make_collaboration_symmetric, \
    remove_asymmetric_collaborations
import wandb
import datetime
import logging
import numpy as np
from torch.utils.data import Dataset, DataLoader

log = logging.getLogger(__name__)


class trainClientsFL(object):

    def __init__(self, num_clients, m, available_clients, cfg, hydra_cfg):

        # self.device = 'cuda' if 'gpu' else 'cpu'
        self.num_clients = num_clients
        self.m = m
        self.available_clients = available_clients
        self.cfg = cfg
        self.hydra_cfg = hydra_cfg
        self.device = torch.device('cuda') if cfg.device else torch.device('cpu')
        print(self.device)
        self.training_sets = self.test_dataset = None
        self.model = None
        self.epoch = 0
        self.client_id = 0

    def train_clients(self):

        #############INITIATE wandb#################
        if str(
                self.hydra_cfg["split"]) == "dirichlet" or str(
            self.hydra_cfg["split"]) == "split_three_clients":
            missing = str(self.cfg.split.alpha)
            alpha = self.cfg.split.alpha
        elif str(self.hydra_cfg["split"]) == "random_split":
            missing = str(0)
            alpha = 0
        else:
            missing = str(
                self.cfg.split.missing_classes)
            alpha = self.cfg.split.alpha
        if self.cfg.use_wandb:
            note = f"sampling_{str(self.cfg.sampling)}_cardinality_{str(self.cfg.cardinality)}" \
                   f"_alpha_{str(alpha)}_mindata_{str(self.cfg.split.min_dataset_size)}_p_{str(self.cfg.p)}" \
                   f"_periodicity_of_bandit_{str(self.cfg.periodicity_of_bandit)}_repetition_{str(self.cfg.repetition)}" \
                   f"_greedy_set_{str(self.cfg.greedy_set)}_weighted_avg_{str(self.cfg.weighted_avg)}_rgl_{str(self.cfg.rgl)}_constrained_greedy_{str(self.cfg.constrained_greedy)}" \
                   f"_malicious_{str(self.cfg.malicious_clients)}_attack_{str(self.cfg.selected_attack)}"
            time_stamp = run_wandb("_Constrainedrgl", self.cfg, self.hydra_cfg, note)
            logging.info(f" the time stamp is {time_stamp}")

        # logging.info(f" the time stamp is {time_stamp}")
        bools = [self.cfg.bandit, self.cfg.global_, self.cfg.real_bandit, self.cfg.FedAvg]
        assert bools.count(True) <= 1
        bools2 = [self.cfg.rgl, self.cfg.constrained_greedy]
        assert bools2.count(True) == 1 and bools2.count(False) == 1
        # Loading the data
        log.info("################  Instanciating the data  ##############")
        print(OmegaConf.to_yaml(self.cfg))
        datamodule = instantiate(self.cfg.datamodule, _recursive_=False)

        # ############## load and split the data for clients #################
        self.cfg.model.num_classes = datamodule.num_classes

        random.seed(self.cfg.datamodule.seed)
        torch.manual_seed(self.cfg.datamodule.seed)

        # set seed
        set_seed(seed=self.cfg.datamodule.seed)

        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, self.cfg.gpus))
        logging.info(f"{torch.cuda.device_count()}")
        logging.info(f"the GPU device is {torch.cuda.current_device()}")

        # train and test data:
        log.info("load_and_split")
        train_sets, val_sets, _, local_test_sets, global_test_set, val_pub_sets, distribution, transformed_public, data_weights = datamodule.load_and_split(
            self.cfg,
            pub=False,
            local_test=True,
            val_pub=False,
            transformed_pub=False)

        logging.info(f" data weights are {data_weights}")
        # pub_data = datamodule.public_loader(public_data)
        global_test_set = datamodule.global_test_loader(global_test_set)

        logging.info(f"WE SUCCESSFULLY LOADED TRAIN, VAL AND TEST")

        # Create clients
        log.info("Preparing the clients and the models and datasets..")
        clients = []
        local_pub_set = {}
        ids_to_clients = {}

        if self.cfg.malicious_clients:
            client_ids_list = [i for i in range(self.cfg.num_clients)]
            random_list_size = int(len(client_ids_list) * self.cfg.corrupted_ids)
            malicious_client_ids = random.sample(client_ids_list, random_list_size)
            logging.info(f"malicicous clients ids are: {malicious_client_ids}")
            benign_client_ids = [client_id for client_id in list(range(0, self.cfg.num_clients)) if
                                 client_id not in malicious_client_ids]
            logging.info(f"benign clients ids are: {benign_client_ids}")

        log.info("Constructing global model..")
        s = time.time()
        if self.cfg.task == "NLP_sent":
            VOCAB_DIR = os.path.join(os.path.expanduser('~'), 'decentralised_learning/Federated_learning/src'
                                                              '/model/glove/embs.json')
            # VOCAB_DIR = os.path.join(os.path.expanduser('~'), 'personalized_FL/src'
            #                                                   '/model/glove/embs.json')
            emb, indd, vocab = get_word_emb_arr(VOCAB_DIR)
            global_model = instantiate(self.cfg.model, emb=emb)
            global_model.apply(initialize_weights)
            global_model = global_model.to(device=self.device)
            e = time.time()
            logging.info(f"the time for global is {e - s}")
        else:
            indd = None
            global_model = instantiate(self.cfg.model)
            global_model.apply(initialize_weights)
            global_model = global_model.to(device=self.device)

        for i in range(self.cfg.num_clients):
            log.info(f"Construct client {i}..")
            train_loader = datamodule.train_loaders(train_sets)
            # this would be the validation dataset

            val_loader = datamodule.val_loaders(val_sets)

            if self.cfg.malicious_clients:
                if self.cfg.selected_attack.label_flip:
                    num_classes = self.cfg.datamodule.num_classes
                    logging.info(f"malicious {i}")
                    if i in malicious_client_ids:
                        malicious_train_dataset = FlippedLabelsDataset(train_loader.dataset,
                                                                       num_classes)
                        train_loader = DataLoader(malicious_train_dataset, batch_size=train_loader.batch_size,
                                                  shuffle=True)

                        malicious_val_dataset = FlippedLabelsDataset(val_loader.dataset,
                                                                     num_classes)
                        val_loader = DataLoader(malicious_val_dataset, batch_size=val_loader.batch_size, shuffle=True)

            # This is the public dataset coming from the validation
            if val_pub_sets:
                local_val_pub_loader = datamodule.val_loaders(val_pub_sets)
                local_pub_set[i] = local_val_pub_loader
            elif transformed_public:
                logging.info(f"use a transformed public dataset")
                local_val_pub_loader = datamodule.val_loaders(transformed_public)
                local_pub_set[i] = local_val_pub_loader
            else:
                # logging.info(f" pub is validation ")
                local_pub_set[i] = val_loader

            # This is the local test set
            local_test = datamodule.val_loaders(local_test_sets)

            # each client will have its own model
            self.cfg.client.client_id = self.client_id

            model = copy.deepcopy(global_model)

            client = instantiate(self.cfg.client,
                                 device=self.device,
                                 train_loaders=train_loader,
                                 model=model,
                                 val_loaders=val_loader,
                                 test_loaders=local_test,
                                 indd=indd)


            ids_to_clients[i] = client
            clients.append(client)
            self.client_id += 1
            datamodule.next_client()

        self.client_id = 0
        datamodule.set_client()

        ##TRAINING num_clients  for t rounds

        log.info("#################### Start training  #########################")
        bandit_reward = 0

        ##decay
        schedulers = []
        for client in clients:
            schedulers.append(torch.optim.lr_scheduler.StepLR(client.optimizer, step_size=10, gamma=0.998))

        clients_to_collaborate = {client.client_id: None for client in clients}
        clients_not_collaborate = {client.client_id: None for client in clients}
        weights_to_collaborate = {client.client_id: None for client in clients}

        active_clients = []

        best_val_acc = [-1 for client in clients]
        best_test_acc = [-1 for client in clients]
        best_agg_val_acc = [-1 for client in clients]
        best_agg_test_acc = [-1 for client in clients]
        # first_call_greedy = True
        first_call_greedy = {client.client_id: True for client in clients}

        first_set = {client.client_id: None for client in clients}
        for t in range(self.cfg.rounds):
            colaborated_with_malicious = {client.client_id: None for client in clients}

            log.info("####### This is ROUND number {}".format(t))
            datamodule.set_client()
            self.client_id = 0
            train_loss_batch = []
            train_acc_batch = []

            log.info("####### This is round starts with number of clients {}".format(self.cfg.m))
            if t == 0:
                available_clients = instantiate(self.cfg.selection, clients=clients,
                                                num_selected_clients=self.cfg.available_clients, t=t)
                active_clients = available_clients

            self.client_id = 0
            local_weights = {}
            local_weights_base = {}
            logging.info(f"active clients {active_clients}")

            train_loss, train_acc, local_val_test_acc, local_val_test_loss, local_test_acc, local_test_loss, local_val_agg_acc, local_val_agg_loss \
                , local_test_agg_acc, local_test_agg_loss, training_times, global_local_test_acc, global_local_test_loss = \
                [], [], [], [], [], [], [], [], [], [], [], [], []

            for client in active_clients:
                log.info(" training client {} in round {}".format(client, t))
                start_train_time = time.time()
                results = client.train(t=t)
                end_train_time = time.time()
                local_weights_base[client.client_id] = copy.deepcopy(client.model.base.state_dict())
                local_weights[client.client_id] = (copy.deepcopy(results["update_weight"]))  # detach

                if self.cfg.malicious_clients:
                    if client.client_id in malicious_client_ids:
                        logging.info(f"selected_attack: {self.cfg.selected_attack}")
                        if self.cfg.selected_attack.random_update:
                            random_update_attack(client)
                        elif self.cfg.selected_attack.same_model:
                            same_model_attack(client)
                        elif self.cfg.selected_attack.sign_flip:
                            sign_flip_model_attack(client)
                        local_weights[client.client_id] = copy.deepcopy(client.model.state_dict())
                        local_weights_base[client.client_id] = copy.deepcopy(client.model.base.state_dict())

                training_times.append(end_train_time - start_train_time)
                train_acc.append(results["train_acc"])
                train_loss.append(results["train_loss"])

                logging.info(f"validation..")
                val_test_results = client.validation()
                if self.cfg.malicious_clients:
                    if client.client_id in benign_client_ids:
                        local_val_test_acc.append(val_test_results["global_val_acc"])
                        local_val_test_loss.append(val_test_results["global_val_loss"])
                else:
                    local_val_test_acc.append(val_test_results["global_val_acc"])
                    local_val_test_loss.append(val_test_results["global_val_loss"])

                if val_test_results["global_val_acc"] >= best_val_acc[client.client_id]:
                    if self.cfg.malicious_clients:
                        if client.client_id in benign_client_ids:
                            best_val_acc[client.client_id] = val_test_results["global_val_acc"]
                            saving_model(client,
                                         f'./rgl_models_{time_stamp}/client_{client.client_id}_num_client_{self.cfg.num_clients}_seed_{self.cfg.datamodule.seed}_split_{str(self.hydra_cfg["datamodule"])}_missing_{missing}_LclS_{self.cfg.client.local_steps}_sampling_{self.cfg.sampling}')
                    else:
                        best_val_acc[client.client_id] = val_test_results["global_val_acc"]
                        saving_model(client,
                                     f'./rgl_models_{time_stamp}/client_{client.client_id}_num_client_{self.cfg.num_clients}_seed_{self.cfg.datamodule.seed}_split_{str(self.hydra_cfg["datamodule"])}_missing_{missing}_LclS_{self.cfg.client.local_steps}_sampling_{self.cfg.sampling}')

                logging.info(f"local test..")
                local_test_results = client.validation(test_=True)
                if self.cfg.malicious_clients:
                    if client.client_id in benign_client_ids:
                        local_test_acc.append(local_test_results["global_val_acc"])
                        local_test_loss.append(local_test_results["global_val_loss"])
                else:
                    local_test_acc.append(local_test_results["global_val_acc"])
                    local_test_loss.append(local_test_results["global_val_loss"])

                if local_test_results["global_val_acc"] > best_test_acc[client.client_id]:
                    if self.cfg.malicious_clients:
                        if client.client_id in benign_client_ids:
                            best_test_acc[client.client_id] = local_test_results["global_val_acc"]
                    else:
                        best_test_acc[client.client_id] = local_test_results["global_val_acc"]


            train_acc_batch.append(sum(train_acc) / len(train_acc))
            train_loss_batch.append(sum(train_loss) / len(train_loss))

            if not self.cfg.sampling:
                symmetry_matrix = [[0 for _ in range(self.cfg.num_clients)] for _ in range(self.cfg.num_clients)]
                frequency_of_selection = {client.client_id: 0 for client in active_clients}
                if self.cfg.repetition:
                    cond = t % self.cfg.periodicity_of_bandit == 0 and t != 0 and t >= 2
                else:
                    cond = t == self.cfg.periodicity_of_bandit and t != 0


                if self.cfg.bandit and cond:
                    for indx, client in enumerate(active_clients):
                        if self.cfg.random_sel:
                            # Randomly select 50% of the elements
                            active_clients_temp = active_clients[indx:] + active_clients[:indx]
                            first_elem = active_clients_temp[0]
                            num_to_select = len(active_clients_temp) // 2
                            random_selection = random.sample(active_clients_temp[1:], num_to_select)
                            active_clients_temp = [first_elem] + random_selection

                            logging.info(f"the length of the active_clients_temp is {len(active_clients_temp)}")
                        else:
                            active_clients_temp = active_clients[indx:] + active_clients[:indx]
                        if self.cfg.greedy_set:
                            if first_call_greedy[client.client_id]:
                                active_clients_temp_ = [client.client_id for client in active_clients_temp]
                                logging.info(f"the length of the active_clients_temp_ is {len(active_clients_temp_)}")

                            if self.cfg.repetition and self.cfg.greedy_set:
                                if clients_to_collaborate[client.client_id] is not None and first_call_greedy[
                                    client.client_id]:
                                    first_call_greedy[client.client_id] = False
                                    greedy_set = [c for c in active_clients_temp_ if
                                                  c in clients_to_collaborate[client.client_id]]
                                    if self.cfg.random_sel:
                                        greedy_set = [c for c in active_clients_temp_ if
                                                      c in clients_to_collaborate[client.client_id] if c is not client.client_id]
                                        selected_clients = random.sample(greedy_set, k=int(len(greedy_set) * 0.7))
                                        greedy_set = [client.client_id] + selected_clients
                                        logging.info(f"random greedy_set {greedy_set}")
                                    clients_not_collaborate[client.client_id] = [c for c in active_clients_temp_ if
                                                                                 c not in clients_to_collaborate[
                                                                                     client.client_id]]

                                    first_set[client.client_id] = greedy_set
                                    active_clients_temp_ = greedy_set
                                    logging.info(f"greedy for client {active_clients_temp_[0]} and {client.client_id}")
                                if clients_to_collaborate[client.client_id] is not None and not first_call_greedy[
                                    client.client_id]:
                                        active_clients_temp_ = first_set[client.client_id]

                        else:
                            active_clients_temp_ = [client.client_id for client in active_clients_temp]



                        x, frequency_of_selection = optimized_rgl_p(model=client.model,
                                                                    local_weights=local_weights,
                                                                    active_clients=active_clients_temp_,
                                                                    device=self.device,
                                                                    data_set=local_pub_set[client.client_id],
                                                                    sampling=self.cfg.sampling,
                                                                    p_=self.cfg.p,  # you can change it
                                                                    agg_base=False,
                                                                    selected=frequency_of_selection,
                                                                    data_weights=data_weights,
                                                                    cardinality=self.cfg.cardinality,
                                                                    t=t,
                                                                    task=self.cfg.task,
                                                                    indd=indd)
                        if self.cfg.malicious_clients:
                            if client.client_id in benign_client_ids:
                                colaborated_with_malicious[client.client_id] = [client_id for client_id in x if
                                                                                client_id in malicious_client_ids]
                            else:
                                colaborated_with_malicious[client.client_id] = []

                        logging.info(f"decided set {x}")
                        logging.info(f"decided set length {len(x)}")
                        clients_to_collaborate[client.client_id] = x
                        l = []
                        for c in clients_to_collaborate[client.client_id]:
                            l.append(data_weights[c])
                            symmetry_matrix[client.client_id][c] = 1
                        weights_to_collaborate[client.client_id] = l

                    logging.info(f"selected clients {frequency_of_selection}")


                    if self.cfg.malicious_clients:
                        logging.info(f"collaborated_with_malicious {colaborated_with_malicious}")
                    for client in active_clients:
                        if self.cfg.rgl_base:
                            new_weights = average_weights(local_weights_base, clients_to_collaborate[client.client_id])
                            client.model.base.load_state_dict(new_weights)
                        else:

                            if clients_to_collaborate[client.client_id] is None or len(
                                    clients_to_collaborate[client.client_id]) == 1:
                                continue
                            else:
                                logging.info(f"model_averaging..")
                                if self.cfg.weighted_avg:
                                    new_weights = average_weights(local_weights,
                                                                  clients_to_collaborate[client.client_id],
                                                                  weights_to_collaborate[client.client_id])
                                    client.model.load_state_dict(new_weights)
                                else:
                                    new_weights = average_weights(local_weights,
                                                                  clients_to_collaborate[client.client_id])
                                    client.model.load_state_dict(new_weights)
                        local_weights_base[client.client_id] = copy.deepcopy(client.model.base.state_dict())
                        local_weights[client.client_id] = copy.deepcopy(client.model.state_dict())
                end_agg_time = time.time()

            ########  Testing phase #######
            if self.cfg.malicious_clients:
                local_val_agg_acc, local_val_agg_loss, best_agg_val_acc, \
                local_test_agg_acc, local_test_agg_loss, best_agg_test_acc = \
                    testing_(active_clients, best_agg_val_acc, best_agg_test_acc, self.device,
                             benign_client_ids, malicious_clients=self.cfg.malicious_clients)
            else:
                local_val_agg_acc, local_val_agg_loss, best_agg_val_acc, \
                local_test_agg_acc, local_test_agg_loss, best_agg_test_acc = \
                    testing_(active_clients, best_agg_val_acc, best_agg_test_acc, self.device)

            ##########update scheduler ##############
            if t != 0:
                for scheduler in schedulers:
                    scheduler.step()

            if t == (self.cfg.rounds - 2):
                for client in available_clients:
                    saving_model(client,
                                 f'./rgl_global_last_models/client_{client.client_id}_num_client_{self.cfg.num_clients}_seed_{self.cfg.datamodule.seed}_split_{str(self.hydra_cfg["datamodule"])}_missing_{missing}_LclS_{self.cfg.client.local_steps}_sampling_{self.cfg.sampling}')

            ########### WANDB LOG ###############
            print("########## wandb log ##################")
            if self.cfg.use_wandb:
                wandb_log(best_val_acc, best_test_acc, best_agg_val_acc, best_agg_test_acc, train_loss_batch,
                          train_acc_batch,
                          local_val_test_acc,
                          local_val_test_loss, local_val_agg_acc, local_val_agg_loss, local_test_agg_acc,
                          local_test_agg_loss,
                          local_test_acc, local_test_loss, t)

                if self.cfg.bandit and t % self.cfg.periodicity_of_bandit == 0 and t != 0 and not self.cfg.sampling and cond:
                    wandb.log({
                        'client/num_selected_clients_0': len(clients_to_collaborate[active_clients[0].client_id]),
                        'client/num_selected_clients_1': len(clients_to_collaborate[active_clients[1].client_id]),
                    },
                        step=t)
                    if self.cfg.malicious_clients and cond:
                        malicious = list(chain(*list(colaborated_with_malicious.values())))
                        wandb.log({
                            'attack/benign_collaborated_with_mal': len(malicious),
                        },
                            step=t)
