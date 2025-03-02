import csv
import os
import random

from matplotlib import pyplot as plt
from omegaconf import OmegaConf
import logging
import torch

from fl_client.client_celoss import Client
from src.data.sent140_utils import get_word_emb_arr
from src.utils.attacks import FlippedLabelsDataset, gradient_boost_attack, random_update_attack, same_model_attack, \
    sign_flip_model_attack
import wandb

from src.utils.robust_averaging import knorm_average_weights, multi_krum_average_weights, krum_average_weights

log = logging.getLogger(__name__)
from hydra.utils import instantiate
import copy
import time
from src.utils.train_utils import average_weights, test, random_selection, initialize_weights, set_seed, run_wandb, \
    testing, saving_model, wandb_log, testing_
import logging
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
                self.hydra_cfg["split"]) == "dirichlet":
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
                   f"_alpha_{str(alpha)}_mindata_{str(self.cfg.split.min_dataset_size)}" \
                   f"_malicious_{str(self.cfg.malicious_clients)}_attack_{str(self.cfg.selected_attack)}"
            run_wandb("_FedAvg", self.cfg, self.hydra_cfg, note)

        bools = [self.cfg.bandit, self.cfg.global_, self.cfg.real_bandit, self.cfg.FedAvg]
        assert bools.count(True) <= 1
        # Loading the data
        log.info("################  Instanciating the data  ##############")
        print(OmegaConf.to_yaml(self.cfg.datamodule))
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

        # pub_data = datamodule.public_loader(public_data)
        global_test_set = datamodule.global_test_loader(global_test_set)

        logging.info(f"WE SUCCESSFULLY LOADED TRAIN, VAL AND TEST")

        # Create clients
        log.info("Preparing the clients and the models and datasets..")
        clients = []
        local_pub_set = {}


        if self.cfg.malicious_clients:
            client_ids_list = [i for i in range(self.cfg.num_clients)]
            random_list_size = int(len(client_ids_list) * self.cfg.corrupted_ids)
            malicious_client_ids = random.sample(client_ids_list, random_list_size)
            logging.info(f"malicicous clients ids are: {malicious_client_ids}")
            benign_client_ids = [client_id for client_id in list(range(0, self.cfg.num_clients)) if
                                 client_id not in malicious_client_ids]


        log.info("Constructing global model..")
        if self.cfg.task == "NLP_sent":
            VOCAB_DIR = os.path.join(os.path.expanduser('~'), 'decentralised_learning/Federated_learning/src'
                                                              '/model/glove/embs.json')
            # VOCAB_DIR = os.path.join(os.path.expanduser('~'), 'personalized_FL/src'
            #                                                   '/model/glove/embs.json')
            emb, indd, vocab = get_word_emb_arr(VOCAB_DIR)
            global_model = instantiate(self.cfg.model, emb=emb)
            global_model.apply(initialize_weights)
            global_model = global_model.to(device=self.device)
        else:
            indd = None
            global_model = instantiate(self.cfg.model)
            global_model.apply(initialize_weights)
            global_model = global_model.to(device=self.device)

        for i in range(self.cfg.num_clients):
            log.info(f"Construct client {i}..")
            torch.cuda.empty_cache()
            train_loader = datamodule.train_loaders(train_sets)

            # this would be the validation dataset
            val_loader = datamodule.val_loaders(val_sets)
            ## Label flip attack:
            if self.cfg.malicious_clients:
                if self.cfg.selected_attack.label_flip:
                    num_classes = self.cfg.datamodule.num_classes
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

            # This is the local test set
            local_test = datamodule.val_loaders(local_test_sets)

            # each client will have its own model
            self.cfg.client.client_id = self.client_id

            client = instantiate(self.cfg.client,
                                 device=self.device,
                                 train_loaders=train_loader,
                                 model=copy.deepcopy(global_model),
                                 val_loaders=val_loader,
                                 test_loaders=local_test,
                                 indd=indd)

            # client = Client(client_id=i,
            #                 local_steps=2,
            #                 task="NLP_sent",
            #                 learning_rate=0.01,
            #                 batch_size=4,
            #                 topk=False,
            #                 device=self.device,
            #                 train_loaders=train_loader,
            #                 model=copy.deepcopy(global_model),
            #                 val_loaders=val_loader,
            #                 test_loaders=local_test,
            #                 indd=indd)
            clients.append(client)
            self.client_id += 1
            datamodule.next_client()

        self.client_id = 0
        datamodule.set_client()

        ##TRAINING num_clients  for t rounds

        log.info("#################### Start training  #########################")

        ##decay
        schedulers = []
        for client in clients:
            schedulers.append(torch.optim.lr_scheduler.StepLR(client.optimizer, step_size=10, gamma=0.998))

        active_clients = []

        best_val_acc = [-1 for client in clients]
        best_test_acc = [-1 for client in clients]
        best_agg_val_acc = [-1 for client in clients]
        best_agg_test_acc = [-1 for client in clients]

        for t in range(self.cfg.rounds):
            log.info("####### This is ROUND number {}".format(t))
            datamodule.set_client()
            self.client_id = 0
            train_loss_batch = []
            train_acc_batch = []

            # select new clients and distribute data among clients
            log.info("####### This is round starts with number of clients {}".format(self.cfg.m))
            if t == 0:
                available_clients = instantiate(self.cfg.selection, clients=clients,
                                                num_selected_clients=self.cfg.available_clients, t=t)
                active_clients = available_clients

            if self.cfg.sampling and t % self.cfg.h == 0:
                active_clients = instantiate(self.cfg.selection, clients=available_clients,
                                             num_selected_clients=self.cfg.m, t=t)
                if self.cfg.malicious_clients:
                    malicious_selected_clients = [client.client_id for client in active_clients if
                                                  client.client_id in malicious_client_ids]

                    benign_clients = [client.client_id for client in active_clients if client.client_id not in
                                      malicious_client_ids]
                    logging.info(f"benign clients are {benign_clients}")
                    logging.info(f"malicicous clients are: {malicious_selected_clients}")

            self.client_id = 0
            local_weights = {}

            train_loss, train_acc, local_val_test_acc, local_val_test_loss, local_test_acc, local_test_loss, local_val_agg_acc, local_val_agg_loss \
                , local_test_agg_acc, local_test_agg_loss, training_times, global_local_test_acc, global_local_test_loss = \
                [], [], [], [], [], [], [], [], [], [], [], [], []

            # if t % 5 == 0:
            #     for client in active_clients:
            #         saving_model(client,
            #                      f'./fedavg_models_checkpoint/client_{client.client_id}_num_client_{self.cfg.num_clients}_seed_{self.cfg.datamodule.seed}_split_{str(self.hydra_cfg["datamodule"])}_missing_{missing}_LclS_{self.cfg.client.local_steps}_sampling_{self.cfg.sampling}')

            for client in active_clients:
                log.info(" training client {} in round {}".format(client, t))
                start_time = time.time()
                results = client.train()
                end_time = time.time()
                training_times.append(end_time - start_time)
                train_acc.append(results["train_acc"])
                train_loss.append(results["train_loss"])
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

                logging.info(f"validation..")
                val_test_results = client.validation()
                if self.cfg.malicious_clients:
                    if client.client_id in benign_client_ids:
                        local_val_test_acc.append(val_test_results["global_val_acc"])
                        local_val_test_loss.append(val_test_results["global_val_loss"])
                else:
                    local_val_test_acc.append(val_test_results["global_val_acc"])
                    local_val_test_loss.append(val_test_results["global_val_loss"])

                if val_test_results["global_val_acc"] > best_val_acc[client.client_id]:
                    if self.cfg.malicious_clients:
                        if client.client_id in benign_client_ids:
                            best_val_acc[client.client_id] = val_test_results["global_val_acc"]
                    else:
                        best_val_acc[client.client_id] = val_test_results["global_val_acc"]

                    # saving_model(client,
                    #              f'./fedavg_models/client_{client.client_id}_num_client_{self.cfg.num_clients}_seed_{self.cfg.datamodule.seed}_split_{str(self.hydra_cfg["datamodule"])}_missing_{missing}_LclS_{self.cfg.client.local_steps}_sampling_{self.cfg.sampling}')

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



                # logging.info(f"global local test..")
                # global_local_test_results = test(client.model, self.device, global_test_set)
                # global_local_test_acc.append(global_local_test_results["global_val_acc"])
                # global_local_test_loss.append(global_local_test_results['global_val_loss'])

            train_acc_batch.append(sum(train_acc) / len(train_acc))
            train_loss_batch.append(sum(train_loss) / len(train_loss))

            ######## update model weights #######
            logging.info(f"global weight averaging ...")
            if self.cfg.robust_averaging:
                k = int(self.cfg.m * (1 - self.cfg.corrupted_ids))
                if self.cfg.averaging_method.knorm:
                    logging.info(f"robust averaging using knorm")
                    new_weights = knorm_average_weights(local_weights, k)
                elif self.cfg.averaging_method.krum:
                    new_weights = krum_average_weights(local_weights, k - 2)
                elif self.cfg.averaging_method.multi_krum:
                    new_weights = multi_krum_average_weights(local_weights, k-2, k)
            else:
                new_weights = average_weights(local_weights, data_weights=data_weights)

            global_model.load_state_dict(new_weights)
            for client in available_clients:
                client.model.load_state_dict(new_weights)

            ####### Saving last model ########
            if t == (self.cfg.rounds - 1):
                for client in available_clients:
                    if self.cfg.malicious_clients:
                        saving_model(client,
                                     f'./fedavg_models/client_{client.client_id}_num_client_{self.cfg.num_clients}_seed_{self.cfg.datamodule.seed}_split_{str(self.hydra_cfg["datamodule"])}_missing_{missing}_LclS_{self.cfg.client.local_steps}_sampling_{self.cfg.sampling}')
                    else:
                        saving_model(client,
                                     f'./fedavg_models/client_{client.client_id}_num_client_{self.cfg.num_clients}_seed_{self.cfg.datamodule.seed}_split_{str(self.hydra_cfg["datamodule"])}_missing_{missing}_LclS_{self.cfg.client.local_steps}_sampling_{self.cfg.sampling}')

            ######## TESTING for the global model ############
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

            print("########## wandb log ##################")
            if self.cfg.sampling:
                best_agg_val_acc_ = [x for x in best_agg_val_acc if x != -1]
                best_agg_test_acc_ = [x for x in best_agg_test_acc if x != -1]
                best_val_acc_ = [x for x in best_val_acc if x != -1]
                best_test_acc_ = [x for x in best_test_acc if x != -1]
                wandb_log(best_val_acc_, best_test_acc_, best_agg_val_acc_, best_agg_test_acc_, train_loss_batch,
                          train_acc_batch,
                          local_val_test_acc,
                          local_val_test_loss, local_val_agg_acc, local_val_agg_loss, local_test_agg_acc,
                          local_test_agg_loss,
                          local_test_acc, local_test_loss, t)

            ########### WANDB LOG ###############
            else:
                wandb_log(best_val_acc, best_test_acc, best_agg_val_acc, best_agg_test_acc, train_loss_batch,
                          train_acc_batch,
                          local_val_test_acc,
                          local_val_test_loss, local_val_agg_acc, local_val_agg_loss, local_test_agg_acc,
                          local_test_agg_loss,
                          local_test_acc, local_test_loss, t)

            # if self.cfg.malicious_clients:
            #     wandb.log({'malicious/number_of_malicious_clients': len(malicious_selected_clients),
            #                })
