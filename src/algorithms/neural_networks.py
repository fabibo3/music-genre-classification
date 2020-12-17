"""
Neural Network architectures and training procedures
"""
__author__ = "Fabian Bongratz"

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import os
from datasets.datasets import MusicDataset
from utils.utils import accuracy
from utils.folders import get_experiment_folder
from sklearn.model_selection import train_test_split
from typing import Union

_n_epochs_key = "epochs"
_learning_rate_key = "learning_rate"

def run_nn_model(model_path,
                 test_dataset: torch.utils.data.Dataset,
                 experiment_name: str) -> dict:
    """
    Apply a neural network model to a test dataset
    ------
    @param model_path: The path where a trained model can be found
    @param test_dataset: The test data
    @param experiment_name: The folder name of the current experiment
    ------
    @return predictions for the test data
    """
    # Test dataset
    if(type(test_dataset)==MusicDataset):
        test_files, X_test, _ = test_dataset.get_whole_dataset_labels_zero_based()
    else:
        test_files, X_test, _ = test_dataset.get_whole_dataset()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    X_test = torch.tensor(X_test).float().to(device)

    # Apply model
    model = torch.load(model_path, map_location=device)

    # Predict
    print('Predicting classes...')
    result = model(X_test)
    result = np.argmax(result.detach().cpu().numpy(), axis=1)
    predictions = {}
    for i, file_id in enumerate(test_files):
        predictions[file_id] = result[i]+1 # Labels not zero_based in the nn model

    return predictions

def search_nn_parameters(config: dict,
                         experiment_name: str,
                         train_dataset: torch.utils.data.dataset,
                         val_dataset: torch.utils.data.dataset=None):
    """
    Fit a neural network model using train and validation set
    @param config: A dict containing training parameters and parameter ranges
    @param train_dataset: The training split, either as a MusicDataset or as a
    tuple (data, labels)
    @param val_dataset: The validation split, see train_split
    @param internal_cv: True if cross validation should be done internally
    during training
    ------
    @return:
        - a list of the names of the parameters
        - a list of tried parameter configurations
        - a list of corresponding results
    """
    # Get parameters
    if(type(config[_n_epochs_key])==list):
        epochs = np.arange(config[_n_epochs_key][0],
                                config[_n_epochs_key][1],
                                config[_n_epochs_key][2])
    else:
        epochs = config[_n_epochs_key]

    if(type(config[_learning_rate_key])==list):
        learning_rates = np.arange(config[_learning_rate_key][0],
                                config[_learning_rate_key][1],
                                config[_learning_rate_key][2])
    else:
        learning_rates = config[_learning_rate_key]
        learning_rates = [learning_rates]

    model_architecture = config.get("architecture", "SmallNet5")

    parameter_names = []
    parameter_sets = []
    results = []

    # Get validation data
    if(val_dataset != None):
        if(type(val_dataset)==MusicDataset):
            _, X_val, y_val = val_dataset.get_whole_dataset_labels_zero_based()
        else:
            _, X_val, y_val = val_dataset.get_whole_dataset()

    # GPU
    if(torch.cuda.is_available()):
        task_type = 'GPU'
        devices = str(torch.cuda.current_device())
    else:
        task_type = 'CPU'
        devices = None

    params = config.copy()

    best_value = 0.0
    best_iter = 0
    for i_lr, lr in enumerate(learning_rates):
        params['learning_rate'] = lr
        if(model_architecture=='CNN_2'):
            model = CNN_2(train_dataset.datashape[1:],
                          dropout_prob=config.get("dropout_prob", 0.5))
        else:
            # SmallNet5 by default
            model = SmallNet5(len(train_dataset[0]))

        # Train model
        trained_model, \
                best_epoch,\
                best_res_out = train_model(model,
                                params,
                                train_dataset,
                                val_dataset,
                                experiment_name)
        params['best_epoch'] = best_epoch

        print(f"Best epoch for lr {lr}: {best_epoch} with val accuracy\
              {best_res_out}")

        results.append(best_res_out)
        parameter_sets.append(list(params.values()))
        parameter_names = list(params.keys())


    return parameter_names, parameter_sets, results

def train_model(model: torch.nn.Module,
                config: dict,
                train_data: torch.utils.data.dataset,
                val_data: torch.utils.data.dataset,
                experiment_name: str):
    """
    Training procedure.
    ----------
    @param model: torch.nn.Module, Model object initialized from a torch.nn.Module
    @param config: dict, Training parameters
    @param train_data: The training dataset
    @param val_data: The validation dataset
    @param experiment_name: str, The id of the current experiment
    ------
    @return
    -------
        - The trained model
        - Best epoch according to validation set and evaluation measure. If no
        validation set is provided, this is w.r.t. the best loss achieved
        - Best value of validation measure or best loss
    """
    # Init
    best_epoch = 0
    best_train_loss = 0
    best_val_measure = 0
    best_state = model.state_dict()
    train_loss_history = []
    val_measure_history = []

    # Define parameters
    train_params = {}
    eval_metric = config.get('eval_metric', 'Accuracy')
    loss_function = config.get('loss_function', 'CrossEntropy')
    epochs = config.get(_n_epochs_key, 10)
    learning_rate = config.get(_learning_rate_key, 0.1)
    weight_decay = config.get('weight_decay', 0.0)
    early_stop = config.get('early_stop', False)
    lr_decay_every = config.get('lr_decay_every', 100)
    decay_rate = config.get('decay_rate', 0.1)
    log_nth = config.get('log_nth', 1)
    batch_size = config.get('batch_size', len(train_data))
    n_iter_per_epoch = np.ceil(len(train_data)/batch_size).astype(int)

    # Optimizer
    default_adam_args = {"lr": 1e-4,
                         "betas": (0.9, 0.999),
                         "eps": 1e-8,
                         "weight_decay": weight_decay}
    optim=torch.optim.Adam(model.parameters(), **default_adam_args)
    lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optim,
                                           gamma=decay_rate)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.float()

    if(val_data is not None):
        _ , X_val, label_val = val_data.get_whole_dataset()
        validate = True
        X_val = torch.tensor(X_val).float().to(device)
        label_val = torch.tensor(label_val).long()
        print(f'Using {X_val.shape[0]} validation samples in training procedure')
    else:
        print("No validation set used in training procedure")
        validate = False

    train_loader = torch.utils.data.DataLoader(train_data,
                                               batch_size=batch_size)

    if(loss_function=='CrossEntropy'):
        loss_function = nn.CrossEntropyLoss()
    else:
        raise ValueError("Loss function not recognized")

    print('#'*50)
    print('START TRAIN ON {}.'.format(device))
    print(f'Using {len(train_data)} training samples')

    # Iterate through epochs
    for epoch in range(1, epochs + 1):
        for iteration, (_, X_train, label_train) in enumerate(train_loader):
            model.train()
            X_train, label_train = X_train.float(), label_train.long()

            X_train, label_train = X_train.to(device), label_train.to(device)
            optim.zero_grad()
            # Forward pass
            output = model(X_train)
            # Loss
            loss = loss_function(output, label_train)
            # Backprop
            loss.backward()
            optim.step()

            train_loss_history.append(loss.detach().item())

            if iteration % log_nth == 0:
                print('[Iteration {} / {}] TRAIN LOSS: {:.3f}'.format(
                    iteration, n_iter_per_epoch, loss))

        if(epoch == 1 or best_train_loss > loss):
            best_train_loss = loss

        # Evaluate on validation set
        if(validate):
            model.eval()
            pred = model(X_val)
            if(eval_metric == 'Accuracy'):
                pred = np.argmax(pred.detach().cpu().numpy(), axis=1)
                val_mes = accuracy(pred, label_val.numpy())
            print('[Epoch {} / {}] VAL acc: {:.3f}'.format(
                        epoch, epochs, val_mes))

            # Evaluate on validation set
            val_measure_history.append(val_mes)

            # Save currently best model according to validation set
            if(eval_metric == 'Accuracy'):
                # Maximum?
                if(epoch == 1 or best_val_measure < val_mes):
                    best_val_measure = val_mes
                    best_state = model.state_dict()
                    best_epoch = epoch

        # Store best model according to loss
        else:
            if(epoch == 1 or best_train_loss > loss):
                best_state = model.state_dict()
                best_epoch = epoch


        # Decay learning rate
        if(epoch % lr_decay_every == 0):
            lr_scheduler.step()

    # Reload best model
    print(f"Best model in epoch {best_epoch}.\n")
    if(early_stop):
        print(f"Reload model from epoch {best_epoch}")
        model.load_state_dict(best_state)
    # Store in eval mode to avoid problems
    model.eval()
    model.save(os.path.join(get_experiment_folder(experiment_name),"best.model"))

    log_file_path =\
    os.path.join(get_experiment_folder(experiment_name),"log.txt")
 
    # Write to file
    with open(log_file_path, 'w') as out_file:
        out_file.write("Training loss history:\n")
        for t in train_loss_history:
            out_file.write("{}\n".format(t))

    print('FINISH training.')

    if(X_val != None):
        best_res_out = best_val_measure
    else:
        best_res_out = best_train_loss

    fig, ax1 = plt.subplots()
    color = 'tab:blue'
    ax1.plot(train_loss_history, color=color)
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss", color=color)
    ax2 = ax1.twinx()
    color = 'tab:red'
    ax2.set_ylabel("acc", color=color)
    ax2.plot(np.arange(n_iter_per_epoch, epochs*n_iter_per_epoch+1,
                       n_iter_per_epoch),
             val_measure_history, color=color)
    fig.tight_layout()
    plt.savefig(os.path.join(get_experiment_folder(experiment_name),
                             "loss_plot.png"))

    return model, best_epoch, best_res_out

class SmallNet3(nn.Module):
    """
    Implementation of a small neural network consisting of 3 fully connected
    hidden layers and ReLU nonlinearities
    """
    def __init__(self, input_dim, channels=(1024, 512, 256), num_classes=8):
        """
        @param input_dim: The dimension of the input vectors
        @param channels: The channel dimensions of the five layers
        @param num_classes: The number of output classes
        """

        super(SmallNet3, self).__init__()
        self.num_classes = num_classes
        self.layer1 = nn.Linear(input_dim, channels[0], bias=True)
        self.layer2 = nn.Linear(channels[0], channels[1], bias=True)
        self.layer3 = nn.Linear(channels[1], channels[2], bias=True)
        self.layer4 = nn.Linear(channels[2], num_classes, bias=False)

    def forward(self, X):
        """
        @param X: The data that should be passed through the network
        ------
        @return: The output of the network
        """
        out = F.relu(self.layer1(X))
        out = F.relu(self.layer2(out))
        out = F.relu(self.layer3(out))
        out = self.layer4(out)

        return out

    def save(self, path):
        """
        Save model with its parameters to the given path. Conventionally the
        path should end with "*.model".

        Parameters
        ----------
        path: str
            The path where the model should be saved.
        """

        print('Saving model... {}'.format(path))
        torch.save(self,path)


class SmallNet5(nn.Module):
    """
    Implementation of a small neural network consisting of 5 fully connected
    hidden layers and ReLU nonlinearities
    """
    def __init__(self, input_dim, channels=(2048, 1024, 512, 256, 128), num_classes=8):
        """
        @param input_dim: The dimension of the input vectors
        @param channels: The channel dimensions of the five layers
        @param num_classes: The number of output classes
        """

        super(SmallNet5, self).__init__()
        self.num_classes = num_classes
        self.layer1 = nn.Linear(input_dim, channels[0], bias=True)
        self.layer2 = nn.Linear(channels[0], channels[1], bias=True)
        self.layer3 = nn.Linear(channels[1], channels[2], bias=True)
        self.layer4 = nn.Linear(channels[2], channels[3], bias=True)
        self.layer5 = nn.Linear(channels[3], channels[4], bias=True)
        self.layer6 = nn.Linear(channels[4], num_classes, bias=False)

    def forward(self, X):
        """
        @param X: The data that should be passed through the network
        ------
        @return: The output of the network
        """
        out = F.relu(self.layer1(X))
        out = F.relu(self.layer2(out))
        out = F.relu(self.layer3(out))
        out = F.relu(self.layer4(out))
        out = F.relu(self.layer5(out))
        out = self.layer6(out)

        return out

    def save(self, path):
        """
        Save model with its parameters to the given path. Conventionally the
        path should end with "*.model".

        Parameters
        ----------
        path: str
            The path where the model should be saved.
        """

        print('Saving model... {}'.format(path))
        torch.save(self,path)

class CNN_2(nn.Module):
    """
    Implementation of a small convolutional neural network consisting of two
    convolutional and one fully connected layer. It is described in this blog
    post:
        https://towardsdatascience.com/musical-genre-classification-with-convolutional-neural-networks-ff04f9601a74
    """
    def __init__(self, input_dim, channels=(16, 32, 64), num_classes=8,
                 dropout_prob=0.5):
        """
        @param input_dim: The dimension of the input image in the form
        (time_dim, mel_dim)
        @param channels: The channel dimensions of the three layers
        @param num_classes: The number of output classes
        @param dropout_prob: The dropout probability of the last hidden layer
        """

        super(CNN_2, self).__init__()
        self.num_classes = num_classes
        pool_filter_size = (3,2)
        self.layer1 = nn.Conv2d(1, channels[0], 3)
        out_size = [i-2 for i in input_dim[1:]]
        self.max_pool1 = nn.MaxPool2d(pool_filter_size)
        out_size = [(int)(i/p) for i,p in zip(out_size, pool_filter_size)]
        self.layer2 = nn.Conv2d(channels[0], channels[1], 3)
        out_size = [i-2 for i in out_size]
        self.max_pool2 = nn.MaxPool2d(pool_filter_size)
        out_size = [(int)(i/p) for i,p in zip(out_size, pool_filter_size)]
        self.fc_layer1 = nn.Linear(channels[1]*out_size[0]*out_size[1], channels[2], bias=True)
        self.dropout = nn.Dropout(p=dropout_prob)

        self.out_layer = nn.Linear(channels[2], num_classes,  bias=True)

    def forward(self, X):
        """
        @param X: The data that should be passed through the network
        ------
        @return: The output of the network
        """
        # Conv layers
        out = F.relu(self.layer1(X))
        out = self.max_pool1(out)
        out = F.relu(self.layer2(out))
        out = self.max_pool2(out)

        # FC layer
        out = out.view(X.shape[0], -1)
        out = F.relu(self.fc_layer1(out))
        out = self.dropout(out)
        out = self.out_layer(out)

        return out

    def save(self, path):
        """
        Save model with its parameters to the given path. Conventionally the
        path should end with "*.model".

        Parameters
        ----------
        path: str
            The path where the model should be saved.
        """

        print('Saving model... {}'.format(path))
        torch.save(self,path)
