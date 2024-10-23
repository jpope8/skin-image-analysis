"""
Optuna example that optimizes multi-layer perceptrons using PyTorch.

In this example, we optimize the validation accuracy of fashion product recognition using
PyTorch and FashionMNIST. We optimize the neural network architecture as well as the optimizer
configuration. As it is too time consuming to use the whole FashionMNIST dataset,
we here use a small subset of it.

"""

import os

import optuna
from optuna.trial import TrialState
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data
import time
from torchvision import datasets
from torchvision import transforms

# Will attempt num trails or timeout, whichever occurs first
import json
TRIALS = 100
TIME_LIMIT_HOURS = 16

SAMPLE_SIZE = 300
#DEVICE = torch.device("cpu")
#DEVICE = torch.device("mps")
# Set the device (str that the model and data will use)
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
BATCHSIZE = 32
CLASSES = 2
DIR = "./tone"
EPOCHS = 10
N_TRAIN_EXAMPLES = BATCHSIZE * 12
N_VALID_EXAMPLES = BATCHSIZE * 4

import tone_bias_dataset as dataset
from tone_bias_dataset import HibaDataset
from tone_bias_dataset import Rescale
from tone_bias_dataset import ToTensor
import torchvision

class TrialDummy:
    """
    Simple wrapper around dict to make compatible with Optuna trial object.
    Can use this to create models that exactly replicate one found during tuning.
    """
    def __init__(self, hyperparameters):
        self.hyperparameters = hyperparameters

    def put(self, key, value):
        self.hyperparameters[key] = value

    def get(self, key):
        return self.hyperparameters[key]

    def suggest_int(self, key, min_value, max_value):
        # Do not really need to bounds check but keeps consistent with tuning
        value = self.get(key)
        if value < min_value or min_value > max_value:
            raise ValueError(f"Expected value between in [{min_value},{max_value}] but got {value}")
        return int(value)

    def suggest_float(self, key, min_value, max_value):
        # Do not really need to bounds check but keeps consistent with tuning
        value = self.get(key)
        if value < min_value or min_value > max_value:
            raise ValueError(f"Expected value between in [{min_value},{max_value}] but got {value}")
        return float(value)

    def __str__(self):
        return str(self.hyperparameters)

def create_best_hyperparameters():

    # MANUAL SETTINGS
    hyperparameters = {
        'n_conv_layers': 2,
        'n_units_l0': 32,
        'n_units_conv_l0': 64,
        'n_units_conv_l1': 128,
        'n_linear_layers': 2,
        'n_units_linear_l0': 512,
        'dropout_l0': 0.5,
        'n_units_linear_l1': 256,
        'dropout_l1': 0.5,
        'optimizer': 'Adam',
        'lr': 2.2066163621947597e-05,
    }

    # TRIALS=100, SAMPLESIZE=96*2, EPOCSH=10
    hyperparameters = {
        'n_conv_layers': 3,
        'n_units_l0': 192,
        'n_units_conv_l0': 172,
        'n_units_conv_l1': 22,
        'n_units_conv_l2': 86,
        'n_linear_layers': 3,
        'n_units_linear_l0': 227,
        'dropout_l0': 0.4750108276372097,
        'n_units_linear_l1': 80,
        'dropout_l1': 0.33605861431570366,
        'n_units_linear_l2': 86,
        'dropout_l2': 0.26780264501531464,
        'optimizer': 'Adam',
        'lr': 0.03627331743927454}


    trial = TrialDummy(hyperparameters)
    return trial

def create_best_model():
    classes = CLASSES
    trial = create_best_hyperparameters()
    model = define_isic_model(classes, trial)
    return model


def define_isic_model(classes, trial):
    # We optimize the number of layers, hidden units and dropout ratio in each layer.
    n_conv_layers = trial.suggest_int("n_conv_layers", 1, 6)
    layers = []

    # in_features = 28 * 28
    #in_features = 224 * 224
    image_size = 224
    in_features = 3

    # Add first "special" layer with larger perceptive field
    out_features = trial.suggest_int("n_units_l0", 16, 256)
    layers.append(nn.Conv2d(in_channels=in_features, out_channels=out_features,
                            kernel_size=7, stride=1, padding='same'))
    layers.append(nn.ReLU())
    layers.append(nn.MaxPool2d(kernel_size=(2, 2))) #reduce w,h by half
    image_size = image_size // 2

    in_features = out_features

    for i in range(n_conv_layers):
        out_features = trial.suggest_int(f"n_units_conv_l{i}", 16, 256)
        #nn.Conv2d(in_channels=3, out_channels=32, kernel_size=7, stride=1, padding='same')
        #layers.append(nn.Linear(in_features, out_features))
        layers.append(nn.Conv2d(in_channels=in_features, out_channels=out_features,
                                kernel_size=3, stride=1, padding='same'))
        layers.append(nn.ReLU())
        layers.append(nn.MaxPool2d(kernel_size=(2, 2)))  # reduce w,h by half
        image_size = image_size // 2
        #if i < 6 :
        #    # Careful not to reduce w,h too much, halfing 6 times supports 224x224 images
        #
        in_features = out_features

    n_linear_layers = trial.suggest_int("n_linear_layers", 2, 5)
    layers.append(nn.Flatten())
    in_features = out_features * (image_size * image_size) # found empirically?
    # Amazingly this is computed correctly!
    print(f"DEBUGGING {in_features} = {out_features} * ({image_size}*{image_size})")
    for i in range(n_linear_layers):
        out_features = trial.suggest_int(f"n_units_linear_l{i}", 16, 256)
        layers.append(nn.Linear(in_features, out_features))
        layers.append(nn.ReLU())
        p = trial.suggest_float(f"dropout_l{i}", 0.2, 0.5)
        layers.append(nn.Dropout(p))
        in_features = out_features

    layers.append(nn.Linear(in_features, classes))
    layers.append(nn.LogSoftmax(dim=1))

    return nn.Sequential(*layers)

def get_isic():
    isic_metadata_df = dataset.read_isic_metadata(DIR)
    class_names = ['benign', 'malignant']

    # Shuffle the instances first
    # ### Shuffle the dataframe and reset index (has original, do not believe we need to keep?)
    isic_metadata_df = isic_metadata_df.sample(frac=1)  # effectively shuffle
    isic_metadata_df = isic_metadata_df.reset_index(drop=True)

    # Now split into train and test dataframes (will become train and test sets)
    total = isic_metadata_df.shape[0]
    ratio = 2.0/3.0
    #sample_size = 300 # takes about 90 minutes for each trial
    #sample_size = 96
    train_size = int(ratio * SAMPLE_SIZE)
    val_size = SAMPLE_SIZE - train_size
    # Nice because we can just use slicing to get train/test after shuffling
    train_df = isic_metadata_df[0:train_size]
    train_df = train_df.reset_index(drop=True)
    valid_df = isic_metadata_df[-val_size:]
    valid_df = valid_df.reset_index(drop=True)

    print(f"class_names={class_names} target size {len(class_names)}")
    print(f"train={len(train_df)} ({train_size})  val={len(valid_df)} ({val_size})")

    train_dataset = HibaDataset(train_df,
                                class_names,
                                root_dir="tone",
                                transform=torchvision.transforms.Compose([
                                    Rescale((224, 224)),
                                    # RandomCrop(224),
                                    ToTensor()
                                ]))

    valid_dataset = HibaDataset(valid_df,
                                class_names,
                                root_dir="tone",
                                transform=torchvision.transforms.Compose([
                                    Rescale((224, 224)),
                                    # RandomCrop(224),
                                    ToTensor()
                                ]))

    # =========================================================================#
    # Create batches for the data, the num_workers=num_threads is a sore subject
    # Define optimisation parameters for loading data
    # No GPU/TPU involved but multi-threading huge performance boost
    #   tune the num_threads to your CPU
    #   tune the batch size to your RAM
    # =========================================================================#
    num_threads = 10  # assuming 16 CPUs
    # Note that perhaps a better name for Dataloader is Batcher as it is where the batches are produced.
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCHSIZE, shuffle=True,
                                               num_workers=num_threads)
    valid_loader   = torch.utils.data.DataLoader(valid_dataset, batch_size=BATCHSIZE, shuffle=True,
                                               num_workers=num_threads)
    return train_loader, valid_loader


def objective(trial):
    # Generate the model.
    model = define_isic_model(CLASSES, trial).to(DEVICE)

    # Generate the optimizers.
    optimizer_name = trial.suggest_categorical("optimizer", ["Adam", "RMSprop", "SGD"])
    lr = trial.suggest_float("lr", 1e-5, 1e-1, log=True)
    optimizer = getattr(optim, optimizer_name)(model.parameters(), lr=lr)

    # Get the ISIC dataset.
    train_loader, valid_loader = get_isic()

    # Training of the model.
    for epoch in range(EPOCHS):
        model.train()
        for batch_idx, (data, target, index) in enumerate(train_loader):
            # Limiting training data for faster epochs.
            if batch_idx * BATCHSIZE >= N_TRAIN_EXAMPLES:
                break

            #data, target = data.view(data.size(0), -1).to(DEVICE), target.to(DEVICE)
            data = data.to(DEVICE)
            target = target.to(DEVICE)

            optimizer.zero_grad()
            output = model(data)
            loss = F.nll_loss(output, target)
            loss.backward()
            optimizer.step()

        # Validation of the model.
        model.eval()
        correct = 0
        with torch.no_grad():
            for batch_idx, (data, target, index) in enumerate(valid_loader):
                # Limiting validation data.
                if batch_idx * BATCHSIZE >= N_VALID_EXAMPLES:
                    break
                #data, target = data.view(data.size(0), -1).to(DEVICE), target.to(DEVICE)
                data = data.to(DEVICE)
                target = target.to(DEVICE)
                output = model(data)
                # Get the index of the max log-probability.
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()

        accuracy = correct / min(len(valid_loader.dataset), N_VALID_EXAMPLES)

        trial.report(accuracy, epoch)

        # Handle pruning based on the intermediate value.
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

    return accuracy


def main():
    print(f"SAMPLE_SIZE: {SAMPLE_SIZE}")
    print(f"     EPOCHS: {EPOCHS}")
    print(f"     DEVICE: {DEVICE}")
    print(f"  BATCHSIZE: {BATCHSIZE}")

    start_time = time.time()
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=TRIALS, timeout=3600 * TIME_LIMIT_HOURS)
    epoch_time = time.time() - start_time
    print(f"Hyperparameter search time: {epoch_time:.2f}s")

    pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
    complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])

    print("Study statistics: ")
    print("  Number of finished trials: ", len(study.trials))
    print("  Number of pruned trials: ", len(pruned_trials))
    print("  Number of complete trials: ", len(complete_trials))

    print("Best trial:")
    trial = study.best_trial

    print("  Value: ", trial.value)

    print(f"  Params (type {type(trial.params)}): ")
    for key, value in trial.params.items():
        print("    '{}': {},".format(key, value))

    print(f"SAMPLE_SIZE: {SAMPLE_SIZE}")
    print(f"     EPOCHS: {EPOCHS}")
    print(f"     DEVICE: {DEVICE}")
    print(f"  BATCHSIZE: {BATCHSIZE}")

    # Save to file to be read back in later
    # Note that trial.params is a dict
    filename = f"optuna_{TRIALS}_{EPOCHS}_{DEVICE}_{BATCHSIZE}_{SAMPLE_SIZE}.json"
    trial.params['TRIALS'] = TRIALS
    trial.params['EPOCHS'] = EPOCHS
    trial.params['DEVICE'] = DEVICE
    trial.params['BATCHSIZE'] = BATCHSIZE
    trial.params['SAMPLE_SIZE'] = SAMPLE_SIZE
    with open(filename, 'w') as f:
        # Fails
        #json.dump(trial.params, f)
        lines = list()
        lines.append("{")
        for key, value in trial.params.items():
            lines.append( "    '{}': {},".format(key, value) )
        lines.append("}")
        f.writelines(lines)
    #with open('dict.txt', 'r') as f:
    #    loaded_dict = json.load(f)

if __name__ == "__main__":
    main()
