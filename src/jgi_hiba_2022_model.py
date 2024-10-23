#!/usr/bin/env python
# coding: utf-8

"""
This module provides models suitable for taking images and
predicting {benign,malignant} for skin cancer images.

The generic architecture is as follows

  * ("conv block" is conv, relu, max pooling layers)
  * ("linear block" is linear, relu, dropout layers)
  * initial conv block
  * n conv_blocks
  * flatten layer
  * m linear_blocks
  * next to last layer linear output len(num_classes)
  * last layer log softmax

Two models are provided using this generic architecture
  * model with manually determine hyperparameters
  * model with variable number of conv/linear blocks and variable hyperparameters

The manual model is easier to understand and the variable model is determined via hyperparameter search.

The module also provides functions for saving and loading these models.
"""

import tone_bias_optuna as optuna
import tone_bias_dataset as dataset

import sys
import torch
import torch.nn as nn

"""
When to use CrossEntropyLoss versus NLLLoss in PyTorch

CrossEntropyLoss:
Use when designing a neural network for multi-class classification with no activation
function applied to the output (i.e., the output is a vector of unnormalized scores or logits).
This is the default and recommended choice for multi-class classification problems.

NLLLoss:
Use when designing a neural network for multi-class classification with a log-SoftMax
activation function applied to the output (i.e., the output is a vector of probabilities).
This combination is equivalent to using CrossEntropyLoss, but with log probability predictions as inputs.
Key differences:

CrossEntropyLoss expects unbounded scores (logits) as input,
while NLLLoss expects log probability inputs.

CrossEntropyLoss is the default and recommended choice for multi-class
classification, while NLLLoss is an alternative when using log-SoftMax activation.
"""

class SkinCancerListModel(nn.Module):
    """
    Custom CNN model with series of convolutional blocks, flatten, linear layers blocks.
    Use list of layers (i.e. Sequential model), flexible number of convolutional and linear blocks.
    Expects an input of 3 channels, width 224, height 224.  Outputs LogSoftMax {benign,malignant}.
    """
    def __init__(self, class_names):
        super().__init__()
        self.class_names = class_names

        layers = list()

        # Expects images of h=224, w=224, first conv block is special and has larger kernel size
        width = 224
        height= 224
        in_features = 3 # initial input features, which are the color channels

        # =======================================#
        # Add the convolutional blocks
        # =======================================#
        # input (i-1) and output (i) features for each layer, number of layers is len(conv_features)
        conv_features = [32, 64, 128] # these are the out_features for layer i
        for i in range( len(conv_features) ):
            out_features = conv_features[i]

            # first block we use a larger kernel size
            kernel = 7 if i == 0 else 3
            conv_layer = nn.Conv2d(in_channels=in_features, out_channels=out_features,
                                   kernel_size=kernel, stride=1, padding='same')
            self._init_weights(conv_layer) # trainable layer, init weights
            layers.append(conv_layer)

            # layers.append( nn.BatchNorm2d(num_features=32) )
            layers.append(nn.ReLU())
            # Need to keep track of width x height to determine number input for linear after flatten
            # So every time there is a MaxPool2d, half the width x height
            layers.append(nn.MaxPool2d(kernel_size=(2, 2)))
            width = width // 2
            height= height// 2

            # Update for next loop, sequential is input becomes output for next layer
            in_features = out_features

        # Critical layer in the architecture where we reduce the volume (w x h x f) to a vector
        layers.append( nn.Flatten() )

        #=======================================#
        # Add the linear blocks
        # =======================================#
        # out_features from conv after flattening
        in_features = out_features * width * height # NB: out_features from last conv loop = conv_features[-1]
        linear_features = [512, 256]
        for i in range(len(linear_features)):
            out_features = linear_features[i]

            linear_layer = nn.Linear(in_features, out_features)
            self._init_weights(linear_layer) # trainable layer, init weights
            layers.append(linear_layer)
            layers.append( nn.ReLU() )
            layers.append(nn.Dropout(0.5))

            # Update for next loop, sequential is input becomes output for next layer
            in_features = out_features

        # =======================================#
        # Add the output block
        # =======================================#
        # Option determines which loss function you MUST use to train the model
        # https://pytorch.org/docs/stable/generated/torch.nn.NLLLoss.html#torch.nn.NLLLoss
        # # Requires last layer to be LogSoftMax)
        classification_layer = nn.Linear(out_features, len(class_names))
        self._init_weights(classification_layer)  # trainable layer, init weights
        layers.append(classification_layer)
        layers.append(nn.LogSoftmax(dim=1))

        # Add all the layers to sequential layer, use it in the forward pass
        # Not sure why sequential does not just take a list, but apparently takes variable arg
        # So we have to unpack the list before calling constructor
        self.layers = nn.Sequential(*layers)

    def _init_weights(self, module):
        nn.init.xavier_normal_(module.weight)

    def forward(self, x):
        # Sequential has its own forward that passes putput of previous layer as input to next layer
        # However, useful for debugging to use a custom forward (easy enough to loop over layers)
        # Initialize an empty list to store intermediate outputs
        #outputs = []
        verbose = False
        for layer in self.layers:
            x = layer(x)
            #outputs.append(x)
            if verbose: print(f"layer {type(layer)} {x.shape}")
        return x

    def get_class_names(self):
        return self.class_names


class SkinCancerModel(nn.Module):
    """
    Custom CNN model with series of convolutional blocks, flatten, linear layers blocks.
    NOTE: USE THE SkinCancerListModel, THIS MODEL WAS EXPERIMENTAL AND THAT MODEL IS MORE FLEXIBLE.
    """
    def __init__(self, class_names):
        super().__init__()
        self.class_names = class_names

        #=====================================#
        # Old approach without sequential list
        #=====================================#

        # Expects images of h=224, w=224
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=7, stride=1, padding='same')
        #self.norm1 = nn.BatchNorm2d(num_features=32)
        self.act1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=(2, 2))

        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding='same')
        self.act2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=(2, 2))

        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding='same')
        self.act3 = nn.ReLU()
        self.pool3 = nn.MaxPool2d(kernel_size=(2, 2))

        self.conv4 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding='same')
        self.act4 = nn.ReLU()
        self.pool4 = nn.MaxPool2d(kernel_size=(2, 2))

        self.flat = nn.Flatten()

        # 14 = 224 / 2/ 2 /2 / 2
        self.fc4 = nn.Linear(256 * 14 * 14, 512)
        self.act4 = nn.ReLU()
        self.drop4 = nn.Dropout(0.5)

        self.fc5 = nn.Linear(512, 256)
        self.act5 = nn.ReLU()
        self.drop5 = nn.Dropout(0.5)

        # Option determines which loss function you MUST use to train the model

        # https://pytorch.org/docs/stable/generated/torch.nn.NLLLoss.html#torch.nn.NLLLoss
        # # Requires last layer to be LogSoftMax)
        self.fc6 = nn.Linear(256, len(class_names))
        self.logsoftmax = nn.LogSoftmax(dim=1)

        # https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html#torch.nn.CrossEntropyLoss
        # Requires last layer to be Linear
        # self.fc6 = nn.Linear(256, len(class_names))

        # Initialise
        self._init_weights(self.conv1)
        self._init_weights(self.conv2)
        self._init_weights(self.conv3)
        self._init_weights(self.conv4)
        self._init_weights(self.fc4)
        self._init_weights(self.fc5)
        self._init_weights(self.fc6)
        # self._init_weights(self.softmax)

    def _init_weights(self, module):
        # print(f"Initialising module {module}")
        # https://saturncloud.io/blog/how-to-initialize-weights-in-pytorch-a-guide-for-data-scientists/
        # https://wandb.ai/wandb_fc/tips/reports/How-to-Initialize-Weights-in-PyTorch--VmlldzoxNjcwOTg1
        nn.init.xavier_normal_(module.weight)
        # nn.init.kaiming_normal_(module.weight, mode='fan_in', nonlinearity='relu') # He
        # if isinstance(module, nn.Linear):
        #    module.weight.data.normal_(mean=0.0, std=1.0)
        #    if module.bias is not None:
        #        module.bias.data.zero_()

    def forward(self, x):
        verbose = False
        # =========================#
        # input 3x224x224
        x = self.act1(self.conv1(x))
        #x = self.norm1(x)
        # output 32x224x224

        # input 32x224x224
        x = self.pool1(x)
        # output 32x112x112
        if verbose: print(f"output conv1 {x.shape}")

        # =========================#
        # input 32x112x112
        x = self.act2(self.conv2(x))
        # input 64x112x112

        # input 64x112x112
        x = self.pool2(x)
        # output 64x56x56
        if verbose: print(f"output conv2 {x.shape}")

        # =========================#
        # input 64x56x56
        x = self.act3(self.conv3(x))
        # input 128x56x56

        # input 128x56x56
        x = self.pool3(x)
        # output 128x28x28
        if verbose: print(f"output conv3 {x.shape}")

        # =========================#
        x = self.act4(self.conv4(x))
        x = self.pool4(x)
        if verbose: print(f"output conv4 {x.shape}")

        # =========================#
        # input 128x28x28
        x = self.flat(x)
        # output (128x28x28)
        if verbose: print(f"output flatten {x.shape}")

        # =========================#
        # input (128x28x28)
        x = self.act4(self.fc4(x))
        x = self.drop4(x)
        # output (512)
        if verbose: print(f"output fully connect 4 {x.shape}")

        # =========================#
        # input (512)
        x = self.act5(self.fc5(x))
        x = self.drop5(x)
        #  output (256)
        if verbose: print(f"output fully connect 5 {x.shape}")

        # =========================#
        # input (256)
        x = self.fc6(x)
        #  output len(classes)
        if verbose: print(f"output fully connect 6 {x.shape}")

        # Finally pass through a softmax
        x = self.logsoftmax(x)
        if verbose: print(f"output softmax {x.shape}")
        return x

    def get_class_names(self):
        return self.class_names

def create_loss_function():
    return nn.NLLLoss()
    #return nn.CrossEntropyLoss()

def save_model(model, model_path):
    """
    /Users/james/Documents/pytorch_test/env/lib/python3.11/site-packages/torch_geometric/data/dataset.py:238: FutureWarning: You are using `torch.load` with `weights_only=False` (the current default value), which uses the default pickle module implicitly. It is possible to construct malicious pickle data which will execute arbitrary code during unpickling (See https://github.com/pytorch/pytorch/blob/main/SECURITY.md#untrusted-models for more details). In a future release, the default value for `weights_only` will be flipped to `True`. This limits the functions that could be executed during unpickling. Arbitrary objects will no longer be allowed to be loaded via this mode unless they are explicitly allowlisted by the user via `torch.serialization.add_safe_globals`. We recommend you start setting `weights_only=True` for any use case where you don't have full control of the loaded file. Please open an issue on GitHub for any issues related to this experimental feature.

    Prefer following to get rid of the warning and save entire model (though maybe just weights are fine?)
    torch.save(model.state_dict(), model_path, weights_only=False)
    """
    # Have to use the pickle version to save model structure along with weights
    # https://pytorch.org/tutorials/beginner/basics/saveloadrun_tutorial.html
    #torch.save(model.state_dict(), model_path)
    torch.save(model, model_path)


def create_model(class_names):
    model = SkinCancerModel(class_names)
    return model


def load_model(model_path, class_names):
    """
    Loads previous model from specified path, with default device (i.e. cpu).
    returns model suitable for more training, testing, or deployment.
    """
    # Have to use the pickle version to save model structure along with weights
    # https://pytorch.org/tutorials/beginner/basics/saveloadrun_tutorial.html
    #model = SkinCancerModel(class_names)
    #model.load_state_dict(torch.load(model_path, weights_only=False))
    model = torch.load(model_path, weights_only=False)
    """
    # Prefer following to get rid of the warning and load entire model
    # though maybe just weights are fine?)
    model.load_state_dict( torch.load (model_path), weights_only=False )
    """
    return model


def main():
    #  Get some cli arguments for preprocessing, training, evaluation.
    if len(sys.argv) != 3:
        print(f"Usage: <root directory of ISIC images> <number of epochs for training>")
        print(f"Example: tone 20")
        return
    root_dir_name = sys.argv[1]  # root_dir_name = "./tone"
    epochs = int(sys.argv[2])  # epochs = 50

    # =========================================================================#
    # 1. Read in the metadata and filter to only have attributes for rows with tone
    # =========================================================================#
    isic_metadata_df = dataset.read_isic_metadata(root_dir_name)
    class_names = ['benign', 'malignant']

    # Shuffle the instances first
    # ### Shuffle the dataframe and reset index (has original, do not believe we need to keep?)
    isic_metadata_df = isic_metadata_df.sample(frac=1)  # effectively shuffle
    isic_metadata_df = isic_metadata_df.reset_index(drop=True)

    # Now split into train and test dataframes (will become train and test sets)
    total = isic_metadata_df.shape[0]
    ratio = 0.7
    train_size = int(ratio * total)
    test_size = total - train_size
    # Nice because we can just use slicing to get train/test after shuffling
    train_df = isic_metadata_df[:train_size]
    train_df = train_df.reset_index(drop=True)
    test_df = isic_metadata_df[train_size:]
    test_df = test_df.reset_index(drop=True)

    print(f"class_names={class_names} target size {len(class_names)}")
    print(f"train={len(train_df)} ({train_size})  test={len(test_df)} ({test_size})")
    # train_set, test_set = torch.utils.data.random_split(transformed_dataset, [train, test])

    # Check PyTorch has access to MPS (Metal Performance Shader, Apple's GPU architecture)
    print(f"Is MPS (Metal Performance Shader) built? {torch.backends.mps.is_built()}")
    print(f"Is MPS available? {torch.backends.mps.is_available()}")

    # Set the device (str that the model and data will use)
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Using device: {device}")

    # =====================================================================#
    # Create the mode and training objects
    # Check to see if a model was previously trained, can continue training
    # =====================================================================#
    print(f"Using new model, training start new.")
    #model = Net(class_names)  # create new model with initial weights
    model = optuna.create_best_model()
    print(f"Model {model}")
    # =====================================================================#


if __name__ == '__main__':
    main()

