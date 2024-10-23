#!/usr/bin/env python
# coding: utf-8

# # Prepare data and libraries
# Based on the Hospital Italiano de Buenos Aires - Skin Lesions Images (2019-2022).
# 
# * https://www.isic-archive.com/
# * https://api.isic-archive.com/collections/251/
# 
# See separate EDA for downloading the metadata (cvs file) and imiages using the cli "isic" tool.
# 
# Critically, this dataset has "skin type" using the Fitzpatrick scale.
# 
# * https://en.wikipedia.org/wiki/Fitzpatrick_scale

# standard libraries/modules
import os
import sys
import pandas as pd
import torch
import torchvision
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler
import time
import datetime as chronos

import json

from tempfile import TemporaryDirectory
import torch.optim as optim

# local libraries/modules
import monitor_processes
import tone_bias_model as model_module
import tone_bias_test as test_module

#from jgi_hiba_2022_model import SkinCancerModel
from tone_bias_model import SkinCancerListModel
import tone_bias_dataset as dataset
from tone_bias_dataset import HibaDataset
from tone_bias_dataset import Rescale
from tone_bias_dataset import ToTensor

import tone_bias_optuna as optuna


"""
Training crashed after exactly 16 epochs with
    trainingset=2526 images
    batch_size=16
    num_workers=10

with following error
    RuntimeError: Too many open files.
    Communication with the workers is no longer possible.
    Please increase the limit using `ulimit -n` in the shell or change the sharing strategy
    by calling `torch.multiprocessing.set_sharing_strategy('file_system')` at the beginning of your code
    
    # Due to bug, see https://github.com/pytorch/pytorch/issues/11201, does not work for our problem
    torch.multiprocessing.set_sharing_strategy('file_system')
    
After much debugging, traced to the following

    Epoch 0: process python3.11 (PID: 27164)  PPID: 13874  opened: 7  FDS: 101
    Epoch 1: process python3.11 (PID: 27164)  PPID: 13874  opened: 7  FDS: 111
    Epoch 2: process python3.11 (PID: 27164)  PPID: 13874  opened: 7  FDS: 121
    Epoch 3: process python3.11 (PID: 27164)  PPID: 13874  opened: 7  FDS: 131
    Epoch 4: process python3.11 (PID: 27164)  PPID: 13874  opened: 7  FDS: 141
    Epoch 5: process python3.11 (PID: 27164)  PPID: 13874  opened: 7  FDS: 151
    ...
    Epoch 15:process python3.11 (PID: 27164)  PPID: 13874  opened: 7  FDS: 251
    Epoch 16:process python3.11 (PID: 27164)  PPID: 13874  opened: 7  FDS: 261 (crashed before this)
    
Confirmed (using "ulimit -a") that the maximum number of descriptors per process is 256
    -n: file descriptors 256

Thought the image files being opened in the HibaDataset was the issue,
however, they appear to be closed/cleaned up properly, as can be seen in the sub-processes.
Note that the sub-processes properly shutdown each epoch and new ones are created for next epoch.
    Python process python3.11 (PID: 27165)  PPID: 27164  opened: 0  FDS: 4
    Python process python3.11 (PID: 27338)  PPID: 27164  opened: 0  FDS: 23
    Python process python3.11 (PID: 27339)  PPID: 27164  opened: 0  FDS: 23
    Python process python3.11 (PID: 27340)  PPID: 27164  opened: 0  FDS: 23
    ...

First thought that because 10 are file descriptors are being added each epoch
and we have specified num_workers=10, file descriptors are being added to the
process but never removed. This results in 261 file descriptors in epoch 16 which
exceeds the 256 limit.

The num_workers is the problem but also the number of times we printout affects this.
If we only have a a few batches (less than num_workers) then the number of print outs
is the number added.  My guess is that the printout requires the sub-process to 
attach itself to the parent as a file descriptor.  So the number added is
     min( number of printout, num_workers )
"""


def train_model(device, model, dataloader, criterion, optimizer):
    """
    Trains the model on dataloader using device with optimizer, optimizer, and loss function (criterion).
    Nothing is returned, the model's weights a modified after calling this function.
    Note that after calling this function, the model will have the "best" weights
    found during training, not the last weights from the final epoch.
    The dataloaders is expected to be a dict with two keys "train" (required) and "val" (optional).
    :param device:
    :param model:
    :param dataloader:
    :param criterion:
    :param optimizer:
    :return:
    """

    # Set model to training mode
    model.train() # may implicitly call torch.set_grad_enabled(True)
    #print(f"TORCH: torch.is_grad_enabled is {torch.is_grad_enabled()}")

    running_loss = 0.0
    running_corrects = 0

    # Iterate over data.
    batch = 0
    # If Dataloader has num_workers=n, next line causes n processes to be created
    for inputs, labels, indexes in dataloader:
        inputs = inputs.to(device)
        labels = labels.to(device)

        # zero the parameter gradients each batch
        optimizer.zero_grad()

        # forward pass
        outputs = model(inputs)

        # compute loss on prediction distribution
        loss = criterion(outputs, labels)

        # backward pass
        if not torch.is_grad_enabled():
            raise ValueError(f"Gradient tracking must be enabled for backward pass: {torch.is_grad_enabled()}")
        loss.backward() # need gradient tracking to be enabled, no error raised
        optimizer.step()  # step optimiser each batch

        # running statistics
        _, preds = torch.max(outputs, 1)  # hard decision on distribution
        correct = torch.sum(preds == labels.data) # correct predictions in the batch

        running_loss += loss.item()
        running_corrects += correct
        if batch % 8 == 0:
            print(f'  batch {batch}:  running_corrects {running_corrects} loss: {running_loss:.6f}')

        #if batch != 0 and batch % 144 == 0:
        #    print("-------------------------------")
        #    monitor_processes.print_python_processes()
        #    print("-------------------------------")

        batch += 1

    # 1. We need to compute the average loss per batch, we know the running loss
    #    len(dataloader) is the number of batches and will equal batch
    # 2. Would like (but not necessary) to also know the accuracy, know total correct
    #    len(dataloader.dataset) is the number of instances
    avg_batch_loss = running_loss / len(dataloader) # could also use "batch" variable
    #epoch_acc = running_corrects.double() / len(dataloader)
    accuracy = float(running_corrects) / len(dataloader.dataset)

    # Minor type issue, accuracy is type=<class 'torch.Tensor'>, make it a float
    # avg_batch_loss is already a float


    return avg_batch_loss, accuracy


def main():
    # Get some cli arguments for preprocessing, training, evaluation.
    if len(sys.argv) != 4:
        print(f"Usage: <root directory of ISIC images> <number of epochs for training> <'balance' or 'imbalanced' | path to existing model>")
        print(f"Example: tone 20")
        return
    root_dir_name =     sys.argv[1]  #root_dir_name = "./tone"
    epochs        = int(sys.argv[2]) #epochs = 50
    model_folder  =     sys.argv[3]

    # =========================================================================#
    # Setup the folders and file where the model and results are going to be stored
    # =========================================================================#
    main_folder = "results"

    # Used to save the model, created first time, updated each time module is executed
    train_set_filename = "session_train.csv"
    test_set_filename = "session_test.csv"
    model_filename = "session_model.pth"

    timestamp = chronos.datetime.now()
    # print(f"timestamp = {timestamp}")
    # dt_object = chronos.datetime.fromtimestamp(timestamp)
    # If new model, filename used for both saving model and results, otherwise used to save results
    current_ts = timestamp.strftime("%Y-%m-%d_%H-%M-%S")

    balance_dataset = False
    if not os.path.exists( os.path.join(model_folder) ):
        # New model (and therefore folder) needs to be created
        if "imbalanced" in model_folder:
            balance_dataset = False
            model_folder_name = f"imbalanced_{current_ts}"
        else:
            balance_dataset = True
            model_folder_name = f"balanced_{current_ts}"
        model_folder = os.path.join(main_folder, model_folder_name)


    # Used to save results, will be used each time this module is run
    results_filename = f"{current_ts}.json"

    path_name = os.path.join(model_folder, results_filename)
    print(f"Creating results file: {path_name}")  # Output: 2022-02-07 14:30:00



    #=========================================================================#
    # 1. Read in the metadata and filter to only have attributes for rows with tone
    #=========================================================================#
    class_names = ['benign', 'malignant']
    
    # Check PyTorch has access to MPS (Metal Performance Shader, Apple's GPU architecture)
    print(f"Is MPS (Metal Performance Shader) built? {torch.backends.mps.is_built()}")
    print(f"Is MPS available? {torch.backends.mps.is_available()}")
    
    # Set the device (str that the model and data will use)
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Using device: {device}")

    # ======================================================================= #
    # 2. Define a Custom Model (Convolutional Neural Network)
    # Check to see if a model was previously trained, can continue training
    # ======================================================================= #
    train_set_path = os.path.join(model_folder, train_set_filename)
    test_set_path = os.path.join(model_folder, test_set_filename)
    model_path = os.path.join(model_folder, model_filename)
    if os.path.exists( model_folder ):
        # ===========================================================#
        # Load previous model and the train test sets
        # ===========================================================#
        print(f"Using previous model, will continue training.")

        model = model_module.load_model(model_path, class_names)
        # Overwrite in memory the train/test with previous
        train_df = pd.read_csv(train_set_path)
        test_df  = pd.read_csv(test_set_path)
    else:
        print(f"Using new model, training start new.")

        # Create the model_folder
        os.mkdir(model_folder)

        #===========================================================#
        # Need to create a new train/test set
        # ===========================================================#
        isic_metadata_df = dataset.read_isic_metadata(root_dir_name)

        if balance_dataset:
            isic_metadata_df = dataset.balance_dataset(isic_metadata_df)

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
        # Debug to make sure our dataframes match expected size
        print(f"train={len(train_df)} ({train_size})  test={len(test_df)} ({test_size})")

        # ===========================================================#
        # Need to create a new model
        # ===========================================================#
        #model = SkinCancerModel(class_names)  # create new model with initial weights
        model = SkinCancerListModel(class_names)
        #model = model_module.create_model(class_names)

        # Save the train and test datasets first time
        train_df.to_csv(train_set_path)
        test_df.to_csv(test_set_path)

    print(f"class_names={class_names} target size {len(class_names)}")
    # train_set, test_set = torch.utils.data.random_split(transformed_dataset, [train, test])

    print(f"Model type {type(model)}")
    print(f"{model}")
    # =====================================================================#

    #=========================================================================#
    # 3. Prepare the data using a custom PyTorch Dataset
    #    Returns (images, labels, indexes) instead of just (images, labels)
    #    When used with a DataLoader, returns (batch, images, labels, indexes)
    #    This (may) also apply the cropping, scaling, and any augmentation
    #    after this, it converts the numpy volume to a pytorch tensor
    # Based on https://pytorch.org/tutorials/beginner/basics/data_tutorial.html
    # Based on https://pytorch.org/tutorials/beginner/data_loading_tutorial.html
    #=========================================================================#
    train_dataset = HibaDataset( train_df,
                                class_names,
                                root_dir=root_dir_name,
                                transform=torchvision.transforms.Compose([
                                Rescale( (224,224) ),
                                #RandomCrop(224),
                                ToTensor()
                                ]))

    test_dataset = HibaDataset(test_df,
                                class_names,
                                root_dir=root_dir_name,
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
    # batch_size = 16
    # batch_size = 32 # 7 minutes per epoch with 10 threads, train=2536 images
    batch_size = 16  # ? minutes per epoch with 10 threads, train=2536 images
    # Note that perhaps a better name for Dataloader is Batcher as it is where the batches are produced.
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_threads)
    test_loader  = DataLoader(test_dataset,  batch_size=batch_size, shuffle=True, num_workers=num_threads)

    
    """
    Hit serious issue using ipython notebook and mac/mps
    Issue: General issue where multi-threading has issues with ipython on Mac python
    Ultimately solved by not running in the notebook, works fine outside
    Many of the sites suggested num_threads=0 and/or put "if-guard" and the code does run.
    However, this "solution" has huge performance implication, single-threaded!!!
    https://github.com/pytorch/pytorch/issues/60319
    https://www.reddit.com/r/MLQuestions/comments/n9iu83/pytorch_lightning_bert_attributeerror_cant_get/?rdt=51343
    """

    """
    New issue.  Ran with small batches=8 but ran terribly slow.  With 16 or 32 batch size runs very quick.
    But now after around 20 epochs we get below error
       RuntimeError: Too many open files. Communication with the workers is no longer possible.
       Please increase the limit using `ulimit -n` in the shell or change the sharing strategy by
       calling `torch.multiprocessing.set_sharing_strategy('file_system')` at the beginning of your code
    See https://github.com/pytorch/pytorch/issues/11201 for good discussion.
    Not sure setting this will help as for others also still ran into issues.
    Thought to "close" the file we open in the HibaDataset.__getitem__() function using "open with"

    So which file was not being closed?  We use skimage.io.imread to read file so was it not closing file?
       See https://github.com/scikit-image/scikit-image/issues/6939
    Actually, the file was being closed, issue is new threads are created each time, with a file handler
    and never cleaned up.  Nothing to do with our files, issue with multi-threading and/or DataLoader.
    """

    # =========================================================================#
    # 4. Training loop (with help of train_model)
    # =========================================================================#
    # Define optimisers, schedule, etc for training
    criterion = model_module.create_loss_function()
    #optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    # From optuna 'lr': 2.2066163621947597e-05
    optimizer = optim.Adam(model.parameters(), lr=0.00001)
    #scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
    
    # Need to put model and data onto device
    # To my knowledge this puts entire model into GPU/TPU
    model = model.to(device)

    # Keep track of time spent training and "best" model
    start_training_time = time.time()
    best_batch_loss = None
    best_accuracy   = None
    for epoch in range(epochs):
        start_epoch_time = time.time()
        print(f'Epoch {epoch+1}/{epochs}')
        print('-' * 10)

        avg_batch_loss, train_accuracy = train_model(device, model, train_loader, criterion, optimizer)

        # Update the scheduler each epoch
        #scheduler.step()

        # End of epoch stats
        print(f'Train Loss: {avg_batch_loss:.4f} Train Acc: {train_accuracy:.4f}')
        epoch_time = time.time() - start_epoch_time
        print(f"Epoch time: {epoch_time:.2f}s")
        print("\n")

        if best_batch_loss is None or avg_batch_loss < best_batch_loss:
            best_batch_loss = avg_batch_loss
            #print(f"Lowest average batch loss so far: {best_batch_loss:.4f}s")

        predictions = test_module.predict_with_instance(model, device, test_loader, test_dataset, class_names)
        test_results = test_module.analyse_predictions(predictions)

        with open(path_name, "a") as results_file:
            #print(f"TEST RESULTS: {test_results}")
            #with open(f'results_epoch{epoch}.json', 'w') as f:
            #    json.dump(test_results, f)
            print(f"avg_batch_loss={avg_batch_loss} type={type(avg_batch_loss)}")
            print(f"train_accuracy={train_accuracy} type={type(train_accuracy)}")
            test_results["avg_batch_loss"] = avg_batch_loss
            test_results["train_accuracy"] = train_accuracy
            test_results["epoch"] = epoch

            #json.dump(test_results, results_file)
            #results_str = json.dumps(test_results, indent=4, sort_keys=True)
            results_str = json.dumps(test_results)
            results_file.write( results_str )
            results_file.write("\n")


    # End of training stats
    training_time_elapsed = time.time() - start_training_time
    print(f'Training complete in {training_time_elapsed // 60:.0f}m {training_time_elapsed % 60:.0f}s')
    avg_min_per_epoch = (training_time_elapsed / 60.0) / epochs
    print(f"Average time per epoch (in mins): {avg_min_per_epoch:.2f}")
    #print(f'Best val Acc: {best_accuracy:4f}')
    #print(f'Best val Acc: {best_accuracy:4f}')


    #visualize_model(device, model, train_loader, num_images=5)

    # =========================================================================#
    # 5. Save last trained model (may consider saving "best" model)
    #    Nice to save model and for most part everything is same as though continued training
    #    Except!  The scheduler, presumably the learning rate is reset to larger value :(
    # =========================================================================#
    model_module.save_model(model, model_path) # overwrite any previous

if __name__ == '__main__':
    main()

