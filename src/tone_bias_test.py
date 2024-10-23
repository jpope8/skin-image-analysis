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
# Criitically, this dataset has "skin type" using the Fitzpatrick scale.
# 
# * https://en.wikipedia.org/wiki/Fitzpatrick_scale


import os
import sys
# If we have a version requirement, for no just make sure "more recent"
assert sys.version_info >= (3, 7)

import torch
import pandas as pd
import time

from torch.utils.data import DataLoader
import torchvision

# Only needed for imshow, not sure why we need both show_skin_image and imshow???
import numpy as np

# Best to just use functions in train stead of copying (downside, adds dependency)
import tone_bias_model as model_module

# Indirectly used when we torch.load the testdataset (which is HibaDataset with couple of transforms)
# Though we do not directly use, will get error when trying to load
from tone_bias_dataset import HibaDataset
from tone_bias_dataset import Rescale
from tone_bias_dataset import ToTensor

# Import matplotlib and set common/useful defaults
import matplotlib.pyplot as plt
plt.rc('font', size=14)
plt.rc('axes', labelsize=14, titlesize=14)
plt.rc('legend', fontsize=14)
plt.rc('xtick', labelsize=10)
plt.rc('ytick', labelsize=10)

import skimage

def show_skin_image(image_np):
    """Show image"""
    plt.imshow(image_np)
    
    #new_PIL_image = transform.to_pil_image(image_tensor) 
    #plt.imshow(new_PIL_image)
    
    #plt.scatter(landmarks[:, 0], landmarks[:, 1], s=10, marker='.', c='r')
    plt.pause(0.001)  # pause a bit so that plots are updated

# functions to show an image
def imshow(img):
    #img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


def visualize_model(device, model, dataloader, num_images=6):
    was_training = model.training
    model.eval()
    images_so_far = 0
    fig = plt.figure()

    class_names = model.get_class_names()

    with torch.no_grad():
        for i, data in enumerate(dataloader):
            inputs, labels, indexes = data
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            for j in range(inputs.size()[0]):
                images_so_far += 1
                ax = plt.subplot(num_images//2, 2, images_so_far)
                ax.axis('off')
                ax.set_title(f'predicted: {class_names[preds[j]]}  actual: {class_names[labels[j]]}')
                imshow(inputs.cpu().data[j])

                if images_so_far == num_images:
                    model.train(mode=was_training)
                    return
        model.train(mode=was_training)


def evaluate_model(device, model, testloader):
    """
    Evaluates the model on the test data.
    """
    # Remember to put into eval mode, otherwise dropout will result in random predictions
    model.eval()

    # Predict test images
    correct = 0
    total = 0
    # since we're not training, we don't need to calculate the gradients for our outputs
    with torch.no_grad():
        batch_number = 0
        for batch in testloader:
            images, labels, indexes = batch
            images = images.to(device)
            labels = labels.to(device)
            print(f"BATCH {batch_number}: indexes {indexes}")
            # calculate outputs by running images through the network
            outputs = model(images)
            # the class with the highest energy is what we choose as prediction
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            batch_number += 1

    print(f'Accuracy of the network on the {len(testloader)} batches')
    print(f'test images: {correct/total:4f} (correct {correct} / total {total})')


def evaluate_model_by_class(device, model, testloader, class_names):
    """
    Evaluates the model on the test data by class.
    """
    # prepare to count predictions for each class
    correct_pred = {classname: 0 for classname in class_names}
    total_pred = {classname: 0 for classname in class_names}
    
    # again no gradients needed
    with torch.no_grad():
        for batch in testloader:
            images, labels, indexes = batch
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predictions = torch.max(outputs, 1)
            # collect the correct predictions for each class
            for label, prediction in zip(labels, predictions):
                if label == prediction:
                    correct_pred[class_names[label]] += 1
                total_pred[class_names[label]] += 1
    
    
    # print accuracy for each class
    for classname, correct_count in correct_pred.items():
        pred_for_class = total_pred[classname]
        print( f"    {correct_count} / {pred_for_class}" )
        accuracy = 0.0
        if( pred_for_class > 0 ):
            accuracy = 100 * float(correct_count) / pred_for_class
        print(f'Accuracy for class: {classname:5s} is {accuracy:.1f} %')

def predict_with_instance(model, device, test_loader, test_dataset, class_names):
    # These could be use with the ai360 library, however, currently not using them
    # as the ai360 requires some amount of setup.  Should revisit, but custom
    # derivation of dispatate impact is sufficient for now.
    # NB: the list of y_pred indices need to match with the dataframe given to ai360
    y_true = list()
    y_pred = list()

    correct = 0
    total = 0
    # since we're not training, we don't need to calculate the gradients for our outputs
    k = 0

    # Make sure dropout is not active
    model.eval()

    # Collect the predictions along with the instance for post processing
    instances = dict()

    # NB: When the transformed_dataset is split, the rows (and associated indexes)
    #     are randomly shuffled and then split into test_set
    with torch.no_grad():
        batch_number = 0
        for batch in test_loader:
            # if( batch_index > 0 ):
            #    break
            # My hope is that the loader (we can shuffle because index is returned) and dataframe are "aligned"
            # Confirmed that this is indeed the case with the modified datase "with indices"
            images, labels, indexes = batch
            images = images.to(device)
            labels = labels.to(device)
            #print(f"BATCH {batch_number}: indexes {indexes}")
            #print(f"BATCH {batch_number} / {len(test_loader)}")

            # calculate outputs by running images through the network
            outputs = model(images)

            # the class with the highest energy is what we choose as prediction
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            #if (batch_number % 8 == 0):
            #    print(f"batch {batch_number * batch_size} {(batch_number + 1) * batch_size}")

            # print(f"=== Batch === {predicted}")
            for i in range(len(predicted)):
                output = outputs[i]
                pred = predicted[i]
                label = labels[i]  # actually index into class_names
                index = indexes[i]  # original index from isic_metadata_df, in transformed_dataset

                result = (pred == label)

                y_true.append(class_names[label])  # convert class index into class name
                y_pred.append(class_names[pred])  # convert class index into class name

                # Sanity check - need to make sure the prediction matches the instance in the dataset's dataframe
                # When we split the dataset become a "Subset" object and does not have our custom functions :(
                # Also, the split was ranndom so the indexes are also in random order between train and test
                # It was difficult (and perhaps foolish) to assume the order would be preserved
                #   though could shuffle dataframe before creating the dataset and not do a random split???
                # Regardless, decided to modify dataset (and therefore Subsets) to
                #   return the dataframe index along with the image and label
                # sample = test_set.get_file_path(index)
                # print(f"batch_index={batch_index} i={i} index={index}")
                # _, testset_label_dx, _ = test_set[rindex] # calls __getitem__
                instance = test_dataset.lookup_path(int(index))
                dataframe_label = instance['benign_malignant']
                # print(f"instance={instance}")

                # Add the prediction to the instance and add to collection of instances by index
                instance['prediction'] = class_names[pred]
                instances[int(index)] = instance

            batch_number += 1
    return instances


def confusion_matrix(instances):
    tp_instances = dict()
    tn_instances = dict()
    fp_instances = dict()
    fn_instances = dict()
    for index in sorted(instances.keys()):
        instance = instances[index]
        pred = instance['prediction']
        label = instance['benign_malignant']
        # sex  = instance['sex']
        # tone = instance['skin_tone']
        # age  = instance['age']

        # We consider 'malignant' to be the positive class
        if (pred == 'malignant' and label == 'malignant'):
            # TRUE POSITIVE
            # print( f"{index}: pred={pred} label={label} result={result} tone={tone}  age={age}  sex={sex}")
            # print( f"{index}: pred={pred} label={label} result={result}")
            tp_instances[index] = instance
        elif (pred == 'benign' and label == 'benign'):
            # TRUE NEGATIVE
            tn_instances[index] = instance
        elif (pred == 'malignant' and label == 'benign'):
            # FALSE POSITIVE
            fp_instances[index] = instance
        elif (pred == 'benign' and label == 'malignant'):
            # FALSE NEGATIVE
            fn_instances[index] = instance
    # Sanity check
    if (len(tp_instances) + len(tn_instances) + len(fp_instances) + len(fn_instances) != len(instances)):
        e = f"tp={len(tp_instances)} + tn={len(tn_instances)} + fp={len(fp_instances)} + fn={len(fn_instances)} != {len(instances)}"
        raise ValueError(e)
    return tp_instances, tn_instances, fp_instances, fn_instances

def values_counts(instances, feature, value):
    count = 0
    for index in instances.keys():
        instance = instances[index]
        if( instance[feature] == value ):
            count += 1
    return count


def filter(instances, feature, value):
    filtered_instances = dict()
    for index in instances.keys():
        instance = instances[index]
        if (instance[feature] == value):
            filtered_instances[index] = instance
    return filtered_instances


def disparate_impact_analysis(min_instances, maj_instances):
    # Determine confusion matrix given skin_tone
    tp_min, tn_min, fp_min, fn_min = confusion_matrix(min_instances)
    tp_maj, tn_maj, fp_maj, fn_maj = confusion_matrix(maj_instances)

    tp = len(tp_min) + len(tp_maj)
    tn = len(tn_min) + len(tn_maj)
    fp = len(fp_min) + len(fp_maj)
    fn = len(fn_min) + len(fn_maj)

    # All instances accuracy, precision, recall, f1
    accuracy_num = tp+tn
    accuracy_den= tp + tn + fp + fn
    accuracy = accuracy_num / accuracy_den
    precision = 0.0
    recall = 0.0
    f1 = 0.0
    if tp > 0:
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        f1 = 2 * ((precision * recall) / (precision + recall))

    min_count = len(min_instances)
    maj_count = len(maj_instances)

    # ============================================================#
    # Determine probability (aka "selection rate" = number selected in group / total number in group)
    # Also seen this as probability of "good outcome" from the model given group
    # ============================================================#
    # Selection is the number of positives predicted, rate is then divided by total
    # i.e. selection  rate = (TP + FP) / (TP + FP + TN + FN)
    # Selection could also be TP / (TP + FN), i.e. recall
    min_selected = len(tp_min) + len(fp_min)
    maj_selected = len(tp_maj) + len(fp_maj)

    selection_rate_min = min_selected / min_count
    selection_rate_maj = maj_selected / maj_count
    # ============================================================#

    # This is really a descriptive statistic, did not need model to determine this
    min_prevalence = (len(tp_min) + len(fn_min)) / min_count
    maj_prevalence = (len(tp_maj) + len(fn_maj)) / maj_count

    # Minority instances precision, recall, f1
    min_precision = 0.0
    min_recall = 0.0
    min_f1 = 0.0
    if len(tp_min) > 0:
        min_precision = len(tp_min) / (len(tp_min) + len(fp_min))
        min_recall = len(tp_min) / (len(tp_min) + len(fn_min))
        min_f1 = 2 * ((min_precision * min_recall) / (min_precision + min_recall))

    # Majority instances precision, recall, f1
    maj_precision = 0.0
    maj_recall = 0.0
    maj_f1 = 0.0
    if len(tp_maj) > 0:
        maj_precision = len(tp_maj) / (len(tp_maj) + len(fp_maj))
        maj_recall = len(tp_maj) / (len(tp_maj) + len(fn_maj))
        maj_f1 = 2 * ((maj_precision * maj_recall) / (maj_precision + maj_recall))


    # Positives in the populations
    p_min = len(tp_min) + len(fn_min)
    p_maj = len(tp_maj) + len(fn_maj)
    # print(f"P_min = {p_min}")
    # print(f"P_maj = {p_maj}")

    di = 0.0
    if selection_rate_maj > 0.0:
        di = selection_rate_min / selection_rate_maj


    verbose = False
    if verbose:
        print(f" accuracy: {accuracy}    ( {accuracy_num / accuracy_den:.3f} ) ")
        print(f"precision: {precision:.3f}")
        print(f"   recall: {recall:.3f}")
        print(f"       f1: {f1:.3f}")

        print(f"tp_min=({len(tp_min) / min_count:.3f})  {len(tp_min)}")
        print(f"tn_min=({len(tn_min) / min_count:.3f})  {len(tn_min)}")
        print(f"fp_min=({len(fp_min) / min_count:.3f})  {len(fp_min)}")
        print(f"fn_min=({len(fn_min) / min_count:.3f})  {len(fn_min)}")

        print(f"tp_maj=({len(tp_maj) / maj_count:.3f})  {len(tp_maj)}")
        print(f"tn_maj=({len(tn_maj) / maj_count:.3f})  {len(tn_maj)}")
        print(f"fp_maj=({len(fp_maj) / maj_count:.3f})  {len(fp_maj)}")
        print(f"fn_maj=({len(fn_maj) / maj_count:.3f})  {len(fn_maj)}")

        print(f"min prevalence {min_prevalence:.3f}")
        print(f"maj prevalence {maj_prevalence:.3f}")

        print(f"min_precision {min_precision:.3f}")
        print(f"min_recall    {min_recall:.3f}")
        print(f"    min_f1    {min_f1:.3f}")

        print(f"maj_precision {maj_precision:.3f}")
        print(f"maj_recall    {maj_recall:.3f}")
        print(f"    maj_f1    {maj_f1:.3f}")

        print(f"min group accuracy {(len(tp_min) + len(tn_min)) / min_count:.3f}")
        print(f"maj group accuracy {(len(tp_maj) + len(tn_maj)) / maj_count:.3f}")

        print(f"min selection_rate {min_selected} / {min_count}")
        print(f"maj selection_rate {maj_selected} / {maj_count}")

        print(f"selection_rate_min={selection_rate_min:.3f}")
        print(f"selection_rate_maj={selection_rate_maj:.3f}")
        print(f"DI(min/maj)={di:.3f}")

        if maj_precision > 0.0 and maj_recall > 0.0:
            print(f"DI_precision {min_precision / maj_precision:.3f}")
            print(f"DI_recall    {min_recall / maj_recall:.3f}")

        if selection_rate_min > 0.0:
            print(f"DI(maj/min)={selection_rate_maj / selection_rate_min:.3f}")

    # Prepare results to save later
    results = dict()
    results['accuracy'] = accuracy
    results['precision'] = precision
    results['recall'] = recall
    results['f1'] = f1

    results['selection_rate_min'] = selection_rate_min
    results['selection_rate_maj'] = selection_rate_maj
    results['di'] = di

    results['min_prevalence'] = min_prevalence
    results['maj_prevalence'] = maj_prevalence

    results['min_selected'] = min_selected
    results['min_count'] = min_count
    results['maj_selected'] = maj_selected
    results['maj_count'] = maj_count
    results['min_precision'] = min_precision
    results['min_recall'] = min_recall
    results['min_f1'] = min_f1
    results['maj_precision'] = maj_precision
    results['maj_recall'] = maj_recall
    results['maj_f1'] = maj_f1

    results['tp_min'] = len(tp_min)
    results['tn_min'] = len(tn_min)
    results['fp_min'] = len(fp_min)
    results['fn_min'] = len(fn_min)

    results['tp_maj'] = len(tp_maj)
    results['tn_maj'] = len(tn_maj)
    results['fp_maj'] = len(fp_maj)
    results['fn_maj'] = len(fn_maj)

    return results


def analyse_predictions(instances):
    # =========================================================================#
    # Determine how the model performed overall
    # =========================================================================#
    mycorrect = 0
    mytotal = 0
    for index in sorted(instances.keys()):
        instance = instances[index]
        pred = instance['prediction']
        label = instance['benign_malignant']
        sex = instance['sex']
        tone = instance['skin_tone']
        age = instance['age']
        result = (pred == label)
        if (result):
            mycorrect += 1
        mytotal += 1
        # print( f"{index}: pred={pred} label={label} result={result} tone={tone}  age={age}  sex={sex}")
    print(f"Total={mytotal} correct={mycorrect} my accuracy={mycorrect / mytotal:.3f}")

    # =========================================================================#
    # Now determine how well the model performed by skin tone
    # =========================================================================#
    dark_instances = filter(instances, 'skin_tone', 'dark')
    light_instances = filter(instances, 'skin_tone', 'light')
    print(f"dark {len(dark_instances)}")
    print(f"light {len(light_instances)}")
    male_instances = filter(instances, 'sex', 'male')
    female_instances = filter(instances, 'sex', 'female')
    print(f"male {len(male_instances)}")
    print(f"female {len(female_instances)}")

    print(f"total {len(instances)}")

    # =========================================================================#
    # Now determine the control
    # =========================================================================#
    rich_instances = filter(instances, 'control', 'rich')
    poor_instances = filter(instances, 'control', 'poor')
    print(f"rich {len(rich_instances)}")
    print(f"poor {len(poor_instances)}")

    # =========================================================================#
    # Determine all instances with True prediction of malignant (aka true positives)
    # =========================================================================#
    tp_instances, tn_instances, fp_instances, fn_instances = confusion_matrix(instances)


    # =========================================================================#
    # Analyse the performance of the model (usually recall) on different groups
    # =========================================================================#
    male_count = values_counts(tp_instances, 'sex', 'male')
    female_count = values_counts(tp_instances, 'sex', 'female')
    group_count = len(tp_instances)
    print(f"TP: male_count={male_count} female_count={female_count}")
    if (group_count > 0):
        print(f"TP: P(   male | mole=malignant ) = {male_count / group_count}")
        print(f"TP: P( female | mole=malignant ) = {female_count / group_count}")
    print(f"TP: male + female = {male_count + female_count}  total = {group_count}")

    male_count = values_counts(instances, 'sex', 'male')
    female_count = values_counts(instances, 'sex', 'female')
    group_count = len(instances)
    print()
    print(f"TEST_SET: male_count={male_count} female_count={female_count}")
    print(f"TEST_SET: P(   male ) = {male_count / group_count:.3f}")
    print(f"TEST_SET: P( female ) = {female_count / group_count:.3f}")
    print(f"TEST_SET: male + female = {male_count + female_count}  total = {group_count}")

    light_count = values_counts(instances, 'skin_tone', 'light')
    dark_count = values_counts(instances, 'skin_tone', 'dark')
    group_count = len(instances)
    print()
    print(f"TEST_SET: light_count={light_count} dark_count={dark_count}")
    if (group_count > 0):
        print(f"TEST_SET: P( light ) = {light_count / group_count:.3f}")
        print(f"TEST_SET: P(  dark ) = {dark_count / group_count:.3f}")
    print(f"TEST_SET: light + dark = {light_count + dark_count}  total = {group_count}")

    dark_positives = values_counts(dark_instances, 'benign_malignant', 'malignant')
    light_positives = values_counts(light_instances, 'benign_malignant', 'malignant')
    dark_prevalence = dark_positives / len(dark_instances)
    light_prevalence = light_positives / len(light_instances)
    print(f"Dark Prevalence: {dark_positives} / {len(dark_instances)} = {dark_prevalence:.2f}")
    print(f"Light Prevalence: {light_positives} / {len(light_instances)} = {light_prevalence:.2f}")

    # =========================================================================#
    # Payoff, disparate impact for skin tone and gender
    # =========================================================================#
    print(f"DISPARATE IMPACT: SKIN TONE")
    tone_di_results = disparate_impact_analysis(dark_instances, light_instances)
    print(f"DISPARATE IMPACT: GENDER")
    gender_di_results = disparate_impact_analysis(female_instances, male_instances)
    print(f"DISPARATE IMPACT: CONTROL")
    control_di_results = disparate_impact_analysis(poor_instances, rich_instances)
    #print(f"   num rich: {len(rich_instances)}")
    #print(f"   num poor: {len(poor_instances)}")

    results = dict()

    results["correct"] = mycorrect
    results["total"] = mytotal
    results["accuracy"] = mycorrect / mytotal

    results["dark"] = len(dark_instances)
    results["light"] = len(light_instances)
    results["male"] = len(male_instances)
    results["female"] = len(female_instances)

    results["tone_di_results"] = tone_di_results
    results["gender_di_results"] = gender_di_results
    results["control_di_results"] = control_di_results

    return results



def main():
    # Get some cli arguments for preprocessing, training, evaluation.
    if len(sys.argv) != 2:
        print(f"Usage: <root directory of ISIC images>")
        print(f"Example: tone")
        return
    root_dir_name = sys.argv[1]           #root_dir_name = "./tone"

    #=========================================================================#
    # 1. Read in the metadata and filter to only have attributes for rows with tone
    #=========================================================================#

    #trainset_filename = "session_train.csv"
    testset_filename = "session_test.csv"
    model_filename = "session_model.pth"
    class_names = ['benign', 'malignant']
    
    
    # Check PyTorch has access to MPS (Metal Performance Shader, Apple's GPU architecture)
    print(f"Is MPS (Metal Performance Shader) built? {torch.backends.mps.is_built()}")
    print(f"Is MPS available? {torch.backends.mps.is_available()}")
    
    # Set the device (str that the model and data will use)
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Define optimisation parameters for loading data
    # No GPU/TPU involved but multi-threading huge performance boost
    #   tune the num_threads to your CPU
    #   tune the batch size to your RAM
    num_threads = 10 # assuming 16 CPUs
    #batch_size = 16
    #batch_size = 32 # 7 minutes per epoch with 10 threads, train=2536 images
    batch_size = 16 # ? minutes per epoch with 10 threads, train=2536 images

    #=========================================================================#
    # 3. Define a Custom Model (Convolutional Neural Network)
    #=========================================================================#

    # Check to see if a model and testset file exist
    if not os.path.exists(model_filename) or not os.path.exists(testset_filename):
        print(f"Cannot find trained model {model_filename} or testset {testset_filename}")
        return

    test_df = pd.read_csv( testset_filename )
    test_dataset = HibaDataset(test_df,
                               class_names,
                               root_dir=root_dir_name,
                               transform=torchvision.transforms.Compose([
                                   Rescale((224, 224)),
                                   # RandomCrop(224),
                                   ToTensor()
                               ]))

    #class_names = test_set.get_class_names()
    # hopefully corresponds with the train, otherwise inverse predictions!
    # might save classes to file or somehow save with the testset
    class_names = ['benign', 'malignant']
    model = model_module.load_model(model_filename, class_names)

    print(f"Evaluation Dataset contains {len(test_dataset)} instances")
    print(f"test_set type = {type(test_dataset)}")

    # Create batches for the data, the num_workers=num_threads is a sore subject
    #trainloader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_threads)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=num_threads)

    #=========================================================================#
    # 4. Use model to make predictions on the testset
    #=========================================================================#
    model = model.to(device)
    start_time = time.time()
    #evaluate_model(device, model, test_loader)
    instances = predict_with_instance(model, device, test_loader, test_dataset, class_names)
    epoch_time = time.time() - start_time
    print(f"Evaluating time: {epoch_time:.2f}s")

    # =========================================================================#
    # 4. Evaluate the model's predictions
    # =========================================================================#
    analyse_predictions(instances)


if __name__ == '__main__':
    main()

