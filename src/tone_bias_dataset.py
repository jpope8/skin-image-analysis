#!/usr/bin/env python
# coding: utf-8

"""
# Prepare data and libraries
Based on the Hospital Italiano de Buenos Aires - Skin Lesions Images (2019-2022).

* https://www.isic-archive.com/
* https://api.isic-archive.com/collections/251/

Will need to install isic-cli and download the metadata (cvs file) and images.

isic image download --search 'fitzpatrick_skin_type:I OR fitzpatrick_skin_type:II OR fitzpatrick_skin_type:III OR fitzpatrick_skin_type:IV OR fitzpatrick_skin_type:V OR fitzpatrick_skin_type:VI' .

Critically, this dataset has "skin type" using the Fitzpatrick scale.

* https://en.wikipedia.org/wiki/Fitzpatrick_scale

This code reads in the ISIC metadata, filters for only instances with skin tone and benign/malignant.
The resulting dataframe is split into train and test dataframes which are used to create PyTorch Datasets.
The HibaDataset is a thin wrapper around the pandas dataframe and knows where to read image files.
This makes it easy to "save", in other words, save the dataframe as a CSV file for later.
Then easy to read CSV and create dataset object exactly as was saved.
This avoids using pickle to save (lots of issues), much cleaner with fewer dependencies.
"""

import os
import sys
import pandas as pd

import pathlib
import numpy as np
import skimage

from torch.utils.data import Dataset
import torch
import torchvision
from torchvision import utils

# Import matplotlib and set common/useful defaults
import matplotlib.pyplot as plt
plt.rc('font', size=14)
plt.rc('axes', labelsize=14, titlesize=14)
plt.rc('legend', fontsize=14)
plt.rc('xtick', labelsize=10)
plt.rc('ytick', labelsize=10)



def show_skin_image(image_np):
    """Show image"""
    plt.imshow(image_np)
    
    #new_PIL_image = transform.to_pil_image(image_tensor) 
    #plt.imshow(new_PIL_image)
    
    #plt.scatter(landmarks[:, 0], landmarks[:, 1], s=10, marker='.', c='r')
    plt.pause(0.001)  # pause a bit so that plots are updated



def filter_rows_without_values(df, col, values):
    """
    Returns "view" dataframe, subset of df with only rows where col is not in values.
    :param df:
    :param col:
    :param values:
    :return:
    """
    return df[~df[col].isin(values)]

# In[15]:
def filter_rows_with_values(df, col, values):
    """
    Returns "view" dataframe, subset of df with only rows where col is in values.
    :param df:
    :param col:
    :param values:
    :return:
    """
    return df[df[col].isin(values)]


def convert_type2tone(row):
    """
    Maps one of the skin tone types {I,II,III,IV,V,VI} to {light,dark}.
    :param row:
    :return:
    """
    # returns 0 meaning light tone, and 1 meaning dark tone
    #print(f"{row['fitzpatrick_skin_type']}")
    if row['fitzpatrick_skin_type'] == "I" or row['fitzpatrick_skin_type'] == "II":
        # Types I and II are "light" skin
        #row['tone'] = 0
        return 'light'
    else:
        #row['tone'] = 1
        return 'dark'


# Function to randomly take samples for each class
# Returns new dataframe with sample subset of df
def sample_dataframe(df, class_name, no_sample):
    #no_sample = 5
    balanced_df = df.groupby(
        class_name,
        as_index=False,
        group_keys=False
      ).apply( lambda s: s.sample(no_sample,replace=True)
    )
    return balanced_df


def read_isic_metadata(root_dir_name):
    """
    Reads the ISIC metadata file and eliminates images without skin tone or diagnosis.
    The root_dir_name specified should be a folder with images (*.jpg) and a metadata.csv file.
    Note that the returned dataframe has not been shuffled, likely caller needs to shuffle when returned.
    return a dataframe with the following attributes
        isic_id	attribution
        copyright_license
        acquisition_day
        age_approx
        anatom_site_general	    
        benign_malignant (aka diagnosis)
        clin_size_long_diam_mm
        concomitant_biopsy
        dermoscopic_type
        mel_type
        mel_ulcer
        melanocytic
        nevus_type
        patient_id
        personal_hx_mm
        pixels_x
        pixels_y
        sex
        skin_tone
    """

    # # File/Folder Testing
    # 
    # The data was downloaded from the ISIC archive using the cli tool (need to install this first).
    # 
    # See https://pypi.org/project/isic-cli/
    # 
    # See https://github.com/ImageMarkup/isic-cli
    # 
    # ```
    # mkdir tone
    # 
    # cd tone
    # 
    # isic image download --search 'fitzpatrick_skin_type:I OR fitzpatrick_skin_type:II OR fitzpatrick_skin_type:III OR fitzpatrick_skin_type:IV OR fitzpatrick_skin_type:V OR fitzpatrick_skin_type:VI' .
    # ```
    
    # This has all images from ISIC-HIBA (Argentina), skin_type is non-null (1616 images)
    #csv_file_name='./hiba/HIBA_dataset.csv'
    #meta_file_name='./hiba/SupplementaryData.csv'
    #root_dir_name='./hiba/images'

    verbose = False
    
    # This has all images from ISIC where the, skin_type is non-null (3685 images)
    # This includes the ISIC-HIBA (Argentina) images
    #csv_file_name='./tone/metadata.csv'
    #root_dir_name='./tone'
    csv_file_name= f"{root_dir_name}/metadata.csv"

    
    # Debug to make sure folder has some of expected images
    data_dir = pathlib.Path(root_dir_name) # NB: changing type froom str to Path
    file_paths = list(data_dir.glob(f'*.jpg'))
    image_count = len(file_paths)
    if verbose: print(f"image_count={image_count}")
    for i in range(image_count):
        file_path = file_paths[i]
        if verbose: print(f"file_path={file_path}")
        if i > 10:
            break
    
    # Read the csv file
    isic_metadata_df=pd.read_csv(csv_file_name)

    # Debug printouts
    #isic_metadata_df.info()
    #isic_metadata_df.head()
    
    
    # ### Drop rows that do not have skin type
    isic_metadata_df = isic_metadata_df[isic_metadata_df['fitzpatrick_skin_type'].notna()]
    
    
    # There are four categories
    # * benign
    # * malignant
    # * indeterminate/benign
    # * indeterminate/benign
    # 
    # Drop rows that are not benign or malignant.
    isic_metadata_df = filter_rows_with_values(isic_metadata_df, 'benign_malignant', ['benign', 'malignant'] )
    
    
    # Add new feature based on fitzpatrick_skin_type, this will become the new target,
    # converts from 6 multiclassification into a binary classfication
    isic_metadata_df["skin_tone"] = isic_metadata_df.apply( convert_type2tone, axis=1 )
    #isic_metadata_df.describe(include='all')
    
    #example_metadata = lookup_img_path( isic_metadata_df, "./hiba/images/ISIC_9999251.JPG" )
    #print(example_metadata)
    
    # First see how balanced our classes are
    if verbose: print( f"Total rows {len( isic_metadata_df )}" )
    if verbose: print( isic_metadata_df["skin_tone"].value_counts() / len( isic_metadata_df ) )
    if verbose: print( isic_metadata_df["benign_malignant"].value_counts() / len( isic_metadata_df ) )

    # Add random categorical feature
    isic_metadata_df['control'] = np.random.choice(["poor","rich"], isic_metadata_df.shape[0])


    return isic_metadata_df


def balance_dataset(isic_metadata_df):
    """
    Balance the datasets wrt diagnosis {malignant,benign} and then tone {light,dark}.
    Returned dataframe will have 50/50 {light,dark} and roughly 50/50 {malignant,benign}.
    :param isic_metadata_df:
    :return:
    """
    print("\nUNDERSAMPLING: BEFORE")
    # print(f"{isic_metadata_df}")
    print_counts(isic_metadata_df)

    # ======================================================================= #
    # Order matters?  Balance by tone then diagnosis or by diagnosis then tone?
    # Order does matter, we retain more instances with diagnosis then tone.
    # ======================================================================= #

    minority = isic_metadata_df[isic_metadata_df["benign_malignant"] == "malignant"]  # select minority class instances
    majority = isic_metadata_df[isic_metadata_df["benign_malignant"] == "benign"].sample(
        n=len(minority))  # sample majority class instances
    isic_metadata_df = pd.concat([minority, majority], axis=0)  # concatenate both samples

    minority = isic_metadata_df[isic_metadata_df["skin_tone"] == "dark"]  # select minority class instances
    majority = isic_metadata_df[isic_metadata_df["skin_tone"] == "light"].sample(
        n=len(minority))  # sample majority class instances
    isic_metadata_df = pd.concat([minority, majority], axis=0)  # concatenate both samples

    print("\nUNDERSAMPLING: AFTER")
    # print(f"{isic_metadata_df}")
    print_counts(isic_metadata_df)

    return isic_metadata_df



class HibaDataset(Dataset):
    """Hiba dataset, custom PyTorch dataset that knows how to read images using ISIC metadata.csv file."""

    def __init__(self, p_metadata_df, class_names, root_dir, transform=None):
        """
        Arguments:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the (JPG) images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.metadata_df = p_metadata_df
        #self.metadata_df = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

        # Drop  rows that do not have skin type
        #self.metadata_df = self.metadata_df[self.metadata_df['fitzpatrick_skin_type'].notna()]

        # Drop rows that do not have benign or malignant 
        #self.metadata_df = filter_rows_with_values(self.metadata_df, 'benign_malignant', ['benign', 'malignant'] )
        
        # Add new binary feature based on skin_type
        #self.metadata_df["skin_tone"] = self.metadata_df.apply( convert_type2tone, axis=1 )

        # Sample so that model target feature are balanced (bias feature remains unbalanced)
        #self.metadata_df = sample_dataframe(self.metadata_df, "benign_malignant", 500)
        
        
        self.image_count = len(self.metadata_df)
        #print(f"image_count = {image_count}")
        #for i in range(image_count):
        #    file_path = file_paths[i]

        # What target are we trying to predict?
        # Manually map target to ints   (required for models)
        self.class_names = class_names
        #self.class_names = ['I', 'II', 'III', 'IV' , 'V', 'VI']
        #self.class_names = ['light', 'dark']
        
    
    def __len__(self):
        return self.image_count

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        instance = self.lookup_path( idx )
        file_path = self.get_file_path( instance["image_name"] )

        #img_name = sample['imgid']
        #image = io.imread(img_name)
        # Frankly not sure why we don't just store as Pytorch tensor
        # example stores as a numpy image and then transforms
        # In any cases, the dimensions are different and so code is not compatible
        # specifically the resize and crop are written for numpy :(
        # Perhaps one day can rewrite for tensors but fine for now
        #image_tensor = torchvision.io.read_image( str(file_path) )
        #sample['image'] = image_tensor

        #======#
        # Does not work but what I want to do
        #file = open(file_path, 'r')
        #image_np = skimage.io.imread(file) # dtype=uint8
        #file.close()

        # This opens file and closes, though multi-processing may fail to clean up file descriptor.
        image_np = skimage.io.imread(file_path) # dtype=uint8
        #======#
        
        # [[[209 155 181]
        #  [210 154 183]
        #  [209 154 183]
        #  ...
        # need float32, also imshow, if float, expects  between [0,1]
        # This is also perhaps nice for training as normalised.
        image_np = np.float32(image_np)/255.0

        # Change  the target from {benign,malignant} to the skin_type {I,II,III,IV,V,VI}
        label = self.class_names.index( instance['benign_malignant'] )
        #label = self.class_names.index( row['skin_type'] )
        #label = self.class_names.index( row['skin_tone'] )
        # NB: the label is index of the target name
        # NB: tuple has to be tensors or numbers, cannot include the instance :(
        sample = (image_np, label, idx)

        # apply any transforms, e.g. resize, crop
        if self.transform:
            sample = self.transform(sample)

        return sample

    def get_class_names( self ):
        return self.class_names

    def get_class( self, index ):
        return self.class_names[index]

    def get_file_path( self, image_name ):
        #sample = self.lookup_path(idx)
        #image_name = sample["image_name"]
        # Need to put together filepath
        filepath = os.path.join( self.root_dir, image_name+".jpg" )
        return filepath
    
    def lookup_path( self, idx ):
        """
        index of the instance (e.g. ISIC_0034214) in the dataframe
        returns dict {'file_path','image_name','patient_id','diagnosis',
            'benign_malignant', 'age', 'sex', 'location', 'skin_type'}
        """
        
        row = self.metadata_df.iloc[idx]

        #print( f"row type = {type(row)}" )
        #print( row )
        # the self.metadata_df.iloc[idx] returns a pandas Series
        # basically a dictionary for one row of datframe
        patient_id = row["patient_id"]
        image_name = row["isic_id"]
        diagnosis  = row["diagnosis"]
        benign     = row["benign_malignant"]
        age        = row["age_approx"]
        sex        = row["sex"]
        location   = row["anatom_site_general"]
        skin_type  = row["fitzpatrick_skin_type"]
        skin_tone  = row["skin_tone"]
        control    = row["control"]
        file_path  = self.get_file_path(image_name)
    
        instance = {'file_path': file_path, 'image_name': image_name, 'patient_id': patient_id,
                  'diagnosis': diagnosis, 'benign_malignant': benign, 'age': age, 'sex': sex,
                  'location': location, 'skin_type':skin_type, 'skin_tone':skin_tone,
                  'control': control}
        return instance
    


class Rescale(object):
    """Rescale the image in a sample to a given size.

    Args:
        output_size (tuple or int): Desired output size. If tuple, output is
            matched to output_size. If int, smaller of image edges is matched
            to output_size keeping aspect ratio the same.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        # output_size to rescale the image to
        self.output_size = output_size

    def __call__(self, sample):
        image, label, index = sample

        h, w = image.shape[:2]
        if isinstance(self.output_size, int):
            if h > w:
                new_h, new_w = self.output_size * h / w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size * w / h
        else:
            new_h, new_w = self.output_size

        new_h, new_w = int(new_h), int(new_w)

        img = skimage.transform.resize(image, (new_h, new_w))

        return (img, label, index)


class RandomCrop(object):
    """
    Crop randomly the image in a sample.
    Args:
        output_size (tuple or int): Desired output size. If int, square crop
            is made.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, sample):
        image, label, index = sample

        h, w = image.shape[:2]
        new_h, new_w = self.output_size

        top = np.random.randint(0, h - new_h + 1)
        left = np.random.randint(0, w - new_w + 1)

        image = image[top: top + new_h,
                      left: left + new_w]

        return (image, label, index)


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image, label, index = sample

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C x H x W
        image = image.transpose((2, 0, 1))
        #print(image_np.dtype) = uint8
        # default numpy type is float64, causes issues as models expect float32 :(
        return (torch.from_numpy(image), label, index)

# Create function to show the images in a batch from a dataloader
# Helper function to show a batch
def show_landmarks_batch(sample_batched):
    """Show image with landmarks for a batch of samples."""
    images_batch, labels_batch, indexes = sample_batched
    batch_size = len(images_batch)
    #print(f"batch_size={batch_size}")
    im_size = images_batch.size(2)
    grid_border_size = 2

    grid = utils.make_grid(images_batch)
    plt.imshow(grid.numpy().transpose([1, 2, 0]))


def print_counts(df):
    count_malignant = df['benign_malignant'].value_counts()['malignant']
    count_benign = df['benign_malignant'].value_counts()['benign']
    print(f"Diagnosis: Count malignant: {count_malignant}")
    print(f"Diagnosis: Count    benign: {count_benign}")
    print(f"Diagnosis: {count_benign/len(df):.3f} benign, {count_malignant/len(df):.3f} malignant")
    count_light = df['skin_tone'].value_counts()['light']
    count_dark  = df['skin_tone'].value_counts()['dark']
    print(f"Skin Tone: Count     light: {count_light}")
    print(f"Skin Tone: Count      dark: {count_dark}")
    print(f"Skin Tone: {count_light/len(df):.3f} light, {count_dark/len(df):.3f} dark")

def main():

    # Get some cli arguments for preprocessing, training, evaluation.
    if len(sys.argv) != 3:
        print(f"Usage: <root directory of ISIC images> <balance|imbalanced>")
        print(f"Example: tone balance")
        return
    root_dir_name = sys.argv[1]           #root_dir_name = "./tone"
    balance = sys.argv[2]

    #=========================================================================#
    # 1. Read in the metadata and filter to only have attributes for rows with tone
    #=========================================================================#
    isic_metadata_df = read_isic_metadata( root_dir_name )
    class_names = ['benign', 'malignant']

    if balance == "balance":
        isic_metadata_df = balance_dataset(isic_metadata_df)
        isic_metadata_df.to_csv("balanced_metadata.csv")



def main_dataset():

    # Get some cli arguments for preprocessing, training, evaluation.
    if len(sys.argv) != 2:
        print(f"Usage: <root directory of ISIC images>")
        print(f"Example: tone")
        return
    root_dir_name = sys.argv[1]           #root_dir_name = "./tone"

    #=========================================================================#
    # 1. Read in the metadata and filter to only have attributes for rows with tone
    #=========================================================================#
    isic_metadata_df = read_isic_metadata( root_dir_name )
    class_names = ['benign', 'malignant']

    # Shuffle the instances first
    # ### Shuffle the dataframe and reset index (has original, do not believe we need to keep?)
    isic_metadata_df = isic_metadata_df.sample(frac=1)  # effectively shuffle
    isic_metadata_df = isic_metadata_df.reset_index(drop=True)

    # Now split into train and test dataframes (will become train and test sets)
    total = isic_metadata_df.shape[0]
    ratio = 0.7
    train_size = int(ratio * total)
    test_size  = total - train_size

    # Nice because we can just use slicing to get train/test after shuffling
    train_df = isic_metadata_df[:train_size]
    train_df = train_df.reset_index(drop=True)
    test_df  = isic_metadata_df[train_size:]
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


    #=========================================================================#
    # 2. Prepare the data using a custom PyTorch Dataset
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
                                    ])
                                 )

    test_dataset = HibaDataset( test_df,
                                class_names,
                                root_dir=root_dir_name,
                                transform=torchvision.transforms.Compose([
                                    Rescale((224, 224)),
                                    # RandomCrop(224),
                                    ToTensor()
                                    ])
                                )

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
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=batch_size,
                                               shuffle=True,
                                               num_workers=num_threads)
    test_loader  = torch.utils.data.DataLoader(test_dataset,
                                               batch_size=batch_size,
                                               shuffle=True,
                                               num_workers=num_threads)

    batch_number = 0
    for batch in train_loader:
        images, labels, indexes = batch
        images = images.to(device)
        labels = labels.to(device)

        #output = model(images)
        print(f"Batch {batch_number} / {len(train_loader)}")
        batch_number += 1
    # Seems to take a long time wrapping up???
    print("Done")


if __name__ == '__main__':
    main()

