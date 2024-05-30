# Input ISIC images (which have associated Meta data)
# This data converts those ISIC images into matrices where each pixel is the ITA angle
# Idea is to feed the ITA matrices into a NN to classify skin tone

import cv2
from skimage import io, color
import numpy as np
import math
import os
import glob
from tqdm import tqdm
import pandas as pd
from scipy import stats
from concurrent.futures import ProcessPoolExecutor
import matplotlib.pyplot as plt

os.cpu_count()  # The big bit of processing at the end is set up so you can run it in parrallel

def Fitz_to_number(numeral):
    if numeral == 'I':
        return 1
    elif numeral == 'II':
        return 2
    elif numeral == 'III':
        return 3
    elif numeral == 'IV':
        return 4
    elif numeral == 'V':
        return 5
    elif numeral == 'VI':
        return 6
    else:
        return np.nan
      
# Pre-processing done below

# metadata = pd.read_csv('myimages/metadata.csv')   # change path to be where you've created the meta data
# metadata["Fitz"]=metadata["fitzpatrick_skin_type"].apply(Fitz_to_number)

# folder_path = 'myimages'
# for filename in os.listdir(folder_path):
#     if filename.endswith(".JPG"):
#         file=filename.replace(".JPG", "")
#         if file in metadata["isic_id"].values:
#             ISIC_ID.append(filename)
#             value = metadata[metadata["isic_id"]==file]["Fitz"].values[0]
#             Fitz_Label.append(value)
# df = pd.DataFrame({'ISIC_ID': ISIC_ID, 'Fitz_Label': Fitz_Label})
# df.to_csv("ISIC_ID.csv", index=False)

# This gives a DF which has the ISIC_ID and the Fitz number of each image
# This means you won't have a situation where you have an image where there's no metadata or vice versa
# Commented out because you only need to run it once

# All the functions below are vectorised to run more efficiently on matrices

def rgb_to_lab(r, g, b):
    """ Converts from r g b values to CieLab """
    lab = color.rgb2lab(r, g, b)
    return lab
  
def colour_mask(L, a, b):
  """ Returns NaN if any of L, a, b are NaN but also if L, a, b fall out of 
  accepted ranges for skin tones, make them NaN as well. Might need to tweak these ranges?
  """
    if np.isnan(L) or np.isnan(a) or np.isnan(b):
        return np.nan, np.nan, np.nan
    elif 30<L<80 == False:  
        return np.nan, np.nan, np.nan
    elif 0<a<20 == False:
        return np.nan, np.nan, np.nan
    elif 5<b<25 == False:
        return np.nan, np.nan, np.nan
    else:
        return L, a, b

colour_mask_ufunc = np.vectorize(colour_mask)

def lab_to_ITA(L, a, b):
  """ Converts CieLab to ITA angle"
    if L == np.nan or a == np.nan or b == np.nan:
        return np.nan
    elif b == 0:
        return np.nan
    else:
        return -np.arctan((L-50)/b)*180/math.pi  # Minus sign is correct, I promise!

lab_to_ITA_ufunc = np.vectorize(lab_to_ITA)

def ITA_to_Fitz(ITA_value):
""" Not used yet, but based on the literature, these are the ITA values corresponding with the different Fitz skintones """
    if ITA_value < -30:
        Fitz = 6
    elif -30 <= ITA_value < 10:
        Fitz = 5
    elif 10 <= ITA_value < 28:
        Fitz = 4
    elif 28 <= ITA_value < 41:
        Fitz = 3
    elif 41 <= ITA_value < 55:
        Fitz = 2
    elif 55 <= ITA_value:
        Fitz = 1
    else:
        Fitz = np.nan
    return Fitz

ITA_to_Fitz_ufunc = np.vectorize(ITA_to_Fitz)

def ImageProcesser(Image_File_Name):
""" The Big function that combines everything together
note it has no outputs except for a print statement, but it saves the images into a .npy file"""
    Name = Image_File_Name.replace(".JPG", "")
    TestImage = io.imread("myimages/{}".format(Image_File_Name))
    TestImage_np = np.array(TestImage)

    # Reshape the image into a 2D array
    TestImage_2D = TestImage_np.reshape(-1, TestImage_np.shape[-1])

    # Apply color.rgb2lab to the 2D array
    lab_matrix_2D = color.rgb2lab(TestImage_2D)

    # Reshape the result back into the shape of the original image
    lab_matrix = lab_matrix_2D.reshape(TestImage_np.shape)

    masked = colour_mask_ufunc(lab_matrix[..., 0], lab_matrix[..., 1], lab_matrix[..., 2])
    ITA_matrix = lab_to_ITA_ufunc(masked[0], masked[1], masked[2])
    ITA_matrix= np.array(ITA_matrix)
    np.save("ITA_matrices/{}.npy".format(Name), ITA_matrix)
    print("Saved as ITA_matrices/{}.npy".format(Name))

ISIC_df = pd.read_csv("ISIC_ID.csv", header=0)     # Meta data for the ISIC dataset with ISIC_ID
ISIC_ID = ISIC_df["ISIC_ID"].values
print(len(ISIC_ID))

ExistingITA = os.listdir("/home/hd15639/SkinTone/ITA_matrices")    # List of existing ITA matrices
ExistingITA = [file.replace(".npy", "") for file in ExistingITA]    # Make a folder called ITA_matrices in the same directory as this script
print(len(ExistingITA))

New_ISIC_ID = [ID for ID in ISIC_ID if ID not in ExistingITA]  # Handy if you had to pause your code
print(len(New_ISIC_ID))

if __name__ == "__main__":
    # create a pool of workers
    with ProcessPoolExecutor(max_workers=6) as pool:    # Be sure to specify your number of workers here
        result = list(tqdm(pool.map(ImageProcesser, New_ISIC_ID), total=len(New_ISIC_ID)))
