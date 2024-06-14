from torch.utils.data.dataset import Dataset
from torchvision.transforms import v2
import torch
import pandas as pd
import torchvision
from pathlib import Path
import glob
import os

def fitzpatrick_converter(entry):
    if entry == "I":
        return 0
    elif entry == "II":
        return 0
    elif entry == "III":
        return 1
    elif entry == "IV":
        return 1
    elif entry == "V":
        return 1
    elif entry == "VI":
        return 1
    else:
        return "Error"

class ISIC(Dataset):
    def __init__(self, image_path, state):
        self.image_path = Path(image_path) # One of the classes attributes is the image path
        if state == "Train":
            self.labels = pd.read_csv((self.image_path / 'trainmeta.csv'))
        else:
            self.labels = pd.read_csv((self.image_path / 'testmeta.csv'))
        
        self.labels["SkinTone"]=self.labels["fitzpatrick_skin_type"].apply(lambda x: fitzpatrick_converter(x))
        self.labels["isic_id"]=self.labels["isic_id"].apply(lambda x: x+".JPG")
        
        workable_images = []
        for file in os.listdir(self.image_path):
            if file[-4:]==".JPG":
                workable_images.append(file)


        #workable_images = glob.glob(self.image_path/'*.JPG')
        
        self.labels = self.labels[self.labels["isic_id"].isin(workable_images)]

        self.datalength = len(self.labels)

        self.transforms = v2.Compose([v2.Resize((224, 224)), 
                                      v2.RandomHorizontalFlip(p=0.5), 
                                      v2.ToDtype(torch.float32, scale=True),
                                      v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])   
        # Might want to look up mean and std RGB values for these but not essential
        # This resizes all the images so they're the same + also makes them smaller
        # Might want to play around with some augmentations for robustness in the future!

    def __getitem__(self, index):
        datarow = self.labels.iloc[index]
        filename, label = datarow[["isic_id", "SkinTone"]]
        filename = str(self.image_path)+"/"+filename
        image = torchvision.io.read_image(filename)
        image = self.transforms(image)

        return image, label
    
    def __len__(self):
        return len(self.labels)
    
        

