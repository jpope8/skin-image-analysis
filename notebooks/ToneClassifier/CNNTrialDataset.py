from torch.utils.data.dataset import Dataset
from torchvision.transforms import v2
import torch
import pandas as pd
import torchvision
from pathlib import Path
import glob
import os
import random

def fitzpatrick_converter(entry):    #For labelling images as light or dark skintone based on the fitzpatrick skin tone score
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
    def __init__(self, image_path, state):    # This processes all the images before you use them to train your model
        self.image_path = Path(image_path) # One of the classes attributes is the image path
        # Load in either test or train data generated in TestTrainSplit.py
        if state == "Train":
            self.labels = pd.read_csv((self.image_path / 'trainmeta.csv'))
        else:
            self.labels = pd.read_csv((self.image_path / 'testmeta.csv'))
        self.state=state
        # Make a skin tone label that's either 0 if light skin or 1 else
        self.labels["SkinTone"]=self.labels["fitzpatrick_skin_type"].apply(lambda x: fitzpatrick_converter(x))
        # Make the isic id the same as the corresponding file name
        self.labels["isic_id"]=self.labels["isic_id"].apply(lambda x: x+".JPG")
        
        workable_images = []
        for file in os.listdir(self.image_path):
            if file[-4:]==".JPG":
                workable_images.append(file)
        
        self.labels = self.labels[self.labels["isic_id"].isin(workable_images)]
        
        self.datalength = len(self.labels)

        if state == "Train":
            # Different sorts of data augmentations you can do, have tried various models with different augmentations
            # Note different transformations for training versus testing
            
            self.transforms = v2.Compose([v2.RandomHorizontalFlip(p=0.5), # Randomly flips each image with 50% probability
                                        v2.RandomCrop((224, 224)), # All the images need to be the same size
                                        #v2.Resize((224, 224)), # You can either randomly crop an area or resize everything
                                        v2.ToDtype(torch.float32, scale=True),
                                        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])    # Normalising based on standard values
                                        #v2.FiveCrop((224, 224))   
                                        ])   
            # One option we considered (but found to perform poorly) was to use the FiveCrop augmentation
            # This crops top left, top right, bottom left, bottom right and centre and gives you 5 outputs
            # Assuming that the centre plot has the cancer mole, we randomly pick any of the other 4 plots
            # The hope being that this excludes the mole and picks a normal bit of skin
            # In practise this was unsucessful
        else:
            self.transforms = v2.Compose([v2.Resize((224, 224)), 
                                        v2.ToDtype(torch.float32, scale=True),
                                        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                                        #v2.FiveCrop((224, 224))
                                        ])
        
        if state == "Train":
            self.transforms_resize = v2.Compose([v2.RandomHorizontalFlip(p=0.5), 
                                        v2.Resize((224, 224)), 
                                        v2.ToDtype(torch.float32, scale=True),
                                        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])   
        else:
            self.transforms_resize = v2.Compose([v2.Resize((224, 224)), 
                                        v2.ToDtype(torch.float32, scale=True),
                                        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
        # Train images might randomly flip, test ones never do
        # Might want to look up mean and std RGB values for these but not essential
    

    def __getitem__(self, index):    # This retrieves data when you're actually training the model
        datarow = self.labels.iloc[index]
        filename, label = datarow[["isic_id", "SkinTone"]]
        
        #pixels = datarow[["pixels_x", "pixels_y"]]    # Some code I had used considered excluding certain images which were too small
        
        filename = str(self.image_path)+"/"+filename
        image = torchvision.io.read_image(filename)    
        
        image = self.transforms(image)

        # Code below is only useful when you're using the FiveCrop method, in which case you should comment out the above self.transform line
        # image_tuple = = (top_left, top_right, bottom_left, bottom_right, center)
        # if self.state == "Train":
        #     image = random.choice(image_tuple[:4])  # Randomly chooses one of the first four (non-central) crops if we're training
        #         #image = v2.Resize((224, 224))(image)
        # else:
        #     image = image_tuple[0]  # Always chooses the top left if we're testing
        #         #image = v2.Resize((224, 224))(image)
        
        return image, label
    
    def __len__(self):
        return len(self.labels)
    
        

