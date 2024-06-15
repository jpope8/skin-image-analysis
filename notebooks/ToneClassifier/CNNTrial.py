import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights, resnet50, ResNet50_Weights
from torch.utils.data import DataLoader, WeightedRandomSampler
from sklearn.model_selection import train_test_split
import wandb
import random
from CNNTrialDataset import ISIC
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
import pandas as pd
from PIL import Image
import torchvision.transforms as T
import matplotlib.pyplot as plt

from sklearn.utils.class_weight import compute_class_weight
import numpy as np

print(torch.cuda.is_available())    # All of this code is designed to run on something bigger than just a laptop!

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

epochs = 1
learning_rate = 0.001
BatchSize = 1
Dropout_Rate = 0.5

def main():
    # start a new wandb run to track this script
    # Highly recommend using this with Weights and Biases: https://wandb.ai/disco_huw/SkinToneSeedcorn?nw=nwuserhuwday
    
    wandb.init(
        # set the wandb project where this run will be logged
        project="SkinToneSeedcorn",
        name=f"Image Printer, ResNet50, Imagenet1K_V1, AdamW, OverSampler, random crop (224), class weight, Learning Rate = {learning_rate}, Dropout = {Dropout_Rate}, Epochs = {epochs}, BatchSize = {BatchSize}",
        notes=f"PreTrained ResNet50 using ImageNet1K_V1 Model Weights, AdamW optimisier with lr=0.001 (default)",
        # track hyperparameters and run metadata
        config={
        "architecture": "Resnet 50",
        "dataset": "ISIC",
        "epochs": f"{epochs}",
        }
    )
    trainmeta = pd.read_csv('/home/hd15639/SkinTone/myimages/trainmeta.csv')    # Need to change the path here for the metadata
    
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
    trainmeta["SkinTone"]=trainmeta["fitzpatrick_skin_type"].apply(lambda x: fitzpatrick_converter(x))
    target=list(trainmeta["SkinTone"])

    # This code works out the class imbalance:

    class_sample_count = np.array([len(np.where(target == t)[0]) for t in np.unique(target)])
    weight = 1. / class_sample_count
    samples_weight = np.array([weight[t] for t in target])    

    samples_weight = torch.from_numpy(samples_weight)
    samples_weight = samples_weight.double()
    train_sampler = WeightedRandomSampler(samples_weight, len(samples_weight))    # This is a weighted random sampler
    
    train_dataset = ISIC("/home/hd15639/SkinTone/myimages", "Train")
    train_dataloader = DataLoader(train_dataset, batch_size=BatchSize, shuffle=True, sampler=None, num_workers=6)  
    # sampler can be set to be train_sampler but it incompatible with shuffle=True. num_workers is the number of cores to use!
    
    test_dataset = ISIC("/home/hd15639/SkinTone/myimages", "Test")
    test_dataloader = DataLoader(test_dataset, batch_size=BatchSize, shuffle=False, num_workers=6)  # num_workers is the number of cores to use! JGI server has 12
    
    model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)    # Most of the work here has been done on out the box resnet 50
    num_ftrs = model.fc.in_features

    #model.fc = nn.Linear(num_ftrs, 2)

    model.fc = nn.Sequential(    # Tried using dropout rates here as well
    nn.Dropout(Dropout_Rate),
    nn.Linear(num_ftrs, 2))

    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    model = model.to(device)


    # Weight comes from ClassWeight.py File
    class_weights=torch.tensor([2.96221865, 0.60153444], dtype=torch.float).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights) #Calculates our loss

    #optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)   #SGD=stochastic gradient descent, updates parameters of the model
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    model.train(True)

    print(f"Is the model on cuda? {next(model.parameters()).is_cuda}")
    test_step = 0
    train_step = 0
    for epoch in range(0,epochs):
        print(f"Epoch = {epoch+1}")
        wandb.log({"Epoch": epoch+1})
        Predictions = []
        Targets = []
        vis_count=0
        for i, (data, target) in enumerate(train_dataloader):
            data = data.to(device)
            #print(f"Data device = {data.get_device()}")
            target = target.to(device)
            
            output = model(data)
            loss = criterion(output, target)    # This is where the error is 
            if vis_count < 10:
                # Denormalize the image if necessary
                data = (data + 1) / 2  # This line is needed if your images are normalized to [-1, 1]
                data = data.clamp(0, 1)  # Clamp values to [0, 1]

                transform = T.ToPILImage()
                img = transform(data[0])
                img.save(f'outputimages/output_image_test_{vis_count}.png')
                

                
                vis_count+=1
            else:
                print("Victory")
                break
            
            #print(f"Train loss = {loss}")
            wandb.log({"Train loss": loss, "Train step": train_step})
            train_step+=1

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            Targets.extend(target.detach().cpu())
            output = torch.argmax(output, dim=1)
            output = output.detach().cpu().tolist()
            Predictions.extend(output) 
            
            
            
        # print(Predictions)
        # print(Targets)
        TrainAcc = accuracy_score(Targets, Predictions)
        TrainRec = recall_score(Targets, Predictions)
        TrainPrec = precision_score(Targets, Predictions)
        TrainF1 = f1_score(Targets, Predictions)

        wandb.log({"Train Accuracy": TrainAcc, "Train Precision": TrainPrec, "Train Recall": TrainRec, "Train F1": TrainF1})
        if epoch%1 == 0:    #Set to 1 currently, maybe do it for 5 later
            Predictions = []
            Targets = []
            vis_count=0
            for i, (data, target) in enumerate(test_dataloader):
                data = data.to(device)
                target = target.to(device)
            
                output = model(data).to(device)
                loss = criterion(output, target)
                if vis_count < 10:
                    # Denormalize the image if necessary
                    data = (data + 1) / 2  # This line is needed if your images are normalized to [-1, 1]
                    data = data.clamp(0, 1)  # Clamp values to [0, 1]

                    transform = T.ToPILImage()
                    img = transform(data[0])
                    img.save(f'outputimages/output_image_train_{vis_count}.png')
                    

                    
                    vis_count+=1
                else:
                    print("Victory")
                    break
                output = torch.argmax(output, dim=1)
                output = output.detach().cpu().tolist()
                Predictions.extend(output) 
                
                Targets.extend(target.cpu())

                
                wandb.log({"Test loss": loss, "Test Step": test_step})
                test_step+=1
                #print(f"Test loss = {loss}")
            
            if len(Targets)!=0:      
                TestAcc = accuracy_score(Targets, Predictions)
                TestRec = recall_score(Targets, Predictions)
                TestPrec = precision_score(Targets, Predictions)
                TestF1 = f1_score(Targets, Predictions)

                wandb.log({"Test Accuracy": TestAcc, "Test Precision": TestPrec, "Test Recall": TestRec, "Test F1": TestF1})
                
        
        #print(f"Train Accuracy: {TrainAcc}")
        #print(f"Test Accuracy {TestAcc}")
    wandb.finish()  

if __name__ == '__main__':
    main()
