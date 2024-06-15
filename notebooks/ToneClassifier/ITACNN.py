import torch
import torch.nn as nn
from torchvision.models import resnet18
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
import wandb
import random
from ITATrialData import ISIC

print(torch.cuda.is_available())

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def main():
    # start a new wandb run to track this script
    wandb.init(
        # set the wandb project where this run will be logged
        project="SkinToneSeedcorn",

        # track hyperparameters and run metadata
        config={
        "architecture": "Resnet 18",
        "dataset": "ISIC",
        "epochs": 50,
        }
    )
    train_dataset = ISIC("/home/hd15639/SkinTone/ITA_matrices", "Train")
    train_dataloader = DataLoader(train_dataset, batch_size=8, shuffle=True)

    test_dataset = ISIC("/home/hd15639/SkinTone/myimages", "Test")
    test_dataloader = DataLoader(test_dataset, batch_size=8, shuffle=False)

    model = resnet18(weights=None)
    # Do I want it pre-trained?
    # Bigger model maybe?
    # Resnet probably will do the job
    # Could think more about data augmentations
    # Learning rate is something to play with

    if torch.cuda.device_count() > 0:
        model = nn.DataParallel(model)
    
    criterion = nn.CrossEntropyLoss()   #Calculates our loss
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)   #SGD=stochastic gradient descent, updates parameters of the model

    model.train(True)
    for epoch in range(1,50):
        Predictions = []
        Targets = []
        for i, (data, target) in enumerate(train_dataloader):
            data = data.to(device)
            target = target.to(device)
            Targets.append(target.cpu())
            output = model(data)
            Predictions.append(torch.argmax(output).cpu())
            
            loss = criterion(output, target)
            wandb.log({"Train loss": loss})
            print(f"Train loss {loss}")
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        # print(Predictions)
        # print(Targets)
        TrainAcc=sum(1 for x,y in zip(Predictions, Targets) if x==y)/len(Targets)
        wandb.log({"Train Accuracy": TrainAcc}, step=epoch)
        if epoch%5 == 0:
            Predictions = []
            Targets = []
            for i, (data, target) in enumerate(test_dataloader):
                data = data.to(device)
                target = target.to(device)
            
                output = model(data)
                Predictions.append(torch.argmax(output).cpu())
                Targets.append(target.cpu())

                loss = criterion(output, target)
                wandb.log({"Test loss": loss})
                print(f"Test Loss {loss}")
                   
            TestAcc=sum(1 for x,y in zip(Predictions, Targets) if x==y)/len(Targets)
            wandb.log({"Test Accuracy": TestAcc}, step=epoch)

        #print(f"Train Accuracy: {TrainAcc}")
        #print(f"Test Accuracy {TestAcc}")
    wandb.finish()
            


            



    

if __name__ == '__main__':
    main()

