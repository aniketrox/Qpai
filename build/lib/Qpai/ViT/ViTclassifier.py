from transformer_package.models import ViT
import torch
import numpy as np
import torchvision.transforms as transforms
from torchvision import datasets
import torch.nn as nn


class ViTclassifier_model:

    torch.manual_seed(0)
    np.random.seed(0)

    def __init__(self):
        self.transform = None
        self.batch_size = 0
        self.lr = 0
        self.epochs = 0
        self.model = None
        self.device = None
        self.criterion = None
        self.optimizer = None
        self.trainloader = None


    def select_device(self,device_name):
        if device_name == "cuda":
            if torch.cuda.is_available():
                self.device = torch.device(device_name)
                return self.device
            else:
                print("Sorry hommie!! You have no cuda in your device. You need to install it at first")
                return
        elif device_name == "cpu":
            self.device = torch.device(device_name)
            return self.device
        
        else:
            print("Bro!! specify correctly")
            return
        
    def primary_setup(self,image_size,batch_size):
        self.batch_size = batch_size
        mean, std = (0.5,), (0.5,)
        self.transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize(mean, std),transforms.RandomResizedCrop(image_size),])

    def render_train_data(self,folder_path):
        train_data = datasets.ImageFolder(folder_path,transform=self.transform)
        self.trainloader = torch.utils.data.DataLoader(train_data, batch_size=self.batch_size, shuffle=True)
        print(self.trainloader.dataset)
        return self.trainloader

    def render_test_data(self,folder_path):
        test_data = datasets.ImageFolder(folder_path,transform=self.transform)
        testloader = torch.utils.data.DataLoader(test_data, batch_size=self.batch_size, shuffle=True)
        print(testloader.dataset)
        return testloader
    
    def render_valid_data(self,folder_path):
        valid_data = datasets.ImageFolder(folder_path,transform=self.transform)
        validloader = torch.utils.data.DataLoader(valid_data, batch_size=self.batch_size, shuffle=True)
        print(validloader.dataset)
        return validloader
    
    def model_call(self,image_size, channel_size, patch_size, embed_size, num_heads, classes, num_layers, hidden_size, dropout):
        self.model = ViT(image_size, channel_size, patch_size, embed_size, num_heads, classes, num_layers, hidden_size, dropout=dropout).to(self.device)
        return self.model
    
    def train(self,epochs,lr):
        self.epochs = epochs
        self.lr = lr
        self.criterion = nn.NLLLoss()
        self.optimizer = torch.optim.Adam(params=self.model.parameters(), lr=self.lr)

        

        for epoch in range(1, self.epochs+1):
            self.model.train()
            
            epoch_train_loss = 0
                
            y_true_train = []
            y_pred_train = []

            loss_hist = {}
            loss_hist["train accuracy"] = []
            loss_hist["train loss"] = []
                
            for batch_idx, (img, labels) in enumerate(self.trainloader):
                img = img.to(self.device)
                labels = labels.to(self.device)
                
                preds = self.model(img)
                
                loss = self.criterion(preds, labels)
                
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
                
                y_pred_train.extend(preds.detach().argmax(dim=-1).tolist())
                y_true_train.extend(labels.detach().tolist())
                
                epoch_train_loss += loss.item()
            
            loss_hist["train_loss"].append(epoch_train_loss)
            
            total_correct = len([True for x, y in zip(y_pred_train, y_true_train) if x==y])
            total = len(y_pred_train)
            accuracy = total_correct * 100 / total
            
            loss_hist["train_accuracy"].append(accuracy)
            
            print("-------------------------------------------------")
            print("Epoch: {} Train mean loss: {:.8f}".format(epoch, epoch_train_loss))
            print("       Train Accuracy%: ", accuracy, "==", total_correct, "/", total)
            print("-------------------------------------------------")

        
    