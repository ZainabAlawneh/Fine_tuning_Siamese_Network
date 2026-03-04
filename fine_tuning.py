from modules.signature_verification.utils import load_signet_model
from pathlib import Path
from torch.utils.data import DataLoader, Dataset
import torch 
from modules.signature_verification.preprocess import preprocess_signature
import cv2
import pandas as pd 
import torch.nn as nn
import torch.nn.functional as F
class ContrastiveLoss(nn.Module):
    def __init__(self, margin=1.0, alpha=1.0, beta=1.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin
        self.alpha = alpha
        self.beta = beta
    def forward(self, output1, output2, y):
        # Compute Euclidean distance
        distance = torch.nn.functional.pairwise_distance(output1, output2)
        # print(f"distance is {distance}")

        # Contrastive Loss
        # loss = 0.5 * (label) * (distance ** 2) + \
        #        0.5 * (1 - label) * torch.clamp(self.margin - distance, min=0) ** 2

        loss = self.alpha*(1-y)*torch.pow(distance,2) + self.beta*y *torch.pow(torch.clamp(self.margin - distance, min=0.0), 2)

        mLoss=loss.mean()
        # print(f"loss is {mLoss}")
        return loss.mean()

class Signnature_dataset(Dataset):
    def __init__(self, pairs_list):
        self.pairs_list = pairs_list

    def __len__(self):
        return len(self.pairs_list)

    def __getitem__(self, idx):
        path1, path2, label = self.pairs_list[idx]
        img1 = preprocess_signature(cv2.imread(path1, cv2.IMREAD_GRAYSCALE))
        img2 = preprocess_signature(cv2.imread(path2, cv2.IMREAD_GRAYSCALE))

        label = torch.tensor(label, dtype=torch.float32)
        return img1, img2, label


from sklearn.metrics import roc_curve
import numpy as np

def train(epochs=5):

    model_path = Path(__file__).parent / "assets" / "models"/ "signet.pth"


    model, _  = load_signet_model(model_path)

    print("Model is loaded successfully")

    for params in model.conv_layers.parameters():
        params.requires_grad = False


    for param in model.conv_layers[-2:].parameters():  # unfreeze last 2 conv blocks
        param.requires_grad = True


    optimizer = torch.optim.Adam(
        filter(lambda p:p.requires_grad, model.parameters()),
        lr=1e-5
    )

    model.train()

    data = pd.read_csv("dataset_.csv")
    batch_size = 16
    device = "cuda" if torch.cuda.is_available else "cpu"
    dataset = Signnature_dataset(data.values)

    dataloader = DataLoader(dataset, batch_size, shuffle=True)
    criterion = ContrastiveLoss(margin=1.0)

    thresold = 0.2
    print(f"started on device: {device}")

    for epoch in range(epochs) :
        running_loss = 0.0
        
        all_distances = []
        all_labels = []

        for batch_idx, (img1, img2, label )in enumerate(dataloader):
            img1 = img1.to(device)
            img2 = img2.to(device)
            label = label.to(device)
            optimizer.zero_grad()
            emb1 = model(img1)
            emb2 = model(img2)
            # print(f"emb2 : {emb1}")
            # print(f"emb2 : {emb2}")

            emb1 = F.normalize(emb1, p=2, dim=1)
            emb2 = F.normalize(emb2, p=2, dim=1)

            loss = criterion(emb1, emb2, label)

            loss.backward()
            optimizer.step()

            distance = torch.nn.functional.pairwise_distance(emb1, emb2)
            
            all_distances.append(distance.cpu())
            all_labels.append(label.cpu())


            print(f"Epoch [{epoch+1}/{epochs}], Batch [{batch_idx+1}/{len(dataloader)}]", end='\r')


        all_distances = torch.cat(all_distances).detach().cpu().numpy()
        all_labels = torch.cat(all_labels).detach().cpu().numpy()
        labels_flipped = 1 - all_labels 
        fpr, tpr, thresholds = roc_curve(labels_flipped, -all_distances)

        fnr = 1 - tpr
        eer_threshold = thresholds[np.nanargmin(np.absolute((fnr-fpr)))]
        eer = fpr[np.nanargmin(np.absolute((fnr - fpr)))]
        print(f"Epoch [{epoch+1}/{epochs}], EER: {eer*100:2f}%, threshold: {eer_threshold:.4f}")

    torch.save(model.state_dict(), "fine_tunining2.pth") 