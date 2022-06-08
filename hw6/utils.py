import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision
import os
import torchvision.transforms as transforms
from torch.utils.data import Subset

###
# plot a scatter plot of coordinates with labels labels
# the data contain k classes
###
def scatter_plot(coordinates,labels,k):
    fig, ax = plt.subplots()
    for i in range(k):
        idx = labels == i
        data = coordinates[:,idx]
        ax.scatter(data[0],data[1],label=str(i),alpha=0.3,s=10)
    ax.legend(markerscale=2)
    plt.show()
    

### FOR THE AUTOENCODER PART
def train(num_epochs,dataloader,model,criterion,optimizer):
    for epoch in range(num_epochs):
        for data in dataloader:
            img, _ = data
            # ===================forward=====================
            output,_ = model(img)
            loss = criterion(output, img.squeeze())
            # ===================backward====================
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        # ===================log========================
        print('epoch [{}/{}], loss:{:.4f}'
              .format(epoch + 1, num_epochs, loss.item()))

def get_embedding(model,dataloader):
    model.eval()
    labels = np.zeros((0,))
    embeddings = np.zeros((2,0))
    for data in dataloader:
        X,Y = data
        with torch.no_grad():
            _,code = model(X)
        embeddings = np.hstack([embeddings,code.numpy().T])
        labels = np.hstack([labels,np.squeeze(Y.numpy())])
    return embeddings,labels

def set_seed(seed):
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(seed)
    np.random.seed(seed)


def get_data():
    set_seed(1)
    currDir = f"{os.getcwd()}/Mnist"
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081))])
    train_data = torchvision.datasets.MNIST(root=f'{currDir}/data', train=True, download=True, transform=transform)


    indices = np.random.permutation(len(train_data))
    train_subset = Subset(train_data, indices=indices[0:10000])
    return train_subset


def get_pca_data():
    train = get_data()
    train_size = 10000
    train_loader = torch.utils.data.DataLoader(train, batch_size=train_size, shuffle=False)
    train_iter = iter(train_loader)
    images, labels = train_iter.next()
    return images.view(train_size, 784).numpy().T, labels.numpy()


def get_autoencoder_data():
    train = get_data()
    return torch.utils.data.DataLoader(train, batch_size=32, shuffle=False)