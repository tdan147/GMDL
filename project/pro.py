import numpy as np
import matplotlib.pyplot as plt
from torch import nn, optim, utils, cuda
import torch
import torchvision
from torchvision import datasets, transforms, models
import math
import torch.nn.functional as F


criterion_cross_entropy = nn.CrossEntropyLoss()
criterion_mse = nn.MSELoss()
mse_reduction_none = nn.MSELoss(reduction='none')
batch = 64
device = 'cuda' if cuda.is_available() else 'cpu'

transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.1307,), (0.3081)), ])

transform_cifar = transforms.Compose(
    [transforms.ToTensor(), transforms.Grayscale(num_output_channels=1), transforms.Normalize((.5, ), (.5, ))])

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.manual_seed(1)
np.random.seed(1)
all_train = torchvision.datasets.MNIST(root='./root', train=True, download=True, transform=transform)
train_size = int(0.8 * len(all_train))
train, validation = utils.data.random_split(all_train, [train_size, len(all_train) - train_size])

test = torchvision.datasets.MNIST(root='./root', train=False, download=True, transform=transform)

train_loader = utils.data.DataLoader(train, batch_size=batch, shuffle=True)
validation_loader = utils.data.DataLoader(validation, batch_size=len(validation), shuffle=True)
test_loader = utils.data.DataLoader(test, batch_size=batch, shuffle=False)

testset_cifar = torchvision.datasets.CIFAR10(root="./data", train=False, download=True, transform=transform_cifar)
testloader_cifar = torch.utils.data.DataLoader(testset_cifar, batch_size=4, shuffle=False)



# to remove
data_iter = iter(train_loader)
images, labels = data_iter.next()
print(images.shape)
print(labels.shape)


class Encoder(nn.Module):
    def __init__(self, input_size):
        super(Encoder, self).__init__()
        self.input_size = input_size
        self.fc1_input = int(((self.input_size - 4) / 2 - 4) / 2)   #TODO:change name
        self.conv1 = torch.nn.Conv2d(1, 6, 5)  # in channels, out channels, kernel size
        self.pool = torch.nn.MaxPool2d(2, 2)  # size,stride
        self.conv2 = torch.nn.Conv2d(6, 16, 5)
        self.fc1 = torch.nn.Linear(16 * self.fc1_input * self.fc1_input, 120)
        self.fc2 = torch.nn.Linear(120, 84)
        self.fc3 = torch.nn.Linear(84, 10)

    def forward(self, x):
        # two convolutional layers, a bunch of ff layers
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * self.fc1_input * self.fc1_input)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class Decoder(nn.Module):
    def __init__(self, input_size, classify_output):
        super(Decoder, self).__init__()
        self.input_size = input_size
        self.classify_output = classify_output
        self.upsample2 = input_size - 4
        self.upsample1 = int(self.upsample2 / 2 - 4)
        self.fc3_input = int(self.upsample1 / 2)
        self.t_fc1 = torch.nn.Linear(10, 84)
        self.t_fc2 = torch.nn.Linear(84, 120)
        self.t_fc3 = torch.nn.Linear(120, 16 * self.fc3_input * self.fc3_input)
        self.t_conv1 = nn.ConvTranspose2d(16, 6, 5)
        self.t_conv2 = nn.ConvTranspose2d(6, 1, 5)
        self.classifier = torch.nn.Linear(input_size ** 2, classify_output)

    def forward(self, x):
        x = F.relu(self.t_fc1(x))
        x = F.relu(self.t_fc2(x))
        x = self.t_fc3(x)
        x = x.view(-1, 16, self.fc3_input, self.fc3_input)
        x = nn.Upsample((self.upsample1, self.upsample1))(x)
        x = F.relu(self.t_conv1(x))
        x = nn.Upsample((self.upsample2, self.upsample2))(x)
        x = self.t_conv2(x)
        return x, self.classifier(x.view(-1, self.input_size ** 2))


class AutoEncoder(nn.Module):
    def __init__(self, input_size, classify_output):
        super(AutoEncoder, self).__init__()
        self.encoder = Encoder(input_size)
        self.decoder = Decoder(input_size, classify_output)

    def forward(self, x):
        encoded = self.encoder(x)
        return self.decoder(encoded)


def train_model(epochs, model, optimizer, ood_threshold):
    train_losses = []
    train_acc = []
    for epoch in range(epochs):  # loop over the dataset multiple times

        curr_loss = 0.0
        for i, data in enumerate(train_loader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data

            # mode to device/cuda
            inputs, labels = inputs.to(device), labels.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs, tags = model(inputs)
            output_loss = criterion_mse(outputs, inputs)
            tags_loss = criterion_cross_entropy(tags.squeeze(), labels)
            loss = (output_loss + tags_loss) / 2
            loss.backward()
            optimizer.step()

            curr_loss += loss.item()

        # v_outputs, v_tags = model.forward_train(valid_data)
        # valid_tags_loss = criterion_cross_entropy(v_tags.squeeze(), valid_labels)
        # valid_outputs_loss = criterion_mse(v_outputs, valid_labels)
        # valid_loss = (output_loss + tags_loss) / 2

        train_losses.append(curr_loss / (math.floor(len(train) / batch)))
        # validation_losses.append(valid_loss.item())

        train_acc.append(accuracy(train_loader, model, ood_threshold))
        # valid_acc.append(accuracy(validation_loader, model))
    return train_losses, train_acc
    # return train_losses, validation_losses, train_acc, valid_acc


def map_OOD(v_norm, pred_label, threshold):
    return pred_label if v_norm < threshold else np.asarray([10])


def accuracy(data_loader, model, threshold):
    correct_count, all_count = 0, 0
    for images, labels in data_loader:
        labels = labels.numpy()
        img = images.to(device)
        with torch.no_grad():
            reconstucted, tags = model(img)

        norm = torch.norm((img - reconstucted).view(img.shape[0], -1), dim=1)
        _, pred_labels = torch.max(tags.data, 1)
        pred = np.asarray(list(map(map_OOD, norm.detach().cpu().numpy(), pred_labels.detach().cpu().numpy(), np.ones(labels.shape[0])*threshold)))
        correct_count += (pred == labels).sum().item()
        all_count += labels.shape[0]

    return correct_count / all_count * 100


model = AutoEncoder(28, 10).to(device)
optimizer = optim.Adam(model.parameters(), lr=0.0001)
train_model(2, model, optimizer, 5)
torch.save(model.state_dict(), "./model")

