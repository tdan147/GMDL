import numpy as np
import matplotlib.pyplot as plt
from torch import nn, optim, utils, cuda
import torch
import torchvision
from torchvision import datasets, transforms, models
import math
import torch.nn.functional as F
import ssl
from sklearn.metrics import confusion_matrix

ssl._create_default_https_context = ssl._create_unverified_context

criterion_cross_entropy = nn.CrossEntropyLoss()
criterion_mse = nn.MSELoss()
mse_reduction_none = nn.MSELoss(reduction='none')
batch = 128
device = 'cuda' if cuda.is_available() else 'cpu'

transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.1307,), (0.3081)), ])

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.manual_seed(1)
np.random.seed(1)
all_train = torchvision.datasets.MNIST(root='./root', train=True, download=True, transform=transform)
train_size = int(0.8 * len(all_train))
train, validation = utils.data.random_split(all_train, [train_size, len(all_train) - train_size])

testset_mnist = torchvision.datasets.MNIST(root='./root', train=False, download=True, transform=transform)

train_loader = utils.data.DataLoader(train, batch_size=batch, shuffle=True)
validation_loader = utils.data.DataLoader(validation, batch_size=len(validation), shuffle=True)

test_loader = utils.data.DataLoader(testset_mnist, batch_size=batch, shuffle=False)


def create_dataset_osr(dataset, is_grey):
    transform_osr = transforms.Compose(
        [transforms.ToTensor(), transforms.Grayscale(num_output_channels=1), transforms.Resize(28)])
    testset_osr = dataset(root="./data", train=False, download=True, transform=transform_osr)
    testset_osr_data = testset_osr.data / 255
    mean = testset_osr_data.mean()
    std = testset_osr_data.std()

    transform_osr = transforms.Compose(
        [transforms.ToTensor(), transforms.Resize(28),
         transforms.Normalize((mean,), (std))]) if is_grey else transforms.Compose(
        [transforms.ToTensor(), transforms.Grayscale(num_output_channels=1), transforms.Resize(28),
         transforms.Normalize((mean,), (std))])
    return dataset(root="./data", train=False, download=True, transform=transform_osr)


dataset_cifar = create_dataset_osr(torchvision.datasets.CIFAR10, 0)

dataset_kmnist = create_dataset_osr(torchvision.datasets.KMNIST, 1)


def generate_test(mnist, osr):
    osr.targets = [10 for i in range(len(osr.targets))]
    return torch.utils.data.DataLoader(torch.utils.data.ConcatDataset([mnist, osr]), batch_size=batch, shuffle=True)


test_loader_cifar = generate_test(testset_mnist, dataset_cifar)

test_loader_kmnist = generate_test(testset_mnist, dataset_kmnist)


class Encoder(nn.Module):
    def __init__(self, input_size):
        super(Encoder, self).__init__()
        self.input_size = input_size
        self.fc1_input = int(((self.input_size - 4) / 2 - 4) / 2)  # TODO:change name
        self.conv1 = torch.nn.Conv2d(1, 6, 5)  # in channels, out channels, kernel size
        self.pool = torch.nn.MaxPool2d(2, 2)  # size,stride
        self.conv2 = torch.nn.Conv2d(6, 16, 5)
        self.fc1 = torch.nn.Linear(16 * self.fc1_input * self.fc1_input, 300)
        self.fc2 = torch.nn.Linear(300, 250)
        self.fc3 = torch.nn.Linear(250, 150)

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
    def __init__(self, input_size, num_labels):
        super(Decoder, self).__init__()
        self.input_size = input_size
        self.num_labels = num_labels
        self.upsample2 = input_size - 4
        self.upsample1 = int(self.upsample2 / 2 - 4)
        self.fc3_input = int(self.upsample1 / 2)  # TODO:change name
        self.t_fc1 = torch.nn.Linear(150, 250)
        self.t_fc2 = torch.nn.Linear(250, 300)
        self.t_fc3 = torch.nn.Linear(300, 16 * self.fc3_input * self.fc3_input)
        self.t_conv1 = nn.ConvTranspose2d(16, 6, 5)
        self.t_conv2 = nn.ConvTranspose2d(6, 1, 5)
        self.classifier = torch.nn.Linear(input_size ** 2, num_labels)

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
    def __init__(self, input_size, num_labels):
        super(AutoEncoder, self).__init__()
        self.encoder = Encoder(input_size)
        self.decoder = Decoder(input_size, num_labels)

    def forward(self, x):
        encoded = self.encoder(x)
        return self.decoder(encoded)


def train_model(train_loader, validation_loader, epochs, model, optimizer, num_labels):
    train_losses = []
    validation_losses = []
    train_acc = []
    validation_acc = []
    for epoch in range(epochs):  # loop over the dataset multiple times

        curr_loss = 0.0
        curr_val_loss = 0.0
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

        for i, data in enumerate(validation_loader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data

            # mode to device/cuda
            inputs, labels = inputs.to(device), labels.to(device)

            # forward + optimize
            outputs, tags = model(inputs)
            output_loss = criterion_mse(outputs, inputs)
            tags_loss = criterion_cross_entropy(tags.squeeze(), labels)
            loss = (output_loss + tags_loss) / 2

            curr_val_loss += loss.item()

        print(
            'epoch [{}/{}], train loss:{:.4f}'.format(epoch + 1, epochs, curr_loss / (math.floor(len(train) / batch))))
        print('epoch [{}/{}], validation loss:{:.4f}'.format(epoch + 1, epochs,
                                                             curr_val_loss / (math.floor(len(validation) / batch))))

        train_losses.append(curr_loss / (math.floor(len(train) / batch)))
        validation_losses.append(curr_val_loss / (math.floor(len(validation) / batch)))
        train_acc.append(accuracy_mnist(train_loader, model, num_labels)[0])
        validation_acc.append(accuracy_mnist(validation_loader, model, num_labels)[0])

    return train_losses, validation_losses, train_acc, validation_acc


def reconstruct(model, data):
    reconstruction, labels = model.forward(data.unsqueeze(1).to(device))
    return reconstruction, labels


def map_OOD(v_norm, pred_label, threshold):
    return pred_label if v_norm < threshold else 10


def map_binary(v_norm, threshold):
    return 0 if v_norm < threshold else 1


def extract_features(x, model):
    return model.encoder.conv1(x)


def calc_threshold(data_loader, percentile, model):
    style_loss = []
    for images, _ in data_loader:
        img = images.to(device)
        with torch.no_grad():
            reconstructed, tags = model(img)

        style_loss.append(get_style_loss(img, reconstructed, model).detach().cpu().numpy())
    style_loss = np.concatenate(style_loss)
    return np.percentile(style_loss, percentile)


def get_style_loss(img, reconstructed, model):
    channel = 6
    conv_img = extract_features(img, model).flatten(2, 3)
    conv_reco = extract_features(reconstructed, model).flatten(2, 3)
    gram_img = torch.matmul(conv_img, torch.transpose(conv_img, 1, 2))
    gram_reco = torch.matmul(conv_reco, torch.transpose(conv_reco, 1, 2))
    return torch.sum(((gram_img - gram_reco) ** 2), (1, 2)) / (channel ** 2)


def accuracy_OOD(data_loader, model, threshold, num_labels):
    correct_count, all_count = 0, 0
    confusion = np.zeros((num_labels, num_labels))
    for images, labels in data_loader:
        labels = labels.numpy()
        img = images.to(device)
        with torch.no_grad():
            reconstructed, tags = model(img)

        style_loss = get_style_loss(img, reconstructed, model)
        _, pred_labels = torch.max(tags.data, 1)
        pred = np.asarray(list(map(map_OOD, style_loss.detach().cpu().numpy(), pred_labels.detach().cpu().numpy(),
                                   np.ones(labels.shape[0]) * threshold)))
        correct_count += (pred == labels).sum().item()
        all_count += labels.shape[0]
        confusion += confusion_matrix(labels, pred, labels=range(num_labels))

    return correct_count / all_count * 100, confusion


def accuracy_mnist(data_loader, model, num_labels):
    correct_count, all_count = 0, 0
    confusion = np.zeros((num_labels, num_labels))
    for images, labels in data_loader:
        labels = labels.numpy()
        img = images.to(device)
        with torch.no_grad():
            _, tags = model(img)

        _, pred_labels = torch.max(tags.data, 1)
        pred_labels = pred_labels.detach().cpu().numpy()
        correct_count += (pred_labels == labels).sum().item()
        all_count += labels.shape[0]
        confusion += confusion_matrix(labels, pred_labels, labels=range(num_labels))

    return correct_count / all_count * 100, confusion


def accuracy_binary(data_loader, model, threshold, num_labels):
    correct_count, all_count = 0, 0
    confusion = np.zeros((num_labels, num_labels))
    for images, labels in data_loader:
        labels = np.asarray(list(map(lambda label: 0 if label < 10 else 1, labels.numpy())))
        img = images.to(device)
        with torch.no_grad():
            reconstructed, tags = model(img)

        style_loss = get_style_loss(img, reconstructed, model)
        _, pred_labels = torch.max(tags.data, 1)
        pred = np.asarray(
            list(map(map_binary, style_loss.detach().cpu().numpy(), np.ones(labels.shape[0]) * threshold)))
        correct_count += (pred == labels).sum().item()
        all_count += labels.shape[0]
        confusion += confusion_matrix(labels, pred, labels=range(num_labels))

    return correct_count / all_count * 100, confusion


def plot_reconstruction(model, loader):
    amount_img = 3
    data_iter = iter(loader)
    images, test_labels = data_iter.next()
    images = images[:amount_img].squeeze()
    reconstruction, labels = reconstruct(model, images)
    reconstruction = reconstruction.detach().cpu().squeeze().numpy()

    f, axs = plt.subplots(2, amount_img)

    for i in range(amount_img):
        axs[1, i].set_title(f"predicted label:{labels[i]}")
        axs[0, i].imshow(images[i], cmap='gray')
        axs[1, i].imshow(reconstruction[i], cmap='gray')

    axs[0, 0].set_ylabel("original")
    axs[1, 0].set_ylabel("reconstructed")
    plt.suptitle("Origin vs Reconstructed images")
    plt.show()


def plot(train, valid, title, ylabel):
    plt.figure()
    plt.title(f'{title}')
    plt.plot(np.arange(len(train)), train, label='train')
    plt.plot(np.arange(len(valid)), valid, label='validation')
    plt.xlabel("Epoch")
    plt.ylabel(ylabel)
    plt.legend()
    plt.show()


def train_new_model(train_loader, validation_loader, input_size, num_labels, model_name, epoch):
    model = AutoEncoder(input_size, num_labels).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    train_losses, validation_losses, train_MNIST_acc, validation_MNIST_acc = train_model(train_loader,
                                                                                                    validation_loader,
                                                                                                    epoch, model,
                                                                                                    optimizer,
                                                                                                    num_labels)

    torch.save(model.state_dict(), model_name)
    plot(train_losses, validation_losses, "Train & Validation Loss", "loss")
    plot(train_MNIST_acc, validation_MNIST_acc, "MNIST Accuracy", "accuracy")
    # plot_reconstruction(model, loader)


def load_assess_model(train_loader, test_loader_mnist, test_loader, input_size, num_labels, percentile):
    model = AutoEncoder(input_size, num_labels).to(device)
    model.load_state_dict(torch.load("./model_epoch100"))
    model.eval()
    # plot_reconstruction(model, loader)
    np.set_printoptions(suppress=True)
    threshold_ood = calc_threshold(train_loader, percentile, model)
    acc_mnist, confusion_mnist = accuracy_mnist(test_loader_mnist, model, num_labels)
    acc_binary, confusion_binary = accuracy_binary(test_loader, model, threshold_ood, 2)
    acc_OOD, confusion_OOD = accuracy_OOD(test_loader, model, threshold_ood, num_labels + 1)
    print(f'MNIST Accuracy: {acc_mnist}')
    print(f'MNIST Confusion Matrix:\n {confusion_mnist}')
    print(f'Binary Accuracy: {acc_binary}')
    print(f'Binary Confusion Matrix:\n {confusion_binary}')
    print(f'OOD Accuracy: {acc_OOD}')
    print(f'OOD Confusion Matrix:\n {confusion_OOD}')


model_name = "./model_epoch5"
train_new_model(train_loader, validation_loader, 28, 10, model_name, epoch=5)
# load_assess_model(train_loader, test_loader, test_loader_cifar, 28, 10, percentile=98)
# load_assess_model(train_loader, test_loader, test_loader_kmnist, 28, 10, percentile=98)
