import os
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from matplotlib import pyplot as plt


class ORL_MLP(nn.Module):
    def __init__(self):
        super(ORL_MLP, self).__init__()
        self.input_layer = nn.Linear(92 * 112, 1024)
        self.hidden_layer = nn.Linear(1024, 512)
        self.output_layer = nn.Linear(512, 20)

    def forward(self, x):
        x = F.relu(self.input_layer(x))
        x = F.relu(self.hidden_layer(x))
        x = self.output_layer(x)
        return F.log_softmax(dim=1, input=x)

    def train_nn(self, train_loader, optimizer, criterion, epochs, log_interval):
        train_loss = []
        for epoch in range(epochs):
            batch_loss = 0
            for batch_idx, (train_data, train_labels) in enumerate(train_loader):
                data, target = Variable(train_data), Variable(train_labels)
                model_out = self.forward(data)
                optimizer.zero_grad()
                loss = criterion(model_out, target)
                loss.backward()
                optimizer.step()
                batch_loss += loss
                if batch_idx % log_interval == 0:
                    print('Train Epoch: {} \tLoss: {:.6f}'.format(epoch, loss))
            batch_loss /= len(train_loader)
            train_loss.append(batch_loss)
        return train_loss

    def test_nn(self, test_loader, criterion):
        test_loss = 0
        correct = 0
        for data, target in test_loader:
            data, target = Variable(data), Variable(target)
            model_out = self.forward(data)
            # sum up batch loss
            test_loss += criterion(model_out, target)
            pred = model_out.data.max(1)[1]  # get the index of the max log-probability
            correct += pred.eq(target.data).sum()

        test_loss /= len(test_loader)
        print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(test_loss, correct, len(test_loader.dataset), 100. * correct / len(test_loader.dataset)))


def get_normalized_dataset(x, y):
    x = x / 255
    return TensorDataset(
        torch.from_numpy(x).float(),
        torch.from_numpy(y).long()
    )

def get_dataloader(x: np.array, y: np.array, batch_size: int, shuffle: bool = True, num_workers: int = 0):
    dataset = get_normalized_dataset(x, y)
    return DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers
    )

def read_dataset(data_dir=None):
    if isinstance(data_dir, str):
        path_to_datasets = os.path.abspath(os.path.join(__file__, "../"+data_dir))
        data = np.load(path_to_datasets)
        trainX = data['trainX']
        trainY = data['trainY']
        testX = data['testX']
        testY = data['testY']
        return trainX, trainY, testX, testY
    else:
        print("Please specify the data directory in string format")

def main(data_dir=None):
    if data_dir is not None:
        x_train, y_train, x_test, y_test = read_dataset(data_dir)
        x_train = x_train.reshape(x_train.shape[0], -1)
        x_test = x_test.reshape(x_test.shape[0], -1)

        train_loader = get_dataloader(x_train, y_train, batch_size=200, shuffle=True)
        test_loader = get_dataloader(x_test, y_test, batch_size=200, shuffle=True)

        orl_nn = ORL_MLP()
        learning_rate = 0.01
        # create a stochastic gradient descent optimizer
        optimizer = torch.optim.SGD(orl_nn.parameters(), lr=learning_rate, momentum=0.9)
        # create a loss function
        criterion = nn.CrossEntropyLoss()
        log_interval = 10
        epochs = 100
        train_loss = orl_nn.train_nn(train_loader, optimizer, criterion, epochs, log_interval)
        orl_nn.test_nn(test_loader,criterion)

        plt.plot(range(len(train_loss)), train_loss, label='Training Loss')
        plt.ylabel('Training Loss')
        plt.xlabel('# of Iterations')
        plt.savefig('Training Loss vs Iterations.png')

    else:
        print("Please provide the data directory")

if __name__=='__main__':
    data_dir = 'ORL_faces.npz'
    main(data_dir)


 


 