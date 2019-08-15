from __future__ import print_function
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from torch.autograd import Variable

class BatchNorm(nn.Module):
    def __init__(self, num_features):
        super(BatchNorm, self).__init__()
        self.num_features = num_features
        self.gamma = nn.Parameter(torch.randn(self.num_features))
        self.beta = nn.Parameter(torch.randn(self.num_features))
        self.moving_mean = Variable(torch.zeros(self.num_features))
        self.moving_var = Variable(torch.zeros(self.num_features))
        self.is_training=True
        self.eps=1e-5
        self.momentum=0.1

    def forward(self, X):
        outputs, self.moving_mean, self.moving_var = batch_norm(X, self.gamma, self.beta,self.moving_mean, self.moving_var,
            self.is_training, self.eps, self.momentum)
        return outputs


def batch_norm(X, gamma, beta, moving_mean, moving_var, is_training=True, eps=1e-5, momentum=0.1):

    if len(X.shape) == 2:#BatchNorm1d
        x_mean = torch.mean(X, dim=0, keepdim=True)
        x_var = torch.mean((X - x_mean) ** 2, dim=0, keepdim=True)
        if torch.cuda.is_available():
            x_mean=x_mean.cuda()
            x_var=x_var.cuda()
            moving_mean=moving_mean.cuda()
            moving_var=moving_var.cuda()
        if is_training:
            x_hat = (X - x_mean) / torch.sqrt(x_var + eps)
            moving_mean[:] = moving_momentum * moving_mean + (1. - moving_momentum) * x_mean
            moving_var[:] = moving_momentum * moving_var + (1. - moving_momentum) * x_var
        else:
            x_hat = (x - moving_mean) / torch.sqrt(moving_var + eps)
        outputs = gamma.view_as(x_mean) * x_hat + beta.view_as(x_mean)

    else len(X.shape) == 4:#BatchNorm2d
        x_mean = torch.mean(X, dim=(0, 2, 3))
        x_mean = x_mean.view(1, X.size(1), 1, 1)
        x_var = torch.sqrt(torch.var(X, dim=(0, 2 , 3), unbiased=False) + eps)
        x_var = x_var.view(1, X.size(1), 1, 1)
        invstd = 1/x_var
        x_hat = (X-x_mean)*invstd
        if torch.cuda.is_available():
            x_mean=x_mean.cuda()
            x_var=x_var.cuda()
            moving_mean=moving_mean.cuda()
            moving_var=moving_var.cuda()
        if is_training:
            x_hat = (X-x_mean)*invstd
            moving_mean = momentum * moving_mean.view(1, X.size(1), 1, 1) + (1.0 - momentum) * x_mean
            moving_var = momentum * moving_var.view(1, X.size(1), 1, 1) + (1.0 - momentum) * x_var
        else:
            x_hat = (X - moving_mean.view(1, X.size(1), 1, 1)) / torch.sqrt(moving_var.view(1, X.size(1), 1, 1) + eps)

        outputs = gamma.view_as(x_mean) * x_hat + beta.view_as(x_mean)

    return outputs, moving_mean, moving_var

class conv_bn_net(nn.Module):
    def __init__(self):
        super(conv_bn_net, self).__init__()
        self.stage1 = nn.Sequential(
            nn.Conv2d(1, 6, 3, padding=1),
            BatchNorm(6),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(6, 16, 5),
            BatchNorm(16),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2)
        )
        
        self.classfy = nn.Linear(400, 10)
    def forward(self, x):
        x = self.stage1(x)
        x = x.view(x.shape[0], -1)
        x = self.classfy(x)
        return x

class conv_no_bn_net(nn.Module):
    def __init__(self):
        super(conv_no_bn_net, self).__init__()
        self.stage2 = nn.Sequential(
            nn.Conv2d(1, 6, 3, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(6, 16, 5),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2)
        )
        
        self.classfy = nn.Linear(400, 10)
    def forward(self, x):
        x = self.stage2(x)
        x = x.view(x.shape[0], -1)
        x = self.classfy(x)
        return x

def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        criterion = nn.CrossEntropyLoss()
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))

def test(args, model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    loss_list=[]
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item() # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True) # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    return test_loss, 100*correct










def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=30, metavar='N',
                        help='number of epochs to train (default: 30)')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                        help='SGD momentum (default: 0.1)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    
    parser.add_argument('--save-model', action='store_true', default=False,
                        help='For Saving the current Model')
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")

    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=args.batch_size, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=False, transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=args.test_batch_size, shuffle=True, **kwargs)


    model = conv_bn_net().to(device)
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
    loss_list=[]
    acc_list=[]
    for epoch in range(1, args.epochs + 1):
        train(args, model, device, train_loader, optimizer, epoch)
        loss,acc=test(args, model, device, test_loader)
        loss_list.append(loss)
        acc_list.append(acc)
    x1 = range(1, 31)
    x2 = range(1, 31)
    y1 = loss_list
    y2 = acc_list
    plt.subplot(2, 1, 1)
    plt.plot(x1, y1, 'o-')
    plt.xlabel('Epoch')
    plt.ylabel('Test loss')
    plt.subplot(2, 1, 2)
    plt.plot(x2, y2, 'o-')
    plt.xlabel('Epoch')
    plt.ylabel('Test Acc')
    plt.show()
    plt.savefig("accuracy_loss.jpg")


    if (args.save_model):
        torch.save(model.state_dict(),"mnist_cnn.pt")
        
if __name__ == '__main__':
    main()