from __future__ import print_function, division

import torch as tr
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms



class AlexNet(nn.Module):
    def __init__(self, classes=1000, is_train=True):
        super(AlexNet, self).__init__()
        self.classes = classes
        self.is_train = is_train
        self.conv1 = nn.Conv2d(1, 64, 11, stride=4, padding=2)
        self.conv2 = nn.Conv2d(64, 192, 5, padding=2)
        self.conv3 = nn.Conv2d(192, 384, 3, padding=1)
        self.conv4 = nn.Conv2d(384, 256, 3, padding=1)
        self.conv5 = nn.Conv2d(256, 256, 3,  padding=1)
        self.fc1 = nn.Linear(6*6*256, 4096)
        self.fc2 = nn.Linear(4096, 4096)
        self.fc3 = nn.Linear(4096, classes)


    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), 3, stride=2)
        x = F.max_pool2d(F.relu(self.conv2(x)), 3, stride=2)
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.max_pool2d(F.relu(self.conv5(x)), 3, stride=2)
        x = x.view(x.size(0), 6*6*256)
        x = F.dropout(x, 0.5, self.is_train)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, 0.5, self.is_train)
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return F.log_softmax(x, dim=1)

    def num_flat_features(self, x):
        size = x.size(0)
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

if __name__ == "__main__":
    import argparse
    import torch.optim as optim

    parser = argparse.ArgumentParser('Train alexnet on MNIST')
    parser.add_argument('--batch-size', type=int, default=10)
    parser.add_argument('--val-batch-size', type=int, default=1)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--end-epoch', type=int, default=100)
    args = parser.parse_args()


    trans = transforms.Compose([
                transforms.Resize((224,224)),
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))])

    train_loader = tr.utils.data.DataLoader(
        datasets.FashionMNIST('/media/tx-deepocean/Data/dataset/fashionMNIST', train=True, download=True, transform=trans),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=8
    )

    val_loader = tr.utils.data.DataLoader(
        datasets.FashionMNIST('/media/tx-deepocean/Data/dataset/fashionMNIST', train=False, download=True, transform=trans),
        batch_size=args.val_batch_size,
        shuffle=True
    )

    device = tr.device("cuda:1")

    net = AlexNet(classes=10)
    net.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=args.momentum)
    # optimizer = optim.Adam(net.parameters(), lr=args.lr)

    for epoch in range(args.end_epoch):
        running_loss = 0.0
        for i, data in enumerate(train_loader, 0):
            inputs, labels = data[0].to(device), data[1].to(device)
            # print(inputs.shape)
            optimizer.zero_grad()

            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if i % 200 == 199:
                print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 2000))
                running_loss = 0.0
    print('Finish Training')


