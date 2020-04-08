import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as I


class Net(nn.Module):

    def __init__(self):
        """
        Define layers of the model
        input image : 1 x 224 x 224, grayscale squared images
        """
        super(Net, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=5)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout1 = nn.Dropout(p=0.1)

        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3)
        self.bn2 = nn.BatchNorm2d(64)
        # pooling layer
        self.dropout2 = nn.Dropout(p=0.2)

        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3)
        self.bn3 = nn.BatchNorm2d(128)
        # pooling layer
        self.dropout3 = nn.Dropout(p=0.3)

        self.conv4 = nn.Conv2d(
            in_channels=128, out_channels=256, kernel_size=1)
        self.bn4 = nn.BatchNorm2d(256)
        # pooling layer
        self.dropout4 = nn.Dropout(p=0.4)

        self.fc1 = nn.Linear(in_features=43264, out_features=1000)
        self.bn5 = nn.BatchNorm1d(1000)
        self.dropout5 = nn.Dropout(p=0.5)

        self.fc2 = nn.Linear(in_features=1000, out_features=1000)
        self.bn6 = nn.BatchNorm1d(1000)
        self.dropout6 = nn.Dropout(p=0.6)

        self.fc3 = nn.Linear(1000, 136)

        I.xavier_uniform_(self.fc1.weight.data)
        I.xavier_uniform_(self.fc2.weight.data)
        I.xavier_uniform_(self.fc3.weight.data)

    def forward(self, x):
        """ The feedforward behavior of the model
        Arguments:
            x -- input image
        """
        x = self.dropout1(self.pool(F.elu(self.bn1(self.conv1(x)))))
        x = self.dropout2(self.pool(F.elu(self.bn2(self.conv2(x)))))
        x = self.dropout3(self.pool(F.elu(self.bn3(self.conv3(x)))))
        x = self.dropout4(self.pool(F.elu(self.bn4(self.conv4(x)))))

        # flatten
        x = x.view(x.size(0), -1)

        x = self.dropout5(F.elu(self.bn5(self.fc1(x))))
        x = self.dropout6(F.elu(self.bn6(self.fc2(x))))
        x = self.fc3(x)

        return x
