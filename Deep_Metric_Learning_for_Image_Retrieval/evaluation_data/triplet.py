import torch.nn as nn
import torch

class EmbeddingNet(nn.Module):
    def __init__(self):
        super(EmbeddingNet, self).__init__()
        self.convnet = nn.Sequential(nn.Conv2d(1, 32, 5), nn.PReLU(),
                                     nn.MaxPool2d(2, stride=2),
                                     nn.Conv2d(32, 64, 5), nn.PReLU(),
                                     nn.MaxPool2d(2, stride=2))

        self.fc = nn.Sequential(nn.Linear(64 * 4 * 4, 512),
                                nn.PReLU(),
#                                 nn.Linear(512, 512),
#                                 nn.PReLU(),
#                                 nn.Linear(512, 512)
                                )

    def forward(self, x):
        output = self.convnet(x)
        output = output.view(output.size()[0], -1)
        output = self.fc(output)
        return output

    def get_embedding(self, x):
        return self.forward(x)

class TripletNet(nn.Module):
    def __init__(self, embedding_net):
        super(TripletNet, self).__init__()
        self.embedding_net = embedding_net

    def forward(self, x1):
        output1 = self.embedding_net(x1)
        return output1

    def get_embedding(self, x):
        return self.embedding_net(x)

def get_model(is_gray=False):
    return TripletNet(EmbeddingNet())