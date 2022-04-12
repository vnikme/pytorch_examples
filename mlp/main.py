import torch
import torch.utils.data
from torch import nn, optim

torch.manual_seed(1)


class MLP(nn.Module):
    def __init__(self, layer_sizes, device):
        super(MLP, self).__init__()

        # input parameters
        self.layer_sizes = layer_sizes

        # layers
        self.layers = torch.nn.ModuleList()
        for i in range(1, len(layer_sizes)):
            self.layers.append(
                torch.nn.Linear(
                    layer_sizes[i - 1],
                    layer_sizes[i]
                ).to(device)
            )


    def forward(self, x):
        for i in range(len(self.layers) - 1):
            layer = self.layers[i]
            x = layer(x)
            x = torch.nn.functional.relu(x)
            #x = torch.nn.functional.sigmoid(x)
        x = self.layers[-1](x)
        return x


def prepare_batch(device):
    x, y = [], []
    for i in range(2):
        for j in range(2):
            x.append([i, j])
            #y.append([i ^ j])                   # xor
            #y.append([1 if i or j else 0])      # or
            y.append([i ^ j, 1 if i or j else 0, 1 if i and j else 0, 0, 1])        # xor, or, and, 0, 1
            #y.append([1 if i or j else 0, 1 if i and j else 0, 0, 1])
    x = torch.tensor(x, dtype=torch.float).to(device)
    y = torch.tensor(y, dtype=torch.float).to(device)
    return x, y


def main():
    cuda = torch.cuda.is_available() and False
    device = torch.device("cuda" if cuda else "cpu")
    print(cuda, device)

    model = MLP([2, 7, 7, 7, 7, 7, 5], device)
    criterion = torch.nn.BCEWithLogitsLoss().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=0.001)

    epoch = 0
    while True:
        model.to(device)
        model.train()
        epoch += 1
        x, y = prepare_batch(device)
        model.zero_grad()
        output = model.forward(x)
        print(torch.nn.functional.sigmoid(output))
        print("")
        loss = criterion(output, y)
        loss.backward()
        #torch.nn.utils.clip_grad_norm_(model.parameters(), 3.0)
        opt.step()
        print('Epoch: {}, loss: {}'.format(epoch, float(loss)))


if __name__ == '__main__':
    main()

