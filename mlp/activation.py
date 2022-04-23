import torch


def main():
    x = torch.tensor([
        [1, 2, 3],
        [2, 3, 4],
    ], dtype = torch.float)
    print(x.shape)
    print(torch.pow(x, 2))
    print(torch.mean(torch.pow(x, 2)))
    print(torch.sqrt(torch.mean(torch.pow(x, 2))))
    print(x / torch.sqrt(torch.mean(torch.pow(x, 2))))
    print(torch.mean(torch.pow(x, 2), dim=0))
    print(x / torch.sqrt(torch.mean(torch.pow(x, 2), dim=0)))


if __name__ == '__main__':
    main()

