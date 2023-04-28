from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import MNIST

if __name__ == '__main__':

    # datasetの読み出し
    bs = 128 # batch size 
    transform = transforms.ToTensor()
    testset = MNIST(root='./data', train=False, download=True, transform=transform)
    testloader = DataLoader(testset, batch_size=bs, shuffle=False)

    # dataの取り出し
    tmp = testloader.__iter__()
    x1, y1 = tmp.next() 
    x2, y2 = tmp.next()