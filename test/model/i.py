import numpy as np
import matplotlib.pylab as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision

# own library
from dplm.model import CNNAE
from dplm.utils import normalization, deprocess_img

# settings
vmin = 0.1
vmax = 0.9
FEAT_SIZE  = 10
BATCH_SIZE = 128
NUM_EPOCHS = 10000
device = 'cuda' if torch.cuda.is_available() else 'cpu'
PATH = "Datasetのディレクトリpath"

# load mnist dataset
mnist_data = torchvision.datasets.MNIST('~/tmp/mnist', train=True, download=True)
x_data = mnist_data.data.numpy()
x_norm = normalization(x_data, (x_data.min(), x_data.max()), (vmin, vmax) )   # normalization
x_train = x_norm.reshape([60000,28,28,1]).transpose(0,3,1,2)   # Transpose the array for input to the CNN layer. Note that pytorch is channel first [B, C, H, W].
x_tensor = torch.Tensor(x_train)    # convert numpy array to torch.Tensor


transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor(),torchvision.transforms.Normalize((0.5,), (0.5,))])
trainset = torchvision.datasets.MNIST(root = PATH, train = True, download = True, transform = transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size = BATCH_SIZE, shuffle = True, num_workers = 1) 
print(trainloader)

len(trainloader[0])

num_epochs = 1
outputs = []
for epoch in range(num_epochs):
     for (img, _) in trainloader:
         # img = img.reshape(-1, 28*28) # -> use for Autoencoder_Linear
         recon = model(img)
         loss = criterion(recon, img)
         optimizer.zero_grad()
         loss.backward()
         optimizer.step()
     print(f'Epoch:{epoch+1}, Loss:{loss.item():.4f}')
     outputs.append((epoch, img, recon))

# model = CNNAE()

# criterion = nn.MSELoss()
# optimizer = torch.optim.Adam(model.parameters(),
#                              lr=1e-3, 
#                              weight_decay=1e-5)