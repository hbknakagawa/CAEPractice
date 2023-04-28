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
print(x_tensor.shape)


transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor(),torchvision.transforms.Normalize((0.5,), (0.5,))])
trainset = torchvision.datasets.MNIST(root = PATH, train = True, download = True, transform = transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size = BATCH_SIZE, shuffle = True, num_workers = 0) 

# model of neural network
model = CNNAE(FEAT_SIZE)

# model to gpu
if device == 'cuda':
    model = model.to(device)
else:
    model = model.to(device)


"""
def get_batch( x, BATCH_SIZE):
    '''
    Shuffle the input data and extract data specified by batch size.
    '''
    inds = random.sample(range(60000), k=BATCH_SIZE)
    return x[inds]  

def get_batch( x, BATCH_SIZE):
    import numpy as np
    '''
    Shuffle the input data and extract data specified by batch size.
    '''
    inds = np.random.randint(0, x.shape[0], BATCH_SIZE)
    return x[inds]
"""

def tensor2numpy(x):
    '''
    Convert tensor to numpy array.
    '''
    return x.cpu().detach().numpy().copy()


# Define MSE loss function and Adam optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=5e-4)


# training loop
train_loss_value=[]      #trainingのlossを保持するlist
train_acc_value=[]       #trainingのaccuracyを保持するlist

running_loss = 0.0
for epoch in range(NUM_EPOCHS):
    for (inputs, labels) in trainloader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, inputs)
        loss.backward()
        optimizer.step()
        

    scaled_loss = loss.item()
    if epoch % 10 == 0:
        print('[%d] loss: %.3f' % (epoch, scaled_loss) )
        running_loss = 0.0




          
print('Finished Training')
PATH = '../output/2_model_{}.pth'.format(NUM_EPOCHS)
torch.save(model.state_dict(), PATH)

"""
print('Get image feature, this process takes a few moments...')
_im_feat = model.encoder(x_tensor)
im_feat  = tensor2numpy(_im_feat)
np.save('../output/2_im_feat_{}.npy'.format(NUM_EPOCHS), im_feat)

# plot results
x_img = tensor2numpy(inputs[0]).transpose(1,2,0)
y_img = tensor2numpy(outputs[0]).transpose(1,2,0)
img = [x_img, y_img]

fig, ax = plt.subplots(1,2)
for i, data in enumerate( img ):
    ax[i].cla()
    ax[i].imshow(deprocess_img(data[:,:,0], 0.1,0.9 ), vmin=0, vmax=255, interpolation=None, cmap='gray')
plt.savefig('../output/2_cae_{}.png'.format(NUM_EPOCHS))
plt.show()
"""