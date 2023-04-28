import numpy as np
import matplotlib.pylab as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.datasets as datasets
import torchvision

# own library
from dplm.model import Net

# settings
vmin = 0.1
vmax = 0.9
FEAT_SIZE  = 10
BATCH_SIZE = 128
NUM_EPOCHS = 10000
device = 'cuda' if torch.cuda.is_available() else 'cpu'

WEIGHT_DECAY = 0.005
LEARNING_RATE = 0.0001
PATH = "Datasetのディレクトリpath"


# load mnist dataset
transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor(),torchvision.transforms.Normalize((0.5,), (0.5,))])
trainset = datasets.MNIST(root = PATH, train = True, download = True, transform = transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size = BATCH_SIZE, shuffle = True, num_workers = 1) 


# model of neural network
net = Net()

# model to gpu
if device == 'cuda':
    net = net.to(device)
else:
    net = net.to(device)

# Define MSE loss function and Adam optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(net.parameters(), lr = LEARNING_RATE, weight_decay = WEIGHT_DECAY)


# training loop
train_loss_value=[]      #trainingのlossを保持するlist
train_acc_value=[]       #trainingのaccuracyを保持するlist


for epoch in range(NUM_EPOCHS):
    print('epoch', epoch+1)    #epoch数の出力
    for (inputs, labels) in trainloader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
    sum_loss = 0.0          #lossの合計
    sum_correct = 0         #正解率の合計
    sum_total = 0           #dataの数の合計

    #train dataを使ってテストをする(パラメータ更新がないようになっている)
    for (inputs, labels) in trainloader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        sum_loss += loss.item()                            #lossを足していく
        _, predicted = outputs.max(1)                      #出力の最大値の添字(予想位置)を取得
        sum_total += labels.size(0)                        #labelの数を足していくことでデータの総和を取る
        sum_correct += (predicted == labels).sum().item()  #予想位置と実際の正解を比べ,正解している数だけ足す
    print("train mean loss={}, accuracy={}"
            .format(sum_loss*BATCH_SIZE/len(trainloader.dataset), float(sum_correct/sum_total)))  #lossとaccuracy出力
    train_loss_value.append(sum_loss*BATCH_SIZE/len(trainloader.dataset))  #traindataのlossをグラフ描画のためにlistに保持
    train_acc_value.append(float(sum_correct/sum_total))   #traindataのaccuracyをグラフ描画のためにlistに保持


plt.figure(figsize=(6,6))      #グラフ描画用

#以下グラフ描画
plt.plot(range(NUM_EPOCHS), train_loss_value)
plt.xlim(0, NUM_EPOCHS)
plt.ylim(0, 2.5)
plt.xlabel('EPOCH')
plt.ylabel('LOSS')
plt.legend(['train loss'])
plt.title('loss')
plt.savefig("loss_image.png")
plt.clf()

plt.plot(range(NUM_EPOCHS), train_acc_value)
plt.xlim(0, NUM_EPOCHS)
plt.ylim(0, 1)
plt.xlabel('EPOCH')
plt.ylabel('ACCURACY')
plt.legend(['train acc'])
plt.title('accuracy')
plt.savefig("accuracy_image.png")

"""
print('Finished Training')
PATH = '../output/2_model_{}.pth'.format(NUM_EPOCHS)
torch.save(model.state_dict(), PATH)

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