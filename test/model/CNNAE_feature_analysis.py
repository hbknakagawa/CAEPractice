import numpy as np
import matplotlib.pylab as plt
import torchvision.datasets as datasets
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


# load mnist dataset
idx = np.random.permutation(60000)[:5000] # Randomly extract 5,000 data from 60,000 data.
mnist_data = datasets.MNIST('~/tmp/mnist', train=True, download=True)
target = mnist_data.targets.numpy()[idx]
target_name = [0,1,2,3,4,5,6,7,8,9]
x_train = mnist_data.data.numpy()[idx]

# load image feature extracted by CNNAE
im_feat = np.load('../output/2_im_feat_10000.npy')[idx]

# PCA
pca = PCA(n_components=3, random_state=41)
PCn = pca.fit_transform(x_train)                                   # PCA process
print('PCA data size: ', PCn.shape)

fig = plt.figure()
ax  = fig.add_subplot(111, projection="3d")
for i in target_name:
    index = np.where(target==i)
    ax.scatter(PCn[index,0], PCn[index,1], PCn[index,2],alpha=0.4)

plt.xlabel("First Component")
plt.ylabel("Second Component")
plt.ylabel("Third Component")
plt.savefig('../output/3_mnist_pca.png')
plt.close()

# TSNE
tsne = TSNE(n_components=2)
im_tsne = tsne.fit_transform(x_train)                                 # TSNE process
print('t-SNE data size: ', im_tsne.shape)

fig = plt.figure()
for i in target_name:
    index = np.where(target==i)
    plt.scatter(im_tsne[index,0], im_tsne[index,1], label=str(i), alpha=0.5, s=20)
    
plt.legend(loc=3)
plt.xlabel("First Component")
plt.ylabel("Second Component")
plt.savefig('../output/3_mnist_tsne.png')
plt.close()
