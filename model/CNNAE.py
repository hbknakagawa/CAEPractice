import torch
import torch.nn as nn


class CNNAE(nn.Module):
    '''
    An implementation of convolutional autoencoder for mnist
    
    Args:
        feat_size (int): Size of extracted image features
    '''
    def __init__(self, feat_size=10):
        super(CNNAE, self).__init__()
        self.feat_size = feat_size
        self.activation = torch.tanh # Set the activation function. Here tanh is recommended.

        # Implement each layer to yield the output shape specified in the comments.
        # encoder
        self.conv1 = nn.Conv2d(1,8,15)     #Output Shape [8, 14, 14]
        self.conv2 = nn.Conv2d(8,16,8)       #Output Shape [16, 7, 7]
        self.conv3 = nn.Conv2d(16,32,5)       #Output Shape [32, 3, 3]
        self.l4 = nn.Linear(in_features=32*3*3, out_features=100)   #Output Shape [100]
        self.l5 = nn.Linear(in_features=100, out_features=feat_size) #Output Shape [feat_size]

        # decoder
        self.l6 = nn.Linear(in_features=feat_size, out_features=100)          #Output Shape [100]
        self.l7 = nn.Linear(in_features=100, out_features=32*3*3)          #Output Shape [288]
        self.conv8 = nn.ConvTranspose2d(32,16,5)        #Output Shape [16, 7, 7]
        self.conv9 = nn.ConvTranspose2d(16,8,8)       #Output Shape [8, 14, 14]
        self.conv10 = nn.ConvTranspose2d(8,1,15)       #Output Shape [1, 28, 28]

    
    def encoder(self, x):
        '''
        Extract ``feat_size``-dimensional image features from the input image using nn.Conv2d and nn.Linear.
        The activation function also should be set.
        '''
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = x.reshape(-1,32*3*3)
        x = self.l4(x)
        hid = self.l5(x)
        return hid

    def decoder(self, hid):
        '''
        Reconstruct an image from ``feat_size``-dimensional image features using nn.ConvTranspose2d and nn.Linear.
        The activation function also should be set.
        '''
        x = self.l6(hid)
        x = self.l7(x)
        x = x.reshape(-1,32,3,3)
        x = self.conv8(x)
        x = self.conv9(x)
        x = self.conv10(x)
        return x

    def forward(self, im):
        '''
        Declare foward process
        '''
        return self.decoder( self.encoder(im) )
    


if __name__ == "__main__":
    from torchinfo import summary
    
    batch_size = 50
    model = CNNAE()
    summary(model, input_size=(batch_size, 1, 28, 28 ))
