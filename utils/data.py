import numpy as np
from PIL import Image, ImageDraw
import random


def normalization(data, indataRange, outdataRange):
    '''
    Function to normalize a numpy array within a specified range
    Args:
        data (np.array): Data array
        indataRange (float list):  List of maximum and minimum values of original data, e.g. indataRange=[0.0, 255.0].
        outdataRange (float list): List of maximum and minimum values of output data, e.g. indataRange=[0.0, 1.0].
    Return:
        data (np.array): Normalized data array

    data = (data-indataRange[0])/(indataRange[1]-indataRange[0]) * (outdataRange[1] - outdataRange[0]) + outdataRange[0]
    '''

    denominator = indataRange[1] - indataRange[0]
    if denominator ==0:
        denominator=1

    compressed = (data - indataRange[0]) / denominator
    data = compressed * (outdataRange[1] - outdataRange[0]) + outdataRange[0]
    
    return data



def getLissajous(total_step, num_cycle, x_mag, y_mag, delta, dtype=np.float32):
    '''
    Function to generate a Lissajous curve
    Reference URL: http://www.ne.jp/asahi/tokyo/nkgw/www_2/gakusyu/rikigaku/Lissajous/Lissajous_kaisetu/Lissajous_kaisetu_1.html
    Args:
        total_step (int): Sequence length of Lissajous curve.
        num_cycle  (int): Iteration of the Lissajous curve.
        x_mag    (float): Angular frequency of the x direction.
        y_mag    (float): Angular frequency of the y direction.
        delta    (float): Initial phase of the y direction
    Return:
        data (np.array): Array of Lissajous curves. data shape is [total_step, 2]
    '''

    t = np.linspace(-np.pi,np.pi,total_step)
    x = np.cos(x_mag * t)
    y = np.cos(y_mag * t + delta) 
    
    # print(np.shape([x,y]))

    return np.c_[x,y].astype(dtype)



def getLissajousMovie(total_step, num_cycle, x_mag, y_mag, delta, imsize, circle_r, color, vmin=-0.9, vmax=0.9):
    '''
    Function to generate a Lissajous curve with movie
    Args:
        total_step (int): Sequence length of Lissajous curve.
        num_cycle  (int): Iteration of the Lissajous curve.
        x_mag    (float): Angular frequency of the x direction.
        y_mag    (float): Angular frequency of the y direction.
        delta    (float): Initial phase of the y direction
        imsize     (int): Pixel size of the movie
        circle_r   (int): Radius of the circle moving in the movie.
        color     (list): Color of the circle. Specify color in RGB list, e.g. red is [255,0,0].
        vmin     (float): Minimum value of output data
        vmax     (float): Maximum value of output data
    Return:
        data (np.array): Array of movie and curve. movie shape is [total_step, imsize, imsize, 3], curve shape is [total_step, 2].
    '''

    #Use the normalization function.
    xy = getLissajous( total_step, num_cycle, x_mag, y_mag, delta )
    x, y = np.split(xy, indices_or_sections=2, axis=-1)

    _color = tuple((np.array(color)).astype(np.uint8))

    imgs = []
    for _t in range(total_step):
        # xy position in the image
        _x = np.cos(x_mag * _t)
        _y = np.cos(y_mag * _t + delta) 
        img = Image.new("RGB", (imsize, imsize), "white")
        draw = ImageDraw.Draw(img)
        # Draws a circle with a specified radius
        draw.ellipse((_x,_y,circle_r,circle_r),fill=_color)
        imgs.append(np.expand_dims(np.asarray(img), 0))
    imgs = np.vstack(imgs)
    
    ### normalization
    imgs = normalization(imgs.astype(np.float32), [0, 255], [vmin, vmax])
    seq = normalization(np.c_[x,y].astype(np.float32), [-1.0, 1.0], [vmin, vmax])
    return imgs, seq



def deprocess_img(data, vmin=-0.9, vmax=0.9):
    '''
    Function to normalize a numpy array within a specified range
    Args:
        data (np.array): Data array
        vmin (float):  Minimum value of input data
        vmax (float):  Maximum value of input data
    Return:
        data (np.array with np.uint8): Normalized data array from 0 to 255.
    '''

    #Use the normalization function.
    data = normalization(data, [vmin, vmax], [0, 255])
    # print(data)

    return data.astype(np.uint8)



def get_batch( x, BATCH_SIZE):
    '''
    Shuffle the input data and extract data specified by batch size.
    ''' 
    inds = random.sample(range(60000), k=BATCH_SIZE)
    return x[inds]  

def tensor2numpy(x):
    '''
    Convert tensor to numpy array.
    '''
    return x.to('cpu').detach().numpy().copy()

