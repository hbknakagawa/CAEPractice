from dplm.utils import print_info, print_warn, print_error


### MNIST dataset の読み込みとデータの確認 ###
print_info( "MNIST dataset の読み込みとデータの確認")
import torchvision.transforms as transforms
from torchvision.datasets import MNIST
import matplotlib.animation as animation
from IPython.display import HTML

mnist_data = MNIST('~/tmp/mnist', train=True, download=True)
org_data = mnist_data.data.numpy()
data_min = org_data.min()
data_max = org_data.max()

print( "[Org_data] shape:", org_data.shape )
print( "[Org_data] min={}, max={}".format(data_min, data_max) )
print()



### 課題1-1 normalization関数のテスト ##
print_info( "課題1-1 normalization関数のテスト")
# 自作ライブラリの読み込み
from dplm.utils import normalization

# 正規化後のデータの最大/最小値を指定
target_min = 0.1
target_max = 0.9
norm_data = normalization(org_data, (data_min, data_max), (target_min, target_max) )

print( "[Norm_data] shape:", norm_data.shape )
print( "[Norm_data] min={}, max={}".format(norm_data.min(), norm_data.max()) )
print( "Org_dataとshapeが変わらず、でもデータの最大・最小値が指定値通りであればOK。")
print()



### 課題1-2 getLissajous関数のテスト ###
print_info( "課題1-2 getLissajous関数のテスト")
import numpy as np
import matplotlib.pylab as plt
from dplm.utils import getLissajous

delta_list = [0, np.pi/4, np.pi/2, np.pi*3/4, np.pi]
x_mag_list = [1,1,1,2]
y_mag_list = [1,2,3,3]

loop_ct = 1
for _x, _y in zip( x_mag_list, y_mag_list ):
  for _d in delta_list:
    plt.subplot(4,5,loop_ct)
    data = getLissajous( total_step=120, num_cycle=1, x_mag=_x, y_mag=_y, delta=_d )
    plt.plot(data[:,0], data[:,1])
    plt.xticks(color="None")
    plt.yticks(color="None")
    plt.tick_params(length=0)
    loop_ct += 1

plt.show()
print( "以下リンクの図と同じになればOK。")
print( "http://www.ne.jp/asahi/tokyo/nkgw/www_2/gakusyu/rikigaku/Lissajous/fig-1.gif" )
print()



### 課題1-3 getLissajousMovie関数のテスト ###
print_info( "課題1-3 getLissajousMovie関数のテスト")
from dplm.utils import getLissajousMovie
# 生成されるデータの最大/最小値を指定
# ここでは、あえて練習のために、-0.9から0.9にしています。
vmin, vmax = -0.9, 0.9
img, seq = getLissajousMovie( total_step=120, num_cycle=1, x_mag=1, y_mag=2, delta=np.pi/2, imsize=64, circle_r=3, color=(255, 0, 0), vmin=vmin, vmax=vmax )
print( "img size: {}".format(img.shape) )
print( "img min={}, max={}".format(img.min(), img.max()) )
print( "seq size: {}".format(seq.shape) )

# リサージュ曲線の保存
plt.plot(seq[:,0], c='r')
plt.plot(seq[:,1], c='g')
plt.savefig('../output/01_data_test_Lissajous.png')
plt.close()
print()



### 課題1-4 deprocess_img関数のテスト ###
print_info( "課題1-4 deprocess_img関数のテスト")
from dplm.utils import deprocess_img
print( "Before" )
print( "img min={}, max={}".format(img.min(), img.max()) )
normalized_img = deprocess_img(img, vmin=vmin, vmax=vmax)
print( "After" )
print( "img min={}, max={}".format(normalized_img.min(), normalized_img.max()) )

def video_anime(video):
    fig = plt.figure(figsize=(3,3))  # 表示サイズ指定

    mov = []
    for i in range(len(video)):  # フレームを1枚づつmovにアペンド
        img = plt.imshow(video[i], animated=True)
        plt.axis('off')
        mov.append([img])

    # アニメーション作成        
    anime = animation.ArtistAnimation(fig, mov, interval=50, repeat_delay=1000)
    plt.close()
    return anime

ani = video_anime(normalized_img)
ani.save("../output/01_data_test_LissajousMovie.mp4", writer="ffmpeg")

