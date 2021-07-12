import chainer
from chainer import Variable
import chainer.links as L
import chainer.functions as F
import numpy as np
import math
import cv2
from chainer.links.caffe import CaffeFunction
import chainerrl
from chainerrl.agents import a3c

class DilatedConvBlock(chainer.Chain):

    def __init__(self, d_factor):
        super(DilatedConvBlock, self).__init__(
            diconv=L.DilatedConvolution2D( in_channels=64, out_channels=64, ksize=3, stride=1, pad=d_factor, dilate=d_factor, nobias=False),
            #bn=L.BatchNormalization(64)
        )

        self.train = True

    def __call__(self, x):
        h = F.relu(self.diconv(x))
        #h = F.relu(self.bn(self.diconv(x)))
        return h


class MyFcn_denoise(chainer.Chain, a3c.A3CModel):
 
    def __init__(self, n_actions):
        w = chainer.initializers.HeNormal()
        #net = CaffeFunction('../initial_weight/zhang_cvpr17_denoise_15_gray.caffemodel')
        super(MyFcn_denoise, self).__init__(
            conv1=L.Convolution2D(3, 64, 3, stride=1, pad=1, nobias=False),
            diconv2=DilatedConvBlock(2),
            diconv3=DilatedConvBlock(3),
            diconv4=DilatedConvBlock(4),
            diconv5_pi=DilatedConvBlock(3),
            diconv6_pi=DilatedConvBlock(2),
            conv7_pi=chainerrl.policies.SoftmaxPolicy(L.Convolution2D(64, n_actions, 3, stride=1, pad=1, nobias=False)),
            diconv5_V=DilatedConvBlock(3),
            diconv6_V=DilatedConvBlock(2),
            conv7_V=L.Convolution2D(64, 1, 3, stride=1, pad=1, nobias=False),
        )
        self.train = True
 
    def pi_and_v(self, x):
        h = F.relu(self.conv1(x))
        h = self.diconv2(h)
        h = self.diconv3(h)
        h = self.diconv4(h)
        h_pi = self.diconv5_pi(h)
        h_pi = self.diconv6_pi(h_pi)
        de = self.conv7_pi(h_pi)
        #pout = np.concatenate((pout_r,pout_g,pout_b), axis=1)
        h_V = self.diconv5_V(h)
        h_V = self.diconv6_V(h_V)
        vout = self.conv7_V(h_V)
       
        return de, vout

# if __name__ == '__main__':
#        train_path = './training_LOL_eval15.txt'
#        test_path = './training_LOL_eval15.txt'
#        image_dir_path = './'
#        crop_size = 70
#        loader = MiniBatchLoader(train_path, test_path, image_dir_path, crop_size)
#        train_data_size = MiniBatchLoader.count_paths(train_path)
#        indices = np.random.permutation(train_data_size)
#        r = indices[0:4]
#        raw_x = loader.load_training_data(r)
#        model = MyFcn_denoise(2)
#        p,v = model(raw_x)