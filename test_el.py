from mini_batch_loader import *
from chainer import serializers
from MyFCN_el import *
from chainer import cuda, optimizers, Variable
import sys
import math
import time
import chainerrl
import State
import os
import cv2
from pixelwise_a3c_el import *

# _/_/_/ paths _/_/_/
TRAINING_DATA_PATH = "./data/training_LOL_eval15.txt"
TESTING_DATA_PATH = "./data/training_LOL_eval15.txt"
label_DATA_PATH = "./data/label_LOL_eval15.txt"
IMAGE_DIR_PATH = "./"
SAVE_PATH = "./model/test_"

# _/_/_/ training parameters _/_/_/
LEARNING_RATE = 0.001
TRAIN_BATCH_SIZE = 64
TEST_BATCH_SIZE = 1  # must be 1
N_EPISODES = 30000
EPISODE_LEN = 6
GAMMA = 1.05  # discount factor

# noise setting
MEAN = 0
SIGMA = 45

N_ACTIONS = 27
MOVE_RANGE = 27  # number of actions that move the pixel values. e.g., when MOVE_RANGE=3, there are three actions: pixel_value+=1, +=0, -=1.
CROP_SIZE = 70

GPU_ID = 0


def test(loader, loader2, agent, fout):
    sum_psnr = 0
    sum_reward = 0
    test_data_size = MiniBatchLoader.count_paths(TESTING_DATA_PATH)
    current_state = State.State((TEST_BATCH_SIZE, 1, CROP_SIZE, CROP_SIZE), MOVE_RANGE)
    for i in range(0, test_data_size, TEST_BATCH_SIZE):
        raw_x = loader.load_testing_data(np.array(range(i, i + TEST_BATCH_SIZE)))
        # label = loader2.load_testing_data(np.array(range(i, i+TEST_BATCH_SIZE)))
        current_state.reset(raw_x)
        reward = np.zeros(raw_x.shape, raw_x.dtype) * 255

        for t in range(0, EPISODE_LEN):
            previous_image = current_state.image.copy()
            action = agent.act(current_state.image)
            current_state.step(action)
            # reward = np.square(label - previous_image)*255 - np.square(label - current_state.image)*255
            sum_reward += np.mean(reward) * np.power(GAMMA, t)

        agent.stop_episode()

        # I = np.maximum(0,label)
        # I = np.minimum(1,I)
        p = np.maximum(0, current_state.image)
        p = np.minimum(1, p)
        # I = (I*255+0.5).astype(np.uint8)
        p = (p * 255 + 0.5).astype(np.uint8)
        # sum_psnr += cv2.PSNR(p, I)
        p = np.squeeze(p, axis=0)
        p = np.transpose(p, (1, 2, 0))
        img_path = loader.testing_path_infos[i]
        img_name = img_path.split('/')[2]
        cv2.imwrite('./result_ex2/' + str(i) + '_output.png', p)

    # print("test total reward {a}, PSNR {b}".format(a=sum_reward*255/test_data_size, b=sum_psnr/test_data_size))
    # fout.write("test total reward {a}, PSNR {b}\n".format(a=sum_reward*255/test_data_size, b=sum_psnr/test_data_size))
    sys.stdout.flush()


def main(fout):
    # _/_/_/ load dataset _/_/_/
    mini_batch_loader = MiniBatchLoader(
        TRAINING_DATA_PATH,
        TESTING_DATA_PATH,
        IMAGE_DIR_PATH,
        CROP_SIZE)

    mini_batch_loader_label = MiniBatchLoader(
        label_DATA_PATH,
        label_DATA_PATH,
        IMAGE_DIR_PATH,
        CROP_SIZE)

    chainer.cuda.get_device_from_id(GPU_ID).use()


    current_state = State.State((TRAIN_BATCH_SIZE, 1, CROP_SIZE, CROP_SIZE), MOVE_RANGE)

    # load myfcn model
    model = MyFcn(N_ACTIONS)

    # _/_/_/ setup _/_/_/
    optimizer = chainer.optimizers.Adam(alpha=LEARNING_RATE)
    optimizer.setup(model)

    agent = PixelWiseA3C(model, optimizer, EPISODE_LEN, GAMMA)
    chainer.serializers.load_npz('./pretrained/model.npz', agent.model)
    agent.act_deterministically = True
    agent.model.to_gpu()

    # _/_/_/ testing _/_/_/
    test(mini_batch_loader, mini_batch_loader_label, agent, fout)


if __name__ == '__main__':
    try:
        fout = open('testlog_45_[-13,13].txt', "w")
        start = time.time()
        main(fout)
        end = time.time()
        print("{s}[s]".format(s=end - start))
        print("{s}[m]".format(s=(end - start) / 60))
        print("{s}[h]".format(s=(end - start) / 60 / 60))
        fout.write("{s}[s]\n".format(s=end - start))
        fout.write("{s}[m]\n".format(s=(end - start) / 60))
        fout.write("{s}[h]\n".format(s=(end - start) / 60 / 60))
        fout.close()
    except Exception as error:
        print(error.message)
