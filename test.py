from mini_batch_loader import *
import MyFCN_de
import sys
import time
import State_de
import pixelwise_a3c_de
import os
import torch
import pixelwise_a3c_el
import MyFCN_el
from models import FFDNet
import torch.nn as nn

# _/_/_/ paths _/_/_/
TRAINING_DATA_PATH = "data/training_LOL_eval15.txt"
label_DATA_PATH = "data/label_LOL_eval15.txt"
TESTING_DATA_PATH = "data/training_LOL_eval15.txt"

IMAGE_DIR_PATH = "./"
SAVE_PATH = "./model/ex1_"

# _/_/_/ training parameters _/_/_/
LEARNING_RATE = 0.0005
TRAIN_BATCH_SIZE = 32
TEST_BATCH_SIZE = 1  # must be 1
N_EPISODES = 30000
EPISODE_LEN = 6
SNAPSHOT_EPISODES = 3000
TEST_EPISODES = 3000
GAMMA = 1.05  # discount factor

# noise setting


N_ACTIONS = 27
MOVE_RANGE = 27  # number of actions that move the pixel values. e.g., when MOVE_RANGE=3, there are three actions: pixel_value+=1, +=0, -=1.
CROP_SIZE = 70
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
GPU_ID = 0


def test(loader1,loader2, agent_el, agent_de, fout, model):
    sum_psnr   = 0
    sum_reward = 0
    test_data_size = MiniBatchLoader.count_paths(TESTING_DATA_PATH)
    current_state = State_de.State_de((TEST_BATCH_SIZE, 1, CROP_SIZE, CROP_SIZE), MOVE_RANGE, model)
    for i in range(0, test_data_size, TEST_BATCH_SIZE):
        raw_x = loader1.load_testing_data(np.array(range(i, i+TEST_BATCH_SIZE)))
        label = loader2.load_testing_data(np.array(range(i, i+TEST_BATCH_SIZE)))
        #raw_n = np.random.normal(MEAN,SIGMA,raw_x.shape).astype(raw_x.dtype)/255
        current_state.reset(raw_x)
        reward = np.zeros(raw_x.shape, raw_x.dtype)*255

        for t in range(0, EPISODE_LEN):
            previous_image = current_state.image.copy()
            action_el = agent_el.act(current_state.image)
            current_state.step_el(action_el)
            if t > 4:
                action_de = agent_de.act(current_state.image)
                current_state.step_de(action_de)
            reward = np.square(label - previous_image)*255 - np.square(label - current_state.image)*255
            sum_reward += np.mean(reward)*np.power(GAMMA,t)

        agent_de.stop_episode()

        I = np.maximum(0,label)
        I = np.minimum(1,I)
        p = np.maximum(0,current_state.image)
        p = np.minimum(1,p)
        I = (I*255).astype(np.uint8)
        p = (p*255).astype(np.uint8)
        sum_psnr += cv2.PSNR(p, I)
        p = np.squeeze(p, axis=0)
        p = np.transpose(p, (1, 2, 0))
        cv2.imwrite('./result_ex2/' + str(i) + '_output.png', p)
    print("test total reward {a}, PSNR {b}".format(a=sum_reward*255/test_data_size, b=sum_psnr/test_data_size))
    fout.write("test total reward {a}, PSNR {b}\n".format(a=sum_reward*255/test_data_size, b=sum_psnr/test_data_size))
    sys.stdout.flush()


def main(fout):
    #_/_/_/ load dataset _/_/_/
    mini_batch_loader = MiniBatchLoader(
        TRAINING_DATA_PATH,
        TRAINING_DATA_PATH,
        IMAGE_DIR_PATH,
        CROP_SIZE)
    mini_batch_loader_label = MiniBatchLoader(
        label_DATA_PATH,
        label_DATA_PATH,
        IMAGE_DIR_PATH,
        CROP_SIZE)

    pixelwise_a3c_el.chainer.cuda.get_device_from_id(GPU_ID).use()

    # load ffdnet
    in_ch = 3
    model_fn = 'FFDNet_models/net_rgb.pth'
    # Absolute path to model file
    model_fn = os.path.join(os.path.abspath(os.path.dirname(__file__)), \
                            model_fn)
    # Create model
    print('Loading model ...\n')
    net = FFDNet(num_input_channels=in_ch)

    # Load saved weights

    state_dict = torch.load(model_fn)
    device_ids = [GPU_ID]
    model = nn.DataParallel(net, device_ids=device_ids).cuda()
    # model = net.cuda()
    model.load_state_dict(state_dict)
    model.eval()
    current_state = State_de.State_de((TRAIN_BATCH_SIZE, 1, CROP_SIZE, CROP_SIZE), MOVE_RANGE, model)

    # load myfcn model
    model_el = MyFCN_el.MyFcn(N_ACTIONS)

    # _/_/_/ setup _/_/_/
    optimizer_el = pixelwise_a3c_el.chainer.optimizers.Adam(alpha=LEARNING_RATE)
    optimizer_el.setup(model_el)

    agent_el = pixelwise_a3c_el.PixelWiseA3C(model_el, optimizer_el, EPISODE_LEN, GAMMA)
    pixelwise_a3c_el.chainer.serializers.load_npz('./pretrained/model.npz', agent_el.model)
    agent_el.act_deterministically = True
    agent_el.model.to_gpu()

    # load myfcn model for denoising
    model_de = MyFCN_de.MyFcn_denoise(2)

    # _/_/_/ setup _/_/_/

    optimizer_de = pixelwise_a3c_de.chainer.optimizers.Adam(alpha=LEARNING_RATE)
    optimizer_de.setup(model_de)

    agent_de = pixelwise_a3c_de.PixelWiseA3C(model_de, optimizer_de, EPISODE_LEN, GAMMA)
    pixelwise_a3c_de.chainer.serializers.load_npz('./pretrained/init_denoising.npz', agent_de.model)
    agent_de.act_deterministically = True
    agent_de.model.to_gpu()

    test(mini_batch_loader, mini_batch_loader_label, agent_el, agent_de, fout, model)


if __name__ == '__main__':
    try:
        fout = open('testlog_45_[-13,13].txt', "w")
        start = time.time()
        main(fout)
        end = time.time()
        print("{s}[s]".format(s=end - start))
        print("{s}[m]".format(s=(end - start)/60))
        print("{s}[h]".format(s=(end - start)/60/60))
        fout.write("{s}[s]\n".format(s=end - start))
        fout.write("{s}[m]\n".format(s=(end - start)/60))
        fout.write("{s}[h]\n".format(s=(end - start)/60/60))
        fout.close()
    except Exception as error:
        print(error.message)
