import argparse
import glob
from PIL import Image
import PIL
import random
from utils import *

# the pixel value range is '0-255'(uint8 ) of training data

# macro
DATA_AUG_TIMES = 1  # transform a sample to a different sample for DATA_AUG_TIMES times

parser = argparse.ArgumentParser(description='')
parser.add_argument('--src_dir', dest='src_dir', default='./data/Train400', help='dir of data')
parser.add_argument('--save_dir', dest='save_dir', default='./data', help='dir of patches')
parser.add_argument('--patch_size', dest='pat_size', type=int, default=40, help='patch size')
parser.add_argument('--stride', dest='stride', type=int, default=11, help='stride')
parser.add_argument('--step', dest='step', type=int, default=0, help='step')
parser.add_argument('--batch_size', dest='bat_size', type=int, default=128, help='batch size')
# check output arguments
parser.add_argument('--from_file', dest='from_file', default="./data/img_clean_pats.npy", help='get pic from file')
parser.add_argument('--num_pic', dest='num_pic', type=int, default=10, help='number of pic to pick')
args = parser.parse_args()

try:
    xrange
except:
    xrange = range

def generate_patches_train(isDebug=False):
    global DATA_AUG_TIMES
    count = 0
    filepaths = glob.glob(args.src_dir + '/*.png')
    if isDebug:
        filepaths = filepaths[:10]
    print("number of training data %d" % len(filepaths))

    ######---------------
    count = 0

    im_h = im_w = 128
    for x in range(0 + args.step, im_h - args.pat_size, args.stride):
        for y in range(0 + args.step, im_w - args.pat_size, args.stride):
            count += 1
    count = count * 16000
    print(count)

    inputs_list = np.zeros((count, args.pat_size, args.pat_size, 1))
    labels_list = np.zeros((count, args.pat_size, args.pat_size, 1))

    idx = 0
    # generate patches
    for i in range(4000, 4003):
        if (i%500==0):
            print("Processing image #" + str(i))
        # input_path = get_image_path(True, 64, i)
        # label_path = get_image_path(True, 128, i)
        # sub_input, sub_label = preprocess(input_path, label_path)
        # sub_input = sub_input.reshape([args.pat_size, 128, 1])
        # sub_label = sub_label.reshape([args.pat_size,128 , 1])
        # labels_list[idx] = sub_label
        # inputs_list[idx] = sub_input
        # idx += 1

        input_path = get_image_path(True, 64, i)
        label_path = get_image_path(True, 128, i)
        input_img, label_img = preprocess(input_path, label_path)
        im_h, im_w = input_img.shape
        for x in range(0 + args.step, im_h - args.pat_size, args.stride):
            for y in range(0 + args.step, im_w - args.pat_size, args.stride):
                sub_input = input_img[x:x + args.pat_size, y:y + args.pat_size]
                sub_label = label_img[x:x + args.pat_size, y:y + args.pat_size]
                sub_input = sub_input.reshape([args.pat_size, args.pat_size, 1])
                sub_label = sub_label.reshape([args.pat_size, args.pat_size, 1])

                inputs_list[idx] = sub_input
                labels_list[idx] = sub_label
                idx += 1

    print("idx==" + str(idx))
    if not os.path.exists(args.save_dir):
        os.mkdir(args.save_dir)

    #"img_clean_pats"
    np.save(os.path.join(args.save_dir, "input_data_pats"), inputs_list)
    np.save(os.path.join(args.save_dir, "label_data_pats"), labels_list)
    print("size of inputs tensor = " + str(labels_list.shape))


if __name__ == '__main__':
    generate_patches_train()
