import os
from time import clock

import pandas as pd
import numpy as np
from keras.preprocessing.image import img_to_array
from PIL import Image
from tqdm import tqdm
tqdm
from params import args

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

prediction_dir = args.pred_mask_dir


def dice_coef(y_true, y_pred, smooth=1.0):
    y_true_f = y_true.flatten()
    y_pred_f = y_pred.flatten()
    intersection = np.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (np.sum(y_true_f) + np.sum(y_pred_f) + smooth)


def evaluate(masks_dir, results_dir):
    batch_size = 1
    nbr_test_samples = len(os.listdir(masks_dir))
    df = pd.DataFrame(columns=['name', 'score', 'img_size'])
    mask_filenames = [os.path.join(masks_dir, f) for f in sorted(os.listdir(masks_dir))]
    result_filenames = [os.path.join(results_dir, f) for f in sorted(os.listdir(masks_dir))]
    img_sizes = []
    start_time = clock()
    dices = []
    for i in tqdm(range(int(nbr_test_samples / batch_size))):
        masks = []
        results = []
        mask_filename = None
        img_size = None
        for j in range(batch_size):
            if i * batch_size + j < len(mask_filenames):
                mask_filename = mask_filenames[i * batch_size + j]
                mask = Image.open(mask_filenames[i * batch_size + j])
                result = Image.open(result_filenames[i * batch_size + j])
                img_size = mask.size
                mask = mask.resize((args.input_height, args.input_width), Image.ANTIALIAS)
                result = result.resize((args.input_height, args.input_width), Image.ANTIALIAS)
                img_sizes.append(img_size)
                masks.append(img_to_array(mask)/255)
                results.append(img_to_array(result)/255)
        masks = np.array(masks)
        results = np.array(results)
        batch_dice = dice_coef(masks, results)
        dices.append(batch_dice)
        df.loc[i] = [mask_filename, batch_dice, img_size]
        # print("predicted batch dice {}".format(batch_dice))
    print(np.mean(dices))
    df.to_csv('data/out_data/22_08.csv', index=False)

if __name__ == '__main__':
    evaluate('data/out_data/test_masks', 'data/out_data/output')