import os
from time import clock

import pandas as pd
import numpy as np
from keras.preprocessing.image import img_to_array
from PIL import Image
from tqdm import tqdm
from params import args
# from losses import mean_iou

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

prediction_dir = args.pred_mask_dir


def dice_coef(y_true, y_pred, smooth=1.0):
    y_true_f = y_true.flatten()
    y_pred_f = y_pred.flatten()
    intersection = np.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (np.sum(y_true_f) + np.sum(y_pred_f) + smooth)


def mean_iou(y_true, y_pred, smooth=1.0):
    prec = []
    for t in np.arange(0.5, 1.0, 0.05):
        y_pred_ = np.float32(y_pred > t)
        # y_true_f = y_true.flatten()
        # y_pred_f = y_pred.flatten()
        # intersection = np.sum(y_true_f * y_pred_f)
        # union = y_true_f + y_pred_f - intersection
        inter = np.sum(y_true * y_pred_)
        union = np.sum(y_true) + np.sum(y_pred_)
        score = inter / (union - inter + smooth)
        if score > 1:
            print('wtf')
        # score, up_opt = K.tf.metrics.mean_iou(y_true, y_pred_, 2)
        # K.get_session().run(K.tf.local_variables_initializer())
        # with K.tf.control_dependencies([up_opt]):
        #     score = K.tf.identity(score)
        prec.append(score)
    return np.mean(prec)


def evaluate(masks_dir, results_dir):
    batch_size = 1
    nbr_test_samples = len(os.listdir(masks_dir))
    df = pd.DataFrame(columns=['name', 'score', 'iou'])
    mask_filenames = [os.path.join(masks_dir, f) for f in sorted(os.listdir(masks_dir))]
    result_filenames = [os.path.join(results_dir, f.replace('nrg', 'nrg')).replace('jpg', 'png') for f in sorted(os.listdir(masks_dir))]
    img_sizes = []
    start_time = clock()
    dices = []
    for i in tqdm(range(int(nbr_test_samples / batch_size))):
        try:
            masks = []
            results = []
            mask_filename = None
            img_size = None
            for j in range(batch_size):
                if i * batch_size + j < len(mask_filenames):

                    mask_filename = mask_filenames[i * batch_size + j]
                    mask = Image.open(mask_filenames[i * batch_size + j])

                    result = Image.open(result_filenames[i * batch_size + j])
                    mask = np.int32(np.array(mask) > 128)
                    result = np.int32(np.array(result) > 128)
                    img_size = mask.size
                    # mask = mask.resize((args.input_height, args.input_width), Image.ANTIALIAS)
                    # result = result.resize((args.input_height, args.input_width), Image.ANTIALIAS)
                    img_sizes.append(img_size)
                    masks.append(mask)
                    if masks[0].max() > 1:
                        print('wtf')
                    results.append(result)

            masks = np.array(masks)
            results = np.array(results)
            batch_dice = dice_coef(masks, results)
            batch_iou = mean_iou(masks, results)
            dices.append(batch_dice)
            df.loc[i] = [mask_filename, batch_dice, batch_iou]
            # print("predicted batch dice {}".format(batch_dice))
        except:
            print('error')
            continue
    print(np.mean(dices))
    df.to_csv(os.path.join(os.path.dirname(results_dir), 'df_tr.csv'), index=False)

if __name__ == '__main__':
    evaluate('../data/train/masks',
             '../data/train_mask_pred')