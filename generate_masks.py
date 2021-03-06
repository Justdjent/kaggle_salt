"""
Script generates predictions, splitting original images into tiles, and assembling prediction back together
"""
import argparse
# import json
import os
# from prepare_train_val import get_split
from dataset import SaltDataset, unpad
import cv2
import pandas as pd
# from models import UNet16, LinkNet34, UNet11, UNet
from unet_models import TernausNet34, UNet16
import torch
from pathlib import Path
from tqdm import tqdm
import numpy as np
import utils
# import prepare_data
from torch.utils.data import DataLoader
from torch.nn import functional as F
# from prepare_data import (original_height,
#                           original_width,
#                           h_start, w_start
#                           )
# from crop_utils import join_mask
# import crowdai

from validation import convert_bin_coco
from transforms import (ImageOnly,
                        Normalize,
                        RandomCrop,
                        DualCompose,
                        Rescale)

img_transform = DualCompose([
    # RandomCrop([128, 128]),
    # Rescale([256, 256]),
    ImageOnly(Normalize(mean=(0), std=(1)))
])

PAD = (13, 13, 14, 14)

def get_model(model_path, model_type='unet11', problem_type='parts'):
    """

    :param model_path:
    :param model_type: 'UNet', 'UNet16', 'UNet11', 'LinkNet34'
    :param problem_type: 'binary', 'parts', 'instruments'
    :return:
    """
    num_classes = 1

    if model_type == 'UNet16':
        model = UNet16(num_classes=num_classes)
    # elif model_type == 'UNet11':
    #     model = UNet11(num_classes=num_classes)
    # elif model_type == 'LinkNet34':
    #     model = LinkNet34(num_classes=num_classes)
    # elif model_type == 'UNet':
    else:
        model = TernausNet34(num_classes=num_classes)
    if torch.cuda.is_available():
        state = torch.load(str(model_path))
    else:
        state = torch.load(str(model_path), map_location=lambda storage, loc: storage)
    state = {key.replace('module.', ''): value for key, value in state['model'].items()}
    model.load_state_dict(state)

    if torch.cuda.is_available():
        return model.cuda()

    model.eval()

    return model

# Source https://www.kaggle.com/bguberfain/unet-with-depth
def RLenc(img, order='F', format=True):
    """
    img is binary mask image, shape (r,c)
    order is down-then-right, i.e. Fortran
    format determines if the order needs to be preformatted (according to submission rules) or not

    returns run length as an array or string (if format is True)
    """
    bytes = img.reshape(img.shape[0] * img.shape[1], order=order)
    runs = []  ## list of run lengths
    r = 0  ## the current run length
    pos = 1  ## count starts from 1 per WK
    for c in bytes:
        if (c == 0):
            if r != 0:
                runs.append((pos, r))
                pos += r
                r = 0
            pos += 1
        else:
            r += 1

    # if last run is unsaved (i.e. data ends with 1)
    if r != 0:
        runs.append((pos, r))
        pos += r
        r = 0

    if format:
        z = ''

        for rr in runs:
            z += '{} {} '.format(rr[0], rr[1])
        return z[:-1]
    else:
        return runs

def predict(model, from_file_names, batch_size: int, to_path, problem_type):
    loader = DataLoader(
        dataset=SaltDataset(from_file_names, transform=img_transform, mode='predict', problem_type=problem_type),
        shuffle=False,
        batch_size=batch_size,
        num_workers=args.workers,
        pin_memory=torch.cuda.is_available()
    )
    ss = pd.read_csv('data/sample_submission.csv')
    for batch_num, (inputs, paths) in enumerate(tqdm(loader, desc='Predict')):
        inputs = utils.variable(inputs, volatile=True)

        outputs = model(inputs)

        for i, image_name in enumerate(paths):
            # print(outputs[i][0].shape)
            # for j in range(3):
            #     if problem_type == 'binary':
            # factor = prepare_data.binary_factor
            # t_mask = (F.sigmoid(outputs[i, 0]).data.cpu().numpy()).astype(np.uint8)
            # print(outputs)
            # t_mask = (outputs > 0).float()[0][0].data.cpu().numpy()
            t_mask = F.sigmoid(outputs).float()[0][0].data.cpu().numpy()
            # print(t_mask)
            # print(np.median(t_mask))
                # elif problem_type == 'parts':
                #     # factor = prepare_data.parts_factor
                #     factor = 255
                #     t_mask = (F.sigmoid(outputs[i][j]).data.cpu().numpy() * factor).astype(np.uint8)
                #     # t_mask = (outputs[i][j].data.cpu().numpy() * factor).astype(np.uint8)
                # elif problem_type == 'instruments':
                #     factor = prepare_data.instrument_factor
                #     t_mask = (outputs[i].data.cpu().numpy().argmax(axis=0) * factor).astype(np.uint8)

                # h, w = t_mask.shape

                # full_mask = np.zeros((original_height, original_width))
                # full_mask[h_start:h_start + h, w_start:w_start + w] = t_mask

            true_mask = unpad(t_mask, PAD)
            true_mask = true_mask > 0.5
            true_mask = (true_mask * 255).astype(np.uint8)
            # instrument_folder = Path(paths[i]).parent.parent.name

            to_path.mkdir(exist_ok=True, parents=True)
            # print(image_name)
            cv2.imwrite(str(to_path / image_name), true_mask)
            # ann = convert_bin_coco(full_mask, image_name.split('.')[0])
            # anns.append(ann)
            ss.loc[ss.index[ss['id'] == image_name.split('.')[0]].tolist(), 'rle_mask'] = RLenc(true_mask)

    ss.to_csv('data/submission_1.csv', index=False)
    # fp = open("predictions.json", "w")
    # fp.write(json.dumps(anns))
    # fp.close()

# def submit():
#     api_key = "acbb79b92da3e408762784310464ec42"
#     challenge = crowdai.Challenge("crowdAIMappingChallenge", api_key)
#     result = challenge.submit("predictions.json")
#     print(result)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg('--model_path', type=str, default='runs/debug/', help='path to model folder')
    arg('--model_type', type=str, default='UNet16', help='network architecture',
        choices=['UNet', 'UNet11', 'UNet16', 'LinkNet34'])
    arg('--output_path', type=str, help='path to save images', default='data/test/masks')
    arg('--batch-size', type=int, default=1)
    arg('--fold', type=int, default=0, choices=[0, 1, 2, 3, -1], help='-1: all folds')
    arg('--problem_type', type=str, default='parts', choices=['binary', 'parts', 'instruments'])
    arg('--workers', type=int, default=4)

    args = parser.parse_args()

    if args.fold == -1:
        for fold in [0, 1, 2, 3]:
            # _, file_names = get_split(fold)
            file_names = "../mapping-challenge-starter-kit/data/test_images/annotation.json"
            # file_names = os.listdir('data/stage1_test')
            model = get_model(str(args.model_path.joinpath('model_{fold}.pt'.format(fold=fold))),
                              model_type=args.model_type, problem_type=args.problem_type)

            print('num file_names = {}'.format(len(file_names)))

            output_path = Path(args.output_path)
            output_path.mkdir(exist_ok=True, parents=True)

            predict(model, file_names, args.batch_size, output_path, problem_type=args.problem_type)
            # submit()
    else:
        # file_names = os.listdir("../mapping-challenge-starter-kit/data/test_images")
        file_names = os.listdir("data/test/images")
        # file_names = os.listdir('data/stage1_test')
        # _, file_names = get_split(args.fold)
        model = get_model(str(Path(args.model_path).joinpath('best_model_{fold}.pt'.format(fold=args.fold))),
                          model_type=args.model_type, problem_type=args.problem_type)

        print('num file_names = {}'.format(len(file_names)))

        output_path = Path(args.output_path)
        output_path.mkdir(exist_ok=True, parents=True)

        predict(model, file_names, args.batch_size, output_path, problem_type=args.problem_type)
        # submit()
        # imgs = os.listdir('data/stage1_test/')
        # [join_mask(128, img, 'output/mask/', 'output/joined_mask/', '0') for img in imgs]
        # utils.watershed()


