import os

import pandas as pd
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.losses import binary_crossentropy
from keras.optimizers import Adam

from CyclicLearningRate import CyclicLR
from datasets import build_batch_generator, generate_filenames
from losses import make_loss, dice_coef_clipped, dice_coef, dice_coef_border, mean_iou
from models import make_model
from params import args
from utils import freeze_model, ThreadsafeIter

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

from transforms import (DualCompose,
                        OneOf,
                        ImageOnly,
                        Normalize,
                        HorizontalFlip,
                        VerticalFlip,
                        RandomCrop,
                        RandomRotate90,
                        ShiftScaleRotate,
                        RandomHueSaturationValue,
                        RandomBrightness,
                        RandomContrast)

def main():
    mask_dir = os.path.join(args.dataset_dir, args.train_mask_dir_name)
    val_mask_dir = os.path.join(args.dataset_dir, args.val_mask_dir_name)

    train_data_dir = os.path.join(args.dataset_dir, args.train_data_dir_name)
    val_data_dir = os.path.join(args.dataset_dir, args.val_data_dir_name)

    # mask_dir = 'data/train/masks_fail'
    # val_mask_dir = 'data/val/masks'
    #
    # train_data_dir = 'data/train/images_fail'
    # val_data_dir = 'data/val/images

    if args.net_alias is not None:
        formatted_net_alias = '-{}-'.format(args.net_alias)

    best_model_file =\
        '{}/{}{}loss-{}-fold_{}-{}{:.6f}'.format(args.models_dir, args.network, formatted_net_alias, args.loss_function, args.fold, args.input_width, args.learning_rate) +\
        '-{epoch:d}-{val_loss:0.7f}-{val_dice_coef:0.7f}-{val_dice_coef_clipped:0.7f}.h5'

    model = make_model((None, None, args.stacked_channels + 3))
    freeze_model(model, args.freeze_till_layer)

    if args.weights is None:
        print('No weights passed, training from scratch')
    else:
        print('Loading weights from {}'.format(args.weights))
        model.load_weights(args.weights, by_name=True)

    optimizer = Adam(lr=args.learning_rate)

    if args.show_summary:
        model.summary()

    model.compile(loss=make_loss(args.loss_function),
                  optimizer=optimizer,
                  metrics=[dice_coef_border, dice_coef, binary_crossentropy, dice_coef_clipped, mean_iou])

    crop_size = None

    if args.use_crop:
        crop_size = (args.input_height, args.input_width)
        print('Using crops of shape ({}, {})'.format(args.input_height, args.input_width))
    else:
        print('Using full size images, --use_crop=True to do crops')

    # folds_df = pd.read_csv(os.path.join(args.dataset_dir, args.folds_source))
    # train_ids = generate_filenames(folds_df[folds_df.fold != args.fold]['id'])
    # val_ids = generate_filenames(folds_df[folds_df.fold == args.fold]['id'])
    train_df = pd.read_csv('../data/train_df.csv')
    val_df = pd.read_csv('../data/val_df.csv')
    train_ids = [img + '.png' for img in train_df['id'].values]
    val_ids = [img + '.png' for img in val_df['id'].values]
    # train_ids = os.listdir(train_data_dir)
    # val_ids = os.listdir(val_data_dir)

    print('Training fold #{}, {} in train_ids, {} in val_ids'.format(args.fold, len(train_ids), len(val_ids)))

    train_generator = build_batch_generator(
        train_ids,
        img_dir=train_data_dir,
        batch_size=args.batch_size,
        shuffle=True,
        out_size=(args.out_height, args.out_width),
        crop_size=crop_size,
        mask_dir=mask_dir,
        aug=True
    )

    val_generator = build_batch_generator(
        val_ids,
        img_dir=val_data_dir,
        batch_size=args.batch_size,
        shuffle=False,
        out_size=(args.out_height, args.out_width),
        crop_size=crop_size,
        mask_dir=val_mask_dir,
        aug=False
    )

    best_model = ModelCheckpoint(best_model_file, monitor='val_loss',
                                                  verbose=1,
                                                  save_best_only=False,
                                                  save_weights_only=True)

    callbacks = [best_model, EarlyStopping(patience=45, verbose=10)]
    if args.clr is not None:
        clr_params = args.clr.split(',')
        base_lr = float(clr_params[0])
        max_lr = float(clr_params[1])
        step = int(clr_params[2])
        mode = clr_params[3]
        clr = CyclicLR(base_lr=base_lr, max_lr=max_lr, step_size=step, mode=mode)
        callbacks.append(clr)
    model.fit_generator(
        ThreadsafeIter(train_generator),
        steps_per_epoch=len(train_ids) / args.batch_size + 1,
        epochs=args.epochs,
        validation_data=ThreadsafeIter(val_generator),
        validation_steps=len(val_ids) / args.batch_size + 1,
        callbacks=callbacks,
        max_queue_size=50,
        workers=4)

if __name__ == '__main__':
    main()
