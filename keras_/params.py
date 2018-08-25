import argparse
import distutils.util

parser = argparse.ArgumentParser()
arg = parser.add_argument
arg('--gpu', default="0")
arg('--fold', type=int, default=None)
arg('--n_folds', type=int, default=5)
arg('--folds_source')
arg('--clr')
arg('--seed', type=int, default=80)
arg('--test_size_float', type=float, default=0.1)
arg('--epochs', type=int, default=30)
arg('--img_height', type=int, default=101)
arg('--img_width', type=int, default=101)
arg('--out_height', type=int, default=101)
arg('--out_width', type=int, default=101)
arg('--input_width', type=int, default=96)
arg('--input_height', type=int, default=96)
arg('--use_crop', type=distutils.util.strtobool, default='true')
arg('--learning_rate', type=float, default=0.0001)
arg('--batch_size', type=int, default=1)
arg('--dataset_dir', default='../data')
arg('--models_dir', default='models')
arg('--weights')
arg('--loss_function', default='bce_dice')
arg('--freeze_till_layer', default='input_1')
arg('--show_summary', type=bool, default=False)
arg('--network', default='simple_unet')
arg('--net_alias', default='')
arg('--preprocessing_function', default='caffe')

arg('--pred_mask_dir')
arg('--pred_tta')
arg('--pred_batch_size', default=1)
arg('--test_data_dir', default='data/train/images_fail')
arg('--pred_threads', type=int, default=1)
arg('--submissions_dir', default='submissions')
arg('--pred_sample_csv', default='input/sample_submission.csv')
arg('--predict_on_val', type=bool, default=False)
arg('--stacked_channels', type=int, default=0)
arg('--stacked_channels_dir', default=None)

# Dir names
arg('--train_data_dir_name', default='train/images')
arg('--val_data_dir_name', default='train/images')
arg('--train_mask_dir_name', default='train/masks')
arg('--val_mask_dir_name', default='train/masks')
arg('--test_mask_dir', default='test_masks')

arg('--dirs_to_ensemble', nargs='+')
arg('--ensembling_strategy', default='average')
arg('--folds_dir')
arg('--ensembling_dir')
arg('--ensembling_cpu_threads', type=int, default=6)

args = parser.parse_args()