import argparse

def parser():
    parser = argparse.ArgumentParser()

    parser.add_argument('--device', type=str, default='cuda', help='')#

    parser.add_argument('--log_dir', type=str, default=None, help='')#
    parser.add_argument('--model', type=str, default='pointcloud_diffusion', help='')#
    parser.add_argument('--data_path', type=str, default=None, help='')#


    parser.add_argument('--learning_rate', default=0.0001, type=float, help='learning rate in training')#
    parser.add_argument('--epochs', default=100, type=int, help='learning rate in training')#
    parser.add_argument('--train_split', default=0.8, type=float, help='learning rate in training')#
    parser.add_argument('--valid_split', default=0.1, type=float, help='learning rate in training')#
    parser.add_argument('--batch_size', default=256, type=int, help='learning rate in training')#
    parser.add_argument('--shuffle', default=True, type=bool, help='learning rate in training')#
    parser.add_argument('--num_workers', default=2, type=int, help='learning rate in training')#
    parser.add_argument('--time_step', default=500, type=int, help='learning rate in training')#
    

    opt = parser.parse_args()
    return opt