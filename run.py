import argparse
import torch
from exp.exp_main import Exp_Main
import random
import os
import numpy as np


def main():
    fix_seed = 2022
    random.seed(fix_seed)
    torch.manual_seed(fix_seed)
    np.random.seed(fix_seed)
    parser = argparse.ArgumentParser(description='Wind Power Forecasting')
    parser.add_argument('--is_training', type=int, default=1, help='status')
    parser.add_argument('--model_id', type=str, default='GatedFusion_v3', help='model id for saving')
    parser.add_argument('--model', type=str, required=False, default='GraphTransformer', help='model name')
    parser.add_argument('--test_dir', type=str, default='./test_results', help='Base dir to save test results')
    parser.add_argument('--data', type=str, required=False, default='ERA5WindGraph',
                        help='dataset type: ERA5WindGraph or WindGraph')
    parser.add_argument('--root_path', type=str, default='./dataset_example/WindData/dataset/',
                        help='root path of the wind data file')
    parser.add_argument('--data_path', type=str, default='wind_data.csv', help='wind data file name')
    parser.add_argument('--era5_data_path', type=str, default='./dataset_example/WindData/era5/combined_data.csv',
                        help='Full or relative path to the ERA5 data file')
    parser.add_argument('--features', type=str, default='MS', help='forecasting task, options:[M, S, MS]')
    parser.add_argument('--target', type=str, default='station1', help='target feature')
    parser.add_argument('--freq', type=str, default='h', help='freq for time features encoding')
    parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location of model checkpoints')
    parser.add_argument('--checkpoint_flag', type=int, default=1, help='Whether to checkpoint or not')
    parser.add_argument('--data_step', type=int, default=1, help='Only use every nth point.')
    parser.add_argument('--min_num_nodes', type=int, default=2, help='Minimum number of nodes in a graph')
    parser.add_argument('--n_closest', type=int, default=None,
                        help='number of closest nodes for graph connectivity, None --> complete graph')
    parser.add_argument('--all_stations', type=int, default=1,
                        help='Whether to use all stations or just target for non-spatial models.')
    parser.add_argument('--era5_center_lat', type=float, default=52.25, help='Latitude of the central ERA5 grid point')
    parser.add_argument('--era5_center_lon', type=float, default=-1.0, help='Longitude of the central ERA5 grid point')
    parser.add_argument('--era5_in', type=int, default=6, help='Number of ERA5 input features')
    parser.add_argument('--seq_len', type=int, default=168, help='input sequence length')
    parser.add_argument('--label_len', type=int, default=24, help='start token length')
    parser.add_argument('--pred_len', type=int, default=24, help='prediction sequence length')
    parser.add_argument('--enc_in', type=int, default=7, help='encoder input size (wind farm features)')
    parser.add_argument('--dec_in', type=int, default=7, help='decoder input size (wind farm features)')
    parser.add_argument('--c_out', type=int, default=2, help='output size (e.g., wind speed and power)')
    parser.add_argument('--d_model', type=int, default=256, help='dimension of model')
    parser.add_argument('--n_heads', type=int, default=8, help='num of heads')
    parser.add_argument('--e_layers', type=int, default=2, help='num of encoder layers')
    parser.add_argument('--gnn_layers', type=int, default=2, help='Number of GNN layers')
    parser.add_argument('--d_ff', type=int, default=1024, help='dimension of fcn')
    parser.add_argument('--dropout', type=float, default=0.1, help='dropout')
    parser.add_argument('--embed', type=str, default='timeF', help='time features encoding')
    parser.add_argument('--activation', type=str, default='gelu', help='activation')
    parser.add_argument('--output_attention', action='store_true', help='whether to output attention in encoder')
    parser.add_argument('--kernel_size', type=int, default=3, help='conv kernel size')
    parser.add_argument('--num_workers', type=int, default=0, help='data loader num workers')
    parser.add_argument('--itr', type=int, default=1, help='experiments times')
    parser.add_argument('--train_epochs', type=int, default=100, help='train epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size of train input data')
    parser.add_argument('--patience', type=int, default=10, help='early stopping patience')
    parser.add_argument('--learning_rate', type=float, default=0.0001, help='optimizer learning rate')
    parser.add_argument('--lr_decay_rate', type=float, default=0.85, help='Rate for which to decay lr with')
    parser.add_argument('--lradj', type=str, default='type1', help='adjust learning rate')
    parser.add_argument('--physics_decay_rate', type=float, default=0.85, 
                        help='Decay rate for physics weight per epoch (0.0 to 1.0)')
    parser.add_argument('--physics_weight', type=float, default=0.2, help='weight for physics loss')
    parser.add_argument('--v_rated', type=float, default=12.5, help='Rated wind speed of the turbine (m/s)')

    # --- GPU ---
    parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
    parser.add_argument('--gpu', type=int, default=0, help='gpu')
    parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)
    parser.add_argument('--devices', type=str, default='0,1,2,3', help='device ids of multiple gpus')

    args = parser.parse_args()
    if args.features == 'S':
        assert (np.array([args.c_out, args.enc_in, args.dec_in]) == 1).all(), "c_out, enc_in and dec_in should be 1 for univariate"
    args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False
    if args.use_gpu and args.use_multi_gpu:
        args.devices = args.devices.replace(' ', '')
        device_ids = args.devices.split(',')
        args.device_ids = [int(id_) for id_ in device_ids]
        args.gpu = args.device_ids[0]
    print('Args in experiment:')
    print(args)
    if args.is_training:
        for ii in range(args.itr):
            setting = '{}_{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_df{}_bs{}_do{:.2f}_ph{}_dec{}_{}'.format(
                args.model_id, args.model, args.data, args.features,
                args.seq_len, args.label_len, args.pred_len,
                args.d_model, args.d_ff, args.batch_size, args.dropout,
                args.physics_weight, args.physics_decay_rate, ii)
            exp = Exp_Main(args)
            print(f"exp class: {type(exp).__name__}") 
            print('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
            exp.train(setting)
            print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
            exp.test(setting, base_dir=args.test_dir)
            torch.cuda.empty_cache()
    else:
        ii = 0
        setting = '{}_{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_df{}_bs{}_do{:.2f}_ph{}_dec{}_{}'.format(
                args.model_id, args.model, args.data, args.features,
                args.seq_len, args.label_len, args.pred_len,
                args.d_model, args.d_ff, args.batch_size, args.dropout,
                args.physics_weight, args.physics_decay_rate, ii)
        exp = Exp_Main(args)
        print(f"exp class: {type(exp).__name__}") 
        print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
        exp.test(setting, base_dir=args.test_dir)
        torch.cuda.empty_cache()
if __name__ == "__main__":
    main()
