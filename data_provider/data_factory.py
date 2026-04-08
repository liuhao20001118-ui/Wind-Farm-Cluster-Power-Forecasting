
from torch.utils.data import DataLoader
from data_provider.data_loader import Dataset_wind_data_graph, Dataset_ERA5_Wind_Graph, collate_graph

data_dict = {
    'WindGraph': Dataset_wind_data_graph,
    'ERA5WindGraph': Dataset_ERA5_Wind_Graph
}

def data_provider(args, flag):
    Data = data_dict[args.data]
    timeenc = 0 if args.embed != 'timeF' else 1

    shuffle_flag = False if flag == 'test' else True
    drop_last = True
    batch_size = args.batch_size
    freq = args.freq

    # 将通用参数提取出来 ---
    common_args = {
        'root_path': args.root_path,
        'data_path': args.data_path,
        'flag': flag,
        'size': [args.seq_len, args.label_len, args.pred_len],
        'features': args.features,
        'target': args.target,
        'timeenc': timeenc,
        'freq': freq,
        'data_step': args.data_step,
        'min_num_nodes': args.min_num_nodes,
        'c_out': args.c_out  # <-- 传递 c_out
    }

    if args.data == 'ERA5WindGraph':
        data_set_args = {
            **common_args, # 合并通用参数
            'n_closest': args.n_closest,
            'era5_data_path': args.era5_data_path,
            'era5_center_lat': args.era5_center_lat,
            'era5_center_lon': args.era5_center_lon,
            'enc_in': args.enc_in,
        }
        data_set = Data(**data_set_args)
        collate_fn_to_use = collate_graph
    else:  # WindGraph
        data_set_args = {
            **common_args, # 合并通用参数
            'all_stations': args.all_stations,
            'n_closest': args.n_closest,
        }
        data_set = Data(**data_set_args)
        collate_fn_to_use = collate_graph

    print(flag, len(data_set))

    data_loader = DataLoader(
        data_set,
        batch_size=batch_size,
        shuffle=shuffle_flag,
        num_workers=args.num_workers,
        drop_last=drop_last,
        collate_fn=collate_fn_to_use
    )
    return data_set, data_loader