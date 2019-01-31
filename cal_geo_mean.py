import os, sys
import argparse
import re
import matplotlib.pyplot as plt
import matplotlib.ticker as plticker
import numpy as np
import draw as data

parser = argparse.ArgumentParser(description='GPU measure')
parser.add_argument('-d', '--data-dir', default='.', type=str,
                    help='sssss')
args = parser.parse_args()
data_dir = args.data_dir
OUTPUT_PATH = 'pictures_out'

gpus = ['p100', 'v100', 'gtx2080ti']
algos = ['ipc_gemm', 'winograd', 'fft_tile']
nets = ['alexnet', 'resnet', 'vggnet', 'googlenet']
p100_coreF = ['544', '683', '810', '936', '1063', '1202', '1328']
v100_coreF = ['510', '652', '802', '945', '1087', '1237', '1380']
gtx2080ti_coreF = ['950', '1150', '1350', '1550', '1750', '2050']
gpu_coreFs = [p100_coreF, v100_coreF, gtx2080ti_coreF]
p100_memF = ['715']
v100_memF = ['877']
gtx2080ti_memF = ['5800', '6300', '6800', '7300']
gpu_memFs = [p100_memF, v100_memF, gtx2080ti_memF]

gpu_index = {
    'p100': 0,
    'v100': 1,
    'gtx2080ti': 2    
}



batch_sizes = ['16', '32', '64', '128']
#==================================
auto_alexnet_batch_sizes = ['128', '256', '512', '1024']
auto_resnet_batch_sizes = ['16', '32']
auto_vggnet_batch_sizes = ['16', '32', '64', '128']
auto_googlenet_batch_sizes = ['16', '32', '64', '128']
auto_batch_sizes_list = [auto_alexnet_batch_sizes, auto_resnet_batch_sizes,
                        auto_vggnet_batch_sizes, auto_googlenet_batch_sizes]
#=====================================
ipc_gemm_alexnet_batch_sizes = ['128', '256', '512', '1024']
ipc_gemm_resnet_batch_sizes = ['16', '32']
ipc_gemm_vggnet_batch_sizes = ['16', '32', '64']
ipc_gemm_googlenet_batch_sizes = ['16', '32', '64', '128']
ipc_gemm_batch_sizes_list = [ipc_gemm_alexnet_batch_sizes, ipc_gemm_resnet_batch_sizes, 
                        ipc_gemm_vggnet_batch_sizes, ipc_gemm_googlenet_batch_sizes]
ipc_gemm_nets = ['alexnet', 'resnet', 'vggnet', 'googlenet']
#=======================================
winograd_nonfused_alexnet_batch_sizes = ['128', '256']
winograd_nonfused_resnet_batch_sizes = ['16', '32']
winograd_nonfused_vggnet_batch_sizes = ['16']
winograd_nonfused_googlenet_batch_sizes = ['16', '32', '64']
winograd_nonfused_batch_sizes_list = [winograd_nonfused_alexnet_batch_sizes, winograd_nonfused_resnet_batch_sizes, 
                        winograd_nonfused_vggnet_batch_sizes, winograd_nonfused_googlenet_batch_sizes]
winograd_nets = ['alexnet', 'resnet', 'vggnet', 'googlenet']
#========================================
fft_tile_alexnet_batch_sizes = ['128']
fft_tile_resnet_batch_sizes = [  ]
fft_tile_vggnet_batch_sizes = [  ]
fft_tile_googlenet_batch_sizes = ['16', '32']
fft_tile_batch_sizes_list = [fft_tile_alexnet_batch_sizes, fft_tile_googlenet_batch_sizes]
fft_tile_nets = ['alexnet', 'googlenet']
#=========================================

algo_nets = [ipc_gemm_nets, winograd_nets, fft_tile_nets]
algo_batch_sizes_list = [ipc_gemm_batch_sizes_list, winograd_nonfused_batch_sizes_list, fft_tile_batch_sizes_list]



# ====================***************************************************************===========================
#=====================================
gtx_ipc_gemm_alexnet_batch_sizes = ['64', '128', '256', '512']
gtx_ipc_gemm_resnet_batch_sizes = ['8', '16']
gtx_ipc_gemm_vggnet_batch_sizes = ['16', '32', '64']
gtx_ipc_gemm_googlenet_batch_sizes = ['16', '32', '64']
gtx_ipc_gemm_batch_sizes_list = [gtx_ipc_gemm_alexnet_batch_sizes, gtx_ipc_gemm_resnet_batch_sizes, 
                        gtx_ipc_gemm_vggnet_batch_sizes, gtx_ipc_gemm_googlenet_batch_sizes]
gtx_ipc_gemm_nets = ['alexnet', 'resnet', 'vggnet', 'googlenet']
#=======================================
gtx_winograd_nonfused_alexnet_batch_sizes = ['64', '128', '256']
gtx_winograd_nonfused_resnet_batch_sizes = ['8', '16']
gtx_winograd_nonfused_vggnet_batch_sizes = [ ]
gtx_winograd_nonfused_googlenet_batch_sizes = ['16', '32', '64']
gtx_winograd_nonfused_batch_sizes_list = [gtx_winograd_nonfused_alexnet_batch_sizes, gtx_winograd_nonfused_resnet_batch_sizes, 
                        gtx_winograd_nonfused_googlenet_batch_sizes]
gtx_winograd_nets = ['alexnet', 'resnet', 'googlenet']
#========================================
gtx_fft_tile_alexnet_batch_sizes = ['64']
gtx_fft_tile_resnet_batch_sizes = [  ]
gtx_fft_tile_vggnet_batch_sizes = [  ]
gtx_fft_tile_googlenet_batch_sizes = ['16']
gtx_fft_tile_batch_sizes_list = [gtx_fft_tile_alexnet_batch_sizes, gtx_fft_tile_googlenet_batch_sizes]
gtx_fft_tile_nets = ['alexnet', 'googlenet']
#=========================================

gtx_algo_nets = [gtx_ipc_gemm_nets, gtx_winograd_nets, gtx_fft_tile_nets]
gtx_algo_batch_sizes_list = [gtx_ipc_gemm_batch_sizes_list, gtx_winograd_nonfused_batch_sizes_list, gtx_fft_tile_batch_sizes_list]
# ====================***************************************************************===========================



variens_list = ['gpu', 'algo', 'framework', 'net', 'batch_size', 'coreF', 'memF']

algo_index = {
    'ipc_gemm': 0,
    'winograd': 1,
    'fft_tile': 2
}


algo_index = {
    'ipc_gemm': 0,
    'winograd': 1,
    'fft_tile': 2
}
#======================================
# plot_colors = ['b', 'g', 'r', 'k', 'c', 'm', 'y']
# plot_colors = ['#2a5caa', '#7fb80e', '#f58220', '#1d953f',
#             '#dea32c', '#00ae9d', '#f15a22', '#8552a1']
plot_colors = ['#66c2a5', '#fc8d62', '#8da0cb', '#a6d854',
            '#e78ac3', '#ffd92f', '#e5c494', '#b3b3b3']
markers = ['o', 'v', 's', '*']
linestyles = ['-', '--', '-.', ':']

size_ticks = 24
size_labels = 26

font_legend = {'family' : 'Times New Roman',
'weight' : 'normal',
'size' : 14,
}
font_ticks = {'family' : 'Times New Roman',
'weight' : 'normal',
'size' : 15,
}
font_labels = {'family' : 'Times New Roman',
'weight' : 'normal',
'size' : 26,
}

fig_x = 15
fig_y = 9
y_axis_interval = 10
y_max_ratio = 1.1

y_intervals = [0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 50, 
                100, 200, 500, 1000, 2000, 5000, 10000, 20000, 50000, 100000, 200000, 500000]

num_intervals_min = 4
num_intervals_max = 10
#======================================
# bar_colors = ['lightcoral', 'burlywood', 'y', 'yellowgreen',
#             'lightgreen', 'lightseagreen', 'lightskyblue', 'mediumpurple']
bar_total_width = 70

bar_colors = ['#0A64A4', '#24577B', '#03406A', '#3E94D1']
hatches = ['-', '+', 'x', '\\', '|', '/', 'O', '.']






def get_fix_gpu_algo_config_net_geo_mean_energy(gpu, algo, net):
    i_algo = algo_index[algo]
    i_gpu = gpu_index[gpu]

    local_algo_batch_sizes_list = algo_batch_sizes_list
    local_algo_nets = algo_nets

    if gpu == 'gtx2080ti':
        local_algo_batch_sizes_list = gtx_algo_batch_sizes_list
        local_algo_nets = gtx_algo_nets
        results = np.array([1, 1, 1, 1, 1, 1], np.dtype('float'))
    else:
        results = np.array([1, 1, 1, 1, 1, 1, 1], np.dtype('float'))

    # n = 0

    # for i_net, net in enumerate():
    #     n += len(local_algo_batch_sizes_list[i_algo][i_net])

    i_net = local_algo_nets[i_algo].index(net)
    n = len(local_algo_batch_sizes_list[i_algo][i_net])
    for i_batch, batch in enumerate(local_algo_batch_sizes_list[i_algo][i_net]):
        energys_respect_coreF = np.array(data.get_energy_respect(gpu, algo, 'caffe', net, batch, gpu_coreFs[i_gpu], gpu_memFs[i_gpu][0]), np.dtype('float'))
        energys_sqrt = np.power(energys_respect_coreF, 1/n)
        results = results * energys_sqrt

    return results












# test = get_gpu_algo_geo_mean_power('v100', 'ipc_gemm')
# print(test)




