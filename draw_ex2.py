from draw import *
from cal_geo_mean import *


def get_ex2_2080_geo_mean_energy_on_coreF(gpu, algo, net, batch_size):
    gpu = 'gtx2080ti'
    i_algo = algo_index[algo]
    i_gpu = gpu_index[gpu]

    local_algo_batch_sizes_list = gtx_algo_batch_sizes_list
    local_algo_nets = gtx_algo_nets
    results = np.array([1, 1, 1, 1], np.dtype('float'))


    # n = 0

    # for i_net, net in enumerate():
    #     n += len(local_algo_batch_sizes_list[i_algo][i_net])

    # i_net = local_algo_nets[i_algo].index(net)
    n = len(gpu_coreFs[i_gpu])

    for i_coreF, coreF in enumerate(gpu_coreFs[i_gpu]):
        energys_respect_memF = np.array(data.get_energy_respect(gpu, algo, 'caffe', net, batch_size, coreF, gpu_memFs[i_gpu]), np.dtype('float'))
        energys_sqrt = np.power(energys_respect_memF, 1/n)
        results = results * energys_sqrt

    return results



def get_ex2_2080_geo_mean_energy_on_memF(gpu, algo, net, batch_size):
    gpu = 'gtx2080ti'
    i_algo = algo_index[algo]
    i_gpu = gpu_index[gpu]

    local_algo_batch_sizes_list = gtx_algo_batch_sizes_list
    local_algo_nets = gtx_algo_nets
    results = np.array([1, 1, 1, 1, 1, 1], np.dtype('float'))


    # n = 0

    # for i_net, net in enumerate():
    #     n += len(local_algo_batch_sizes_list[i_algo][i_net])

    # i_net = local_algo_nets[i_algo].index(net)
    n = len(gpu_memFs[i_gpu])

    for i_memF, memF in enumerate(gpu_memFs[i_gpu]):
        energys_respect_memF = np.array(data.get_energy_respect(gpu, algo, 'caffe', net, batch_size, gpu_coreFs[i_gpu], memF), np.dtype('float'))
        energys_sqrt = np.power(energys_respect_memF, 1/n)
        results = results * energys_sqrt

    return results







def draw_ex2_p100_v100_coreF(gpu, algo, save=False):

    algo = 'ipc_gemm'
    i_gpu = gpu_index[gpu]
    i_algo = algo_index[algo]
    gpu_coreF = gpu_coreFs[i_gpu]
    gpu_memF = gpu_memFs[i_gpu]
    #==============


    local_algo_batch_sizes_list = algo_batch_sizes_list
    local_algo_nets = algo_nets
    if gpu == 'gtx2080ti':
        local_algo_batch_sizes_list = gtx_algo_batch_sizes_list
        local_algo_nets = gtx_algo_nets
        line_x = np.array([100, 200, 300, 400, 500, 600], np.dtype('int32'))        
    else:
        line_x = np.array([100, 200, 300, 400, 500, 600, 700], np.dtype('int32'))

    #==============
    fig = plt.figure(figsize=(fig_x, fig_y))
    ax1 = fig.add_subplot(1, 1, 1)
    # ax2 = ax1.twinx()
    # bar_i_width = 0

    # bar_n = len(local_algo_batch_sizes_list[i_algo][i_net])
    # bar_width = bar_total_width / bar_n
    # bar_x = line_x - (bar_total_width - bar_width) / 2
    #==============
    bars = []
    lines = []
    max_energy = 0

    for i_net, net in enumerate(local_algo_nets[i_algo]):
        batch_size = local_algo_batch_sizes_list[i_algo][i_net][-1]
        energy_respect_coreF = get_energy_respect(gpu, algo, 'caffe', net, batch_size, gpu_coreF, gpu_memF[0])
        energy_label_name = '{0}-b{1} Energy'.format(net, batch_size)
        max_energy = max(energy_respect_coreF) if max_energy < max(energy_respect_coreF) else max_energy
        lines.append(ax1.plot(line_x, energy_respect_coreF, color=plot_colors[i_net], marker=markers[i_net],
            markersize=marker_size, markerfacecolor='none', linewidth=2, label=energy_label_name))

    box = ax1.get_position()
    ax1.set_position([box.x0, box.y0, box.width, box.height * 0.8])

    lines = sum(lines, [])
    legend_elements = []
    for i in range(len(local_algo_nets[i_algo])):
        legend_elements.append(lines[i])

    ax1.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, 1.14), ncol=len(lines), prop=font_legend)

    y1_interval_num = 1
    y1_interval = 1
    y1_interval_ratios = [max_energy / standard_interval for standard_interval in y_intervals]
    for i_y1, y1_interval_ratio in enumerate(y1_interval_ratios):
        if (y1_interval_ratio>4) & (y1_interval_ratio<10):
            y1_interval_num = int(y1_interval_ratio) + 1
            y1_interval = y_intervals[i_y1]
            break

    # y2_interval_num = 1
    # y2_interval = 1
    # y2_interval_ratios = [max_energy / standard_interval for standard_interval in y_intervals]
    # for i_y2, y2_interval_ratio in enumerate(y2_interval_ratios):
    #     if (y2_interval_ratio>4) & (y2_interval_ratio<10):
    #         y2_interval_num = int(y2_interval_ratio) + 1
    #         y2_interval = y_intervals[i_y2]
    #         break

    if y1_interval < 1:
        y1_interval = int(y1_interval *100)
        y1_extend = 100
        y1_ticks = [y1_interval * i_y1 / y1_extend for i_y1 in range(y1_interval_num + 1)]
    else:
        y1_ticks = [y1_interval * i_y1 for i_y1 in range(y1_interval_num + 1)]      

    # if y2_interval < 1:
    #     y2_interval = int(y2_interval *100)
    #     y2_extend = 100
    #     y2_ticks = [y2_interval * i_y2 / y2_extend for i_y2 in range(y2_interval_num + 1)]
    # else:
    #     y2_ticks = [y2_interval * i_y2 for i_y2 in range(y2_interval_num + 1)]       

    ax1.set_yticks(y1_ticks)
    # ax2.set_yticks(y2_ticks)
    ax1.set_yticklabels(y1_ticks, fontsize=size_ticks)
    # ax2.set_yticklabels(y2_ticks, fontsize=size_ticks)

    ax1.grid(True, linestyle="--", color="0.5", linewidth=1)

    # ax2.set_ylabel("%s (J/image)" % 'energy', size=size_labels)

    ax1.set_xlabel("%s (Hz)" % 'coreF', size=size_labels)
    ax1.set_ylabel("%s (J/image)" % 'energy', size=size_labels)
    ax1.set_xticks(line_x)
    ax1.set_xticklabels([int(coreF) for coreF in gpu_coreF], size=size_ticks)

    # plt.xticks([int(coreF) for coreF in gpu_coreF], size=ticks_size)
    # plt.title("config [{0}, {1}], varient {2} ".format('net', 'batch', 'coreF'))
    if save == True:
        save_file_name = 'energy_ex2_{0}_{1}_coreF'.format(gpu, algo)
        plt.savefig(os.path.join(OUTPUT_PATH, '%s.pdf' % save_file_name), bbox_inches='tight')
        plt.savefig(os.path.join(OUTPUT_PATH, '%s.png' % save_file_name), bbox_inches='tight')
    else:
        plt.show()



def draw_ex2_2080_memF(algo, save=False):

    gpu = "gtx2080ti"
    i_gpu = gpu_index[gpu]
    i_algo = algo_index[algo]
    gpu_coreF = gpu_coreFs[i_gpu]
    gpu_memF = gpu_memFs[i_gpu]
    print(gpu_memF)
    #==============


    local_algo_batch_sizes_list = gtx_algo_batch_sizes_list
    print(local_algo_batch_sizes_list)
    local_algo_nets = gtx_algo_nets
    line_x = np.array([100, 200, 300, 400], np.dtype('int32'))        


    #==============
    fig = plt.figure(figsize=(fig_x, fig_y))
    ax1 = fig.add_subplot(1, 1, 1)
    # ax2 = ax1.twinx()
    # bar_i_width = 0

    # bar_n = len(local_algo_batch_sizes_list[i_algo][i_net])
    # bar_width = bar_total_width / bar_n
    # bar_x = line_x - (bar_total_width - bar_width) / 2
    #==============
    bars = []
    lines = []
    max_energy = 0

    for i_net, net in enumerate(local_algo_nets[i_algo]):
        batch_size = local_algo_batch_sizes_list[i_algo][i_net][-1]
        energy_respect_coreF = get_ex2_2080_geo_mean_energy_on_coreF(gpu, algo, net, batch_size)
        energy_label_name = '{0}-b{1} Energy'.format(net, batch_size)
        max_energy = max(energy_respect_coreF) if max_energy < max(energy_respect_coreF) else max_energy
        lines.append(ax1.plot(line_x, energy_respect_coreF, color=plot_colors[i_net], marker=markers[i_net],
            markersize=marker_size, markerfacecolor='none', linewidth=2, label=energy_label_name))

    box = ax1.get_position()
    ax1.set_position([box.x0, box.y0, box.width, box.height * 0.8])

    lines = sum(lines, [])
    legend_elements = []
    for i in range(len(local_algo_nets[i_algo])):
        legend_elements.append(lines[i])

    ax1.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, 1.14), ncol=len(lines), prop=font_legend)

    y1_interval_num = 1
    y1_interval = 1
    y1_interval_ratios = [max_energy / standard_interval for standard_interval in y_intervals]
    for i_y1, y1_interval_ratio in enumerate(y1_interval_ratios):
        if (y1_interval_ratio>4) & (y1_interval_ratio<10):
            y1_interval_num = int(y1_interval_ratio) + 1
            y1_interval = y_intervals[i_y1]
            break

    # y2_interval_num = 1
    # y2_interval = 1
    # y2_interval_ratios = [max_energy / standard_interval for standard_interval in y_intervals]
    # for i_y2, y2_interval_ratio in enumerate(y2_interval_ratios):
    #     if (y2_interval_ratio>4) & (y2_interval_ratio<10):
    #         y2_interval_num = int(y2_interval_ratio) + 1
    #         y2_interval = y_intervals[i_y2]
    #         break

    if y1_interval < 1:
        y1_interval = int(y1_interval *100)
        y1_extend = 100
        y1_ticks = [y1_interval * i_y1 / y1_extend for i_y1 in range(y1_interval_num + 1)]
    else:
        y1_ticks = [y1_interval * i_y1 for i_y1 in range(y1_interval_num + 1)]      

    # if y2_interval < 1:
    #     y2_interval = int(y2_interval *100)
    #     y2_extend = 100
    #     y2_ticks = [y2_interval * i_y2 / y2_extend for i_y2 in range(y2_interval_num + 1)]
    # else:
    #     y2_ticks = [y2_interval * i_y2 for i_y2 in range(y2_interval_num + 1)]       

    ax1.set_yticks(y1_ticks)
    # ax2.set_yticks(y2_ticks)
    ax1.set_yticklabels(y1_ticks, fontsize=size_ticks)
    # ax2.set_yticklabels(y2_ticks, fontsize=size_ticks)

    ax1.grid(True, linestyle="--", color="0.5", linewidth=1)

    # ax2.set_ylabel("%s (J/image)" % 'energy', size=size_labels)

    ax1.set_xlabel("%s (Hz)" % 'coreF', size=size_labels)
    ax1.set_ylabel("%s (J/image)" % 'energy', size=size_labels)
    ax1.set_xticks(line_x)
    ax1.set_xticklabels([int(memF) for memF in gpu_memF], size=size_ticks)

    # plt.xticks([int(coreF) for coreF in gpu_coreF], size=ticks_size)
    # plt.title("config [{0}, {1}], varient {2} ".format('net', 'batch', 'coreF'))
    if save == True:
        save_file_name = 'energy_ex2_{0}_{1}_memF'.format(gpu, algo)
        plt.savefig(os.path.join(OUTPUT_PATH, '%s.pdf' % save_file_name), bbox_inches='tight')
        plt.savefig(os.path.join(OUTPUT_PATH, '%s.png' % save_file_name), bbox_inches='tight')
    else:
        plt.show()





def draw_ex2_2080_coreF(algo, save=False):

    algo = 'ipc_gemm'
    gpu = "gtx2080ti"
    i_gpu = gpu_index[gpu]
    i_algo = algo_index[algo]
    gpu_coreF = gpu_coreFs[i_gpu]
    gpu_memF = gpu_memFs[i_gpu]
    print(gpu_coreF)
    #==============


    local_algo_batch_sizes_list = gtx_algo_batch_sizes_list
    local_algo_nets = gtx_algo_nets
    line_x = np.array([100, 200, 300, 400, 500, 600], np.dtype('int32'))        


    #==============
    fig = plt.figure(figsize=(fig_x, fig_y))
    ax1 = fig.add_subplot(1, 1, 1)
    # ax2 = ax1.twinx()
    # bar_i_width = 0

    # bar_n = len(local_algo_batch_sizes_list[i_algo][i_net])
    # bar_width = bar_total_width / bar_n
    # bar_x = line_x - (bar_total_width - bar_width) / 2
    #==============
    bars = []
    lines = []
    max_energy = 0

    for i_net, net in enumerate(local_algo_nets[i_algo]):
        batch_size = local_algo_batch_sizes_list[i_algo][i_net][-1]
        energy_respect_coreF = get_ex2_2080_geo_mean_energy_on_memF(gpu, algo, net, batch_size)
        energy_label_name = '{0}-b{1} Energy'.format(net, batch_size)
        max_energy = max(energy_respect_coreF) if max_energy < max(energy_respect_coreF) else max_energy
        lines.append(ax1.plot(line_x, energy_respect_coreF, color=plot_colors[i_net], marker=markers[i_net],
            markersize=marker_size, markerfacecolor='none', linewidth=2, label=energy_label_name))

    box = ax1.get_position()
    ax1.set_position([box.x0, box.y0, box.width, box.height * 0.8])

    lines = sum(lines, [])
    legend_elements = []
    for i in range(len(local_algo_nets[i_algo])):
        legend_elements.append(lines[i])

    ax1.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, 1.14), ncol=len(lines), prop=font_legend)

    y1_interval_num = 1
    y1_interval = 1
    y1_interval_ratios = [max_energy / standard_interval for standard_interval in y_intervals]
    for i_y1, y1_interval_ratio in enumerate(y1_interval_ratios):
        if (y1_interval_ratio>4) & (y1_interval_ratio<10):
            y1_interval_num = int(y1_interval_ratio) + 1
            y1_interval = y_intervals[i_y1]
            break

    # y2_interval_num = 1
    # y2_interval = 1
    # y2_interval_ratios = [max_energy / standard_interval for standard_interval in y_intervals]
    # for i_y2, y2_interval_ratio in enumerate(y2_interval_ratios):
    #     if (y2_interval_ratio>4) & (y2_interval_ratio<10):
    #         y2_interval_num = int(y2_interval_ratio) + 1
    #         y2_interval = y_intervals[i_y2]
    #         break

    if y1_interval < 1:
        y1_interval = int(y1_interval *100)
        y1_extend = 100
        y1_ticks = [y1_interval * i_y1 / y1_extend for i_y1 in range(y1_interval_num + 1)]
    else:
        y1_ticks = [y1_interval * i_y1 for i_y1 in range(y1_interval_num + 1)]      

    # if y2_interval < 1:
    #     y2_interval = int(y2_interval *100)
    #     y2_extend = 100
    #     y2_ticks = [y2_interval * i_y2 / y2_extend for i_y2 in range(y2_interval_num + 1)]
    # else:
    #     y2_ticks = [y2_interval * i_y2 for i_y2 in range(y2_interval_num + 1)]       

    ax1.set_yticks(y1_ticks)
    # ax2.set_yticks(y2_ticks)
    ax1.set_yticklabels(y1_ticks, fontsize=size_ticks)
    # ax2.set_yticklabels(y2_ticks, fontsize=size_ticks)

    ax1.grid(True, linestyle="--", color="0.5", linewidth=1)

    # ax2.set_ylabel("%s (J/image)" % 'energy', size=size_labels)

    ax1.set_xlabel("%s (Hz)" % 'coreF', size=size_labels)
    ax1.set_ylabel("%s (J/image)" % 'energy', size=size_labels)
    ax1.set_xticks(line_x)
    ax1.set_xticklabels([int(coreF) for coreF in gpu_coreF], size=size_ticks)

    # plt.xticks([int(coreF) for coreF in gpu_coreF], size=ticks_size)
    # plt.title("config [{0}, {1}], varient {2} ".format('net', 'batch', 'coreF'))
    if save == True:
        save_file_name = 'energy_ex2_{0}_{1}_coreF'.format(gpu, algo)
        plt.savefig(os.path.join(OUTPUT_PATH, '%s.pdf' % save_file_name), bbox_inches='tight')
        plt.savefig(os.path.join(OUTPUT_PATH, '%s.png' % save_file_name), bbox_inches='tight')
    else:
        plt.show()







# draw_ex2_p100_v100_coreF('p100', 'ipc_gemm', save=True)
# draw_ex2_p100_v100_coreF('v100', 'ipc_gemm', save=True)

# draw_ex2_2080_memF('ipc_gemm', save=True)
# draw_ex2_2080_coreF('ipc_gemm', save=True)


# draw_ex2_2080_memF('ipc_gemm')
# draw_ex2_2080_coreF('ipc_gemm')





