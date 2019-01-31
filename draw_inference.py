from draw import *
from cal_geo_mean import *

# gpu = ['p100', 'v100', 'gtx2080ti']
# algos = ['ipc_gemm', 'fft_tile', 'winograd']
nets = ['alexnet', 'resnet50', 'vgg16', 'googlenet']
# p100_coreF = ['544', '683', '810', '936', '1063', '1202', '1328']
v100_coreF = ['510', '652', '802', '982', '1087', '1237', '1380']
# gtx2080ti_coreF = ['950', '1150', '1350', '1550', '1750', '2050']
# gpu_coreFs = [p100_coreF, v100_coreF, gtx2080ti_coreF]
# p100_memF = ['715']
# v100_memF = ['877']
# gtx2080ti_memF = ['5800', '6300', '6800', '7300']
# gpu_memFs = [p100_memF, v100_memF, gtx2080ti_memF]

batch_size = '256'

def get_2080_inference_memF_geo_mean_perf_and_power(algo, net, batch_size):
    gpu = "gtx2080ti"

    i_gpu = gpu_index[gpu]

    local_algo_batch_sizes_list = algo_batch_sizes_list
    local_algo_nets = algo_nets

    local_algo_batch_sizes_list = gtx_algo_batch_sizes_list
    local_algo_nets = gtx_algo_nets
    perf_results = np.array([1, 1, 1, 1, 1, 1], np.dtype('float'))
    power_results = np.array([1, 1, 1, 1, 1, 1], np.dtype('float'))


    n = 0
    n = len(gtx2080ti_memF)

    for memF in gtx2080ti_memF:
        perfs_respect_coreF = np.array(data.get_image_perf_respect(gpu, algo, 'giexec', net, batch_size, gpu_coreFs[i_gpu], memF), np.dtype('float'))
        perfs_sqrt = np.power(perfs_respect_coreF, 1/n)
        perf_results = perf_results * perfs_sqrt

        powers_respect_coreF = np.array(data.get_power_respect(gpu, algo, 'giexec', net, batch_size, gpu_coreFs[i_gpu], memF), np.dtype('float'))
        powers_sqrt = np.power(powers_respect_coreF, 1/n)
        power_results = power_results * powers_sqrt


    return power_results, perf_results

def get_2080_inference_coreF_geo_mean_perf_and_power(algo, net, batch_size):
    gpu = "gtx2080ti"

    i_gpu = gpu_index[gpu]

    local_algo_batch_sizes_list = gtx_algo_batch_sizes_list
    local_algo_nets = gtx_algo_nets
    perf_results = np.array([1, 1, 1, 1], np.dtype('float'))
    power_results = np.array([1, 1, 1, 1], np.dtype('float'))


    n = 0
    n = len(gpu_coreFs[i_gpu])

    for coreF in gpu_coreFs[i_gpu]:
        perfs_respect_coreF = np.array(data.get_image_perf_respect(gpu, algo, 'giexec', net, batch_size, coreF, gpu_memFs[i_gpu]), np.dtype('float'))
        perfs_sqrt = np.power(perfs_respect_coreF, 1/n)
        perf_results = perf_results * perfs_sqrt

        powers_respect_coreF = np.array(data.get_power_respect(gpu, algo, 'giexec', net, batch_size, coreF, gpu_memFs[i_gpu]), np.dtype('float'))
        powers_sqrt = np.power(powers_respect_coreF, 1/n)
        power_results = power_results * powers_sqrt


    return power_results, perf_results



def get_2080_inference_geo_mean_energy_on_coreF(gpu, algo, net, batch_size):
    gpu = 'gtx2080ti'

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
        energys_respect_memF = np.array(data.get_energy_respect(gpu, algo, 'giexec', net, batch_size, coreF, gpu_memFs[i_gpu]), np.dtype('float'))
        energys_sqrt = np.power(energys_respect_memF, 1/n)
        results = results * energys_sqrt

    return results



def get_2080_inference_geo_mean_energy_on_memF(gpu, algo, net, batch_size):
    gpu = 'gtx2080ti'

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
        energys_respect_memF = np.array(data.get_energy_respect(gpu, algo, 'giexec', net, batch_size, gpu_coreFs[i_gpu], memF), np.dtype('float'))
        energys_sqrt = np.power(energys_respect_memF, 1/n)
        results = results * energys_sqrt

    return results

# =============================================draw==========================
# =============================================draw==========================
# =============================================draw==========================
# =============================================draw==========================
# =============================================draw==========================
# =============================================draw==========================
# =============================================draw==========================
# =============================================draw==========================
# =============================================draw==========================


def draw_inference_perf_power(
    gpu, algo, nets, batch_size, gpu_coreF, gpu_memF, save=False):

    fig = plt.figure(figsize=(fig_x, fig_y))
    ax1 = fig.add_subplot(1, 1, 1)
    #==============
    ax2 = ax1.twinx()
    bar_i_width = 0

    bar_n = len(batch_sizes)
    bar_width = bar_total_width / bar_n
    np.array(gpu_coreF)
    # bar_x = np.array(gpu_coreF, np.dtype('int32')) - (bar_total_width - bar_width) / 2
    if gpu == 'gtx2080ti':
        line_x = np.array([100, 200, 300, 400, 500, 600], np.dtype('int32'))
    else:
        line_x = np.array([100, 200, 300, 400, 500, 600, 700], np.dtype('int32'))
    bar_x = line_x - (bar_total_width - bar_width) / 2

    #==============
    #==============
    max_power = 0
    max_image_perf = 0
    bars = []
    lines = []
    for i_net, net in enumerate(nets):
        powers_respect_coreF = get_power_respect(gpu, algo, 'giexec', net, batch_size, gpu_coreF, gpu_memF)
        power_label_name = '{0}-b{1} Power'.format(net, batch_size)
        max_power = max(powers_respect_coreF) if max_power < max(powers_respect_coreF) else max_power
        bars.append(ax1.bar(bar_x + bar_i_width * bar_width, powers_respect_coreF, color=bar_colors[i_net],
            width=bar_width, label=power_label_name))
        bar_i_width += 1

    for i_net, net in enumerate(nets):
        image_perf_respect_coreF = get_image_perf_respect(gpu, algo, 'giexec', net, batch_size, gpu_coreF, gpu_memF)       
        max_image_perf = max(image_perf_respect_coreF) if max_image_perf < max(image_perf_respect_coreF) else max_image_perf

        energy_label_name = '{0}-b{1} image_per'.format(net, batch_size)
        lines.append(ax2.plot(line_x, image_perf_respect_coreF, color=plot_colors[i_net], marker=markers[i_net],
            markersize=marker_size, markerfacecolor='none', linewidth=2, label=energy_label_name))
    
    # y1_intervals = int(max_power/12)
    # y1_loc = plticker.MultipleLocator(base=y1_intervals)
    # ax1.yaxis.set_major_locator(y1_loc)
    # ax1.grid(True, linestyle="--", color="0.5", linewidth=1)

    box = ax1.get_position()
    ax1.set_position([box.x0, box.y0, box.width, box.height * 0.8])

    bars = list(bars)
    lines = sum(lines, [])
    legend_elements = []
    for i in range(len(bars)):
        legend_elements.append(bars[i])
        legend_elements.append(lines[i])

    ax1.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, 1.14), ncol=len(bars), prop=font_legend)
    #ax1.legend(loc='upper left', bbox_to_anchor=(0., 0.98, 0.8, 0.1), ncol=4, mode="expand", borderaxespad=0., prop=font_legend)
    #ax2.legend(loc='upper left', bbox_to_anchor=(0., 1.05, 0.8, 0.1), ncol=4, mode="expand", borderaxespad=0., prop=font_legend)
    
    y1_interval_num = 1
    y1_interval = 1
    y1_interval_ratios = [max_power / standard_interval for standard_interval in y_intervals]
    for i_y1, y1_interval_ratio in enumerate(y1_interval_ratios):
        if (y1_interval_ratio>4) & (y1_interval_ratio<10):
            y1_interval_num = int(y1_interval_ratio) + 1
            y1_interval = y_intervals[i_y1]
            break

    y2_interval_num = 1
    y2_interval = 1
    y2_interval_ratios = [max_image_perf / standard_interval for standard_interval in y_intervals]
    for i_y2, y2_interval_ratio in enumerate(y2_interval_ratios):
        if (y2_interval_ratio>4) & (y2_interval_ratio<10):
            y2_interval_num = int(y2_interval_ratio) + 1
            y2_interval = y_intervals[i_y2]
            break

    if y1_interval < 1:
        y1_interval = int(y1_interval *100)
        y1_extend = 100
        y1_ticks = [y1_interval * i_y1 / y1_extend for i_y1 in range(y1_interval_num + 1)]
    else:
        y1_ticks = [y1_interval * i_y1 for i_y1 in range(y1_interval_num + 1)]      

    if y2_interval < 1:
        y2_interval = int(y2_interval *100)
        y2_extend = 100
        y2_ticks = [y2_interval * i_y2 / y2_extend for i_y2 in range(y2_interval_num + 1)]
    else:
        y2_ticks = [y2_interval * i_y2 for i_y2 in range(y2_interval_num + 1)]       

    ax1.set_yticks(y1_ticks)
    ax2.set_yticks(y2_ticks)
    ax1.set_yticklabels(y1_ticks, fontsize=size_ticks)
    ax2.set_yticklabels(y2_ticks, fontsize=size_ticks)
    ax1.grid(True, linestyle="--", color="0.5", linewidth=1)

    ax2.set_ylabel("%s (image/s)" % 'perf', size=size_labels)
    ax1.set_xlabel("%s (Hz)" % 'coreF', size=size_labels)
    ax1.set_ylabel("%s (V)" % 'power', size=size_labels)
    ax1.set_xticks(line_x)
    ax1.set_xticklabels([int(coreF) for coreF in gpu_coreF], size=size_ticks)
    # plt.xticks([int(coreF) for coreF in gpu_coreF])
    # plt.title("config [{0}, {1}], varient {2} ".format('net', 'batch', 'coreF'))
    if save == True:
        save_file_name = 'inference_power_perf_{0}'.format(gpu)
        plt.savefig(os.path.join(OUTPUT_PATH, '%s.pdf' % save_file_name), bbox_inches='tight')
        plt.savefig(os.path.join(OUTPUT_PATH, '%s.png' % save_file_name), bbox_inches='tight')
    else:
        plt.show()



def draw_inference_energy(
    gpu, algo, nets, batch_size, gpu_coreF, gpu_memF, save=False):

    fig = plt.figure(figsize=(fig_x, fig_y))
    ax1 = fig.add_subplot(1, 1, 1)
    #==============
    # ax2 = ax1.twinx()
    bar_i_width = 0

    bar_n = len(batch_sizes)
    bar_width = bar_total_width / bar_n
    np.array(gpu_coreF)
    # bar_x = np.array(gpu_coreF, np.dtype('int32')) - (bar_total_width - bar_width) / 2
    if gpu == 'gtx2080ti':
        line_x = np.array([100, 200, 300, 400, 500, 600], np.dtype('int32'))
    else:
        line_x = np.array([100, 200, 300, 400, 500, 600, 700], np.dtype('int32'))
    # bar_x = line_x - (bar_total_width - bar_width) / 2

    #==============
    #==============
    max_energy = 0
    bars = []
    lines = []
    for i_net, net in enumerate(nets):
        energy_respect_coreF = get_energy_respect(gpu, algo, 'giexec', net, batch_size, gpu_coreF, gpu_memF)
        energy_label_name = '{0} Energy'.format(net)
        max_energy = max(energy_respect_coreF) if max_energy < max(energy_respect_coreF) else max_energy
        lines.append(ax1.plot(line_x, energy_respect_coreF, color=plot_colors[i_net], marker=markers[i_net],  
            markersize=marker_size, markerfacecolor='none', linewidth=2, label=energy_label_name))

    # y1_intervals = int(max_power/12)
    # y1_loc = plticker.MultipleLocator(base=y1_intervals)
    # ax1.yaxis.set_major_locator(y1_loc)
    # ax1.grid(True, linestyle="--", color="0.5", linewidth=1)

    box = ax1.get_position()
    ax1.set_position([box.x0, box.y0, box.width, box.height * 0.8])

    # bars = list(bars)
    lines = sum(lines, [])
    legend_elements = []
    for i in range(len(lines)):
        # legend_elements.append(bars[i])
        legend_elements.append(lines[i])

    ax1.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, 1.14), ncol=len(lines), prop=font_legend)
    #ax1.legend(loc='upper left', bbox_to_anchor=(0., 0.98, 0.8, 0.1), ncol=4, mode="expand", borderaxespad=0., prop=font_legend)
    #ax2.legend(loc='upper left', bbox_to_anchor=(0., 1.05, 0.8, 0.1), ncol=4, mode="expand", borderaxespad=0., prop=font_legend)
    
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
    # y2_interval_ratios = [max_image_perf / standard_interval for standard_interval in y_intervals]
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

    # ax2.set_ylabel("%s (image/s)" % 'perf', size=size_labels)
    ax1.set_xlabel("%s (Hz)" % 'coreF', size=size_labels)
    ax1.set_ylabel("%s (J/image)" % 'Energy', size=size_labels)
    ax1.set_xticks(line_x)
    ax1.set_xticklabels([int(coreF) for coreF in gpu_coreF], size=size_ticks)
    # plt.xticks([int(coreF) for coreF in gpu_coreF])
    # plt.title("config [{0}, {1}], varient {2} ".format('net', 'batch', 'coreF'))
    if save == True:
        save_file_name = 'inference_energy_{0}'.format(gpu)
        plt.savefig(os.path.join(OUTPUT_PATH, '%s.pdf' % save_file_name), bbox_inches='tight')
        plt.savefig(os.path.join(OUTPUT_PATH, '%s.png' % save_file_name), bbox_inches='tight')
    else:
        plt.show()








def draw_inference_2080_coreF(algo, save=False):
    gpu = "gtx2080ti"

    i_gpu = gpu_index[gpu]
    gpu_coreF = gpu_coreFs[i_gpu]

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
    ax2 = ax1.twinx()
    bar_i_width = 0

    bar_n = len(nets)
    bar_width = bar_total_width / bar_n
    bar_x = line_x - (bar_total_width - bar_width) / 2

    bars = []
    lines = []
    max_power = 0
    max_image_perf = 0

    for i_net, net in enumerate(nets):

        powers_respect_coreF, image_perf_respect_coreF = get_2080_inference_memF_geo_mean_perf_and_power(algo, net, batch_size)
        power_label_name = '{0}-b{1} Power'.format(net, batch_size)

        max_power = max(powers_respect_coreF) if max_power < max(powers_respect_coreF) else max_power
        bars.append(ax1.bar(bar_x + bar_i_width * bar_width, powers_respect_coreF, color=bar_colors[i_net],
            width=bar_width, label=power_label_name))
        bar_i_width += 1

        max_image_perf = max(image_perf_respect_coreF) if max_image_perf < max(image_perf_respect_coreF) else max_image_perf

        energy_label_name = '{0}-b{1} image_per'.format(net, batch_size)
        lines.append(ax2.plot(line_x, image_perf_respect_coreF, color=plot_colors[i_net], marker=markers[i_net],
            markersize=marker_size, markerfacecolor='none', linewidth=2, label=energy_label_name))
    
    # y1_intervals = int(max_power/12)
    # y1_loc = plticker.MultipleLocator(base=y1_intervals)
    # ax1.yaxis.set_major_locator(y1_loc)
    # ax1.grid(True, linestyle="--", color="0.5", linewidth=1)

    box = ax1.get_position()
    ax1.set_position([box.x0, box.y0, box.width, box.height * 0.8])
    #ax1.legend(loc='upper center', bbox_to_anchor=(0., 0, 0.9, 0.1), ncol=4, prop=font_legend)
    #ax2.legend(loc='upper center', bbox_to_anchor=(0., 0, 0.92, 0.1), ncol=4, prop=font_legend)
    bars = list(bars)
    lines = sum(lines, [])
    legend_elements = []
    for i in range(len(bars)):
        legend_elements.append(bars[i])
        legend_elements.append(lines[i])

    # print legend_elements
    ax1.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, 1.14), ncol=len(bars), prop=font_legend)
    #ax2.legend(loc='upper center', ncol=4, prop=font_legend)
    y1_interval_num = 1
    y1_interval = 1
    y1_interval_ratios = [max_power / standard_interval for standard_interval in y_intervals]
    for i_y1, y1_interval_ratio in enumerate(y1_interval_ratios):
        if (y1_interval_ratio>4) & (y1_interval_ratio<10):
            y1_interval_num = int(y1_interval_ratio) + 1
            y1_interval = y_intervals[i_y1]
            break

    y2_interval_num = 1
    y2_interval = 1
    y2_interval_ratios = [max_image_perf / standard_interval for standard_interval in y_intervals]
    for i_y2, y2_interval_ratio in enumerate(y2_interval_ratios):
        if (y2_interval_ratio>4) & (y2_interval_ratio<10):
            y2_interval_num = int(y2_interval_ratio) + 1
            y2_interval = y_intervals[i_y2]
            break

    if y1_interval < 1:
        y1_interval = int(y1_interval *100)
        y1_extend = 100
        y1_ticks = [y1_interval * i_y1 / y1_extend for i_y1 in range(y1_interval_num + 1)]
    else:
        y1_ticks = [y1_interval * i_y1 for i_y1 in range(y1_interval_num + 1)]      

    if y2_interval < 1:
        y2_interval = int(y2_interval *100)
        y2_extend = 100
        y2_ticks = [y2_interval * i_y2 / y2_extend for i_y2 in range(y2_interval_num + 1)]
    else:
        y2_ticks = [y2_interval * i_y2 for i_y2 in range(y2_interval_num + 1)]       

    ax1.set_yticks(y1_ticks)
    ax2.set_yticks(y2_ticks)
    ax1.set_yticklabels(y1_ticks, fontsize=size_ticks)
    ax2.set_yticklabels(y2_ticks, fontsize=size_ticks)
    ax1.grid(True, linestyle="--", color="0.5", linewidth=1)

    ax2.set_ylabel("%s (image/s)" % 'perf', size=size_labels)
    ax1.set_xlabel("%s (Hz)" % 'coreF', size=size_labels)
    ax1.set_ylabel("%s (J)" % 'power', size=size_labels)
    ax1.set_xticks(line_x)
    ax1.set_xticklabels([int(coreF) for coreF in gpu_coreF], size=size_ticks)
    # plt.xticks([int(coreF) for coreF in gpu_coreF])
    # plt.title("config [{0}, {1}], varient {2} ".format('net', 'batch', 'coreF'))
    if save == True:
        save_file_name = 'inference_power_perf_{0}_coreF'.format(gpu)
        plt.savefig(os.path.join(OUTPUT_PATH, '%s.pdf' % save_file_name), bbox_inches='tight')
        plt.savefig(os.path.join(OUTPUT_PATH, '%s.png' % save_file_name), bbox_inches='tight')
    else:
        plt.show()


def draw_inference_2080_memF(algo, save=False):
    gpu = "gtx2080ti"
    i_gpu = gpu_index[gpu]
    gpu_coreF = gpu_coreFs[i_gpu]

    #==============
    local_algo_batch_sizes_list = gtx_algo_batch_sizes_list
    local_algo_nets = gtx_algo_nets
    line_x = np.array([100, 200, 300, 400], np.dtype('int32'))

    #==============
    fig = plt.figure(figsize=(fig_x, fig_y))
    ax1 = fig.add_subplot(1, 1, 1)
    ax2 = ax1.twinx()
    bar_i_width = 0

    bar_n = len(nets)
    bar_width = bar_total_width / bar_n
    bar_x = line_x - (bar_total_width - bar_width) / 2

    bars = []
    lines = []
    max_power = 0
    max_image_perf = 0

    for i_net, net in enumerate(nets):


        powers_respect_coreF, image_perf_respect_coreF = get_2080_inference_coreF_geo_mean_perf_and_power(algo, net, batch_size)
        power_label_name = '{0}-b{1} Power'.format(net, batch_size)

        max_power = max(powers_respect_coreF) if max_power < max(powers_respect_coreF) else max_power
        bars.append(ax1.bar(bar_x + bar_i_width * bar_width, powers_respect_coreF, color=bar_colors[i_net],
            width=bar_width, label=power_label_name))
        bar_i_width += 1

        max_image_perf = max(image_perf_respect_coreF) if max_image_perf < max(image_perf_respect_coreF) else max_image_perf

        energy_label_name = '{0}-b{1} image_per'.format(net, batch_size)
        lines.append(ax2.plot(line_x, image_perf_respect_coreF, color=plot_colors[i_net], marker=markers[i_net],
            markersize=marker_size, markerfacecolor='none', linewidth=2, label=energy_label_name))
    
    # y1_intervals = int(max_power/12)
    # y1_loc = plticker.MultipleLocator(base=y1_intervals)
    # ax1.yaxis.set_major_locator(y1_loc)
    # ax1.grid(True, linestyle="--", color="0.5", linewidth=1)

    box = ax1.get_position()
    ax1.set_position([box.x0, box.y0, box.width, box.height * 0.8])
    #ax1.legend(loc='upper center', bbox_to_anchor=(0., 0, 0.9, 0.1), ncol=4, prop=font_legend)
    #ax2.legend(loc='upper center', bbox_to_anchor=(0., 0, 0.92, 0.1), ncol=4, prop=font_legend)
    bars = list(bars)
    lines = sum(lines, [])
    legend_elements = []
    for i in range(len(bars)):
        legend_elements.append(bars[i])
        legend_elements.append(lines[i])

    # print legend_elements
    ax1.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, 1.14), ncol=len(bars), prop=font_legend)
    #ax2.legend(loc='upper center', ncol=4, prop=font_legend)
    y1_interval_num = 1
    y1_interval = 1
    y1_interval_ratios = [max_power / standard_interval for standard_interval in y_intervals]
    for i_y1, y1_interval_ratio in enumerate(y1_interval_ratios):
        if (y1_interval_ratio>4) & (y1_interval_ratio<10):
            y1_interval_num = int(y1_interval_ratio) + 1
            y1_interval = y_intervals[i_y1]
            break

    y2_interval_num = 1
    y2_interval = 1
    y2_interval_ratios = [max_image_perf / standard_interval for standard_interval in y_intervals]
    for i_y2, y2_interval_ratio in enumerate(y2_interval_ratios):
        if (y2_interval_ratio>4) & (y2_interval_ratio<10):
            y2_interval_num = int(y2_interval_ratio) + 1
            y2_interval = y_intervals[i_y2]
            break

    if y1_interval < 1:
        y1_interval = int(y1_interval *100)
        y1_extend = 100
        y1_ticks = [y1_interval * i_y1 / y1_extend for i_y1 in range(y1_interval_num + 1)]
    else:
        y1_ticks = [y1_interval * i_y1 for i_y1 in range(y1_interval_num + 1)]      

    if y2_interval < 1:
        y2_interval = int(y2_interval *100)
        y2_extend = 100
        y2_ticks = [y2_interval * i_y2 / y2_extend for i_y2 in range(y2_interval_num + 1)]
    else:
        y2_ticks = [y2_interval * i_y2 for i_y2 in range(y2_interval_num + 1)]       

    ax1.set_yticks(y1_ticks)
    ax2.set_yticks(y2_ticks)
    ax1.set_yticklabels(y1_ticks, fontsize=size_ticks)
    ax2.set_yticklabels(y2_ticks, fontsize=size_ticks)
    ax1.grid(True, linestyle="--", color="0.5", linewidth=1)

    ax2.set_ylabel("%s (image/s)" % 'perf', size=size_labels)
    ax1.set_xlabel("%s (Hz)" % 'memF', size=size_labels)
    ax1.set_ylabel("%s (J)" % 'power', size=size_labels)
    ax1.set_xticks(line_x)
    ax1.set_xticklabels([int(memF) for memF in gpu_memFs[i_gpu]], size=size_ticks)
    # plt.xticks([int(coreF) for coreF in gpu_coreF])
    # plt.title("config [{0}, {1}], varient {2} ".format('net', 'batch', 'coreF'))
    if save == True:
        save_file_name = 'inference_power_perf_{0}_memF'.format(gpu)
        plt.savefig(os.path.join(OUTPUT_PATH, '%s.pdf' % save_file_name), bbox_inches='tight')
        plt.savefig(os.path.join(OUTPUT_PATH, '%s.png' % save_file_name), bbox_inches='tight')
    else:
        plt.show()




def draw_inference_2080_energy_memF(algo, save=False):

    gpu = "gtx2080ti"
    i_gpu = gpu_index[gpu]
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

    for i_net, net in enumerate(nets):
        energy_respect_coreF = get_2080_inference_geo_mean_energy_on_coreF(gpu, algo, net, batch_size)
        energy_label_name = '{0}-b{1} Energy'.format(net, batch_size)
        max_energy = max(energy_respect_coreF) if max_energy < max(energy_respect_coreF) else max_energy
        lines.append(ax1.plot(line_x, energy_respect_coreF, color=plot_colors[i_net], marker=markers[i_net],
            markersize=marker_size, markerfacecolor='none', linewidth=2, label=energy_label_name))

    box = ax1.get_position()
    ax1.set_position([box.x0, box.y0, box.width, box.height * 0.8])

    lines = sum(lines, [])
    legend_elements = []
    for i in range(len(lines)):
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
        save_file_name = 'inference_energy_{0}_memF'.format(gpu)
        plt.savefig(os.path.join(OUTPUT_PATH, '%s.pdf' % save_file_name), bbox_inches='tight')
        plt.savefig(os.path.join(OUTPUT_PATH, '%s.png' % save_file_name), bbox_inches='tight')
    else:
        plt.show()





def draw_inference_2080_energy_coreF(algo, save=False):

    gpu = "gtx2080ti"
    i_gpu = gpu_index[gpu]

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

    for i_net, net in enumerate(nets):

        energy_respect_coreF = get_2080_inference_geo_mean_energy_on_memF(gpu, algo, net, batch_size)
        energy_label_name = '{0}-b{1} Energy'.format(net, batch_size)
        max_energy = max(energy_respect_coreF) if max_energy < max(energy_respect_coreF) else max_energy
        lines.append(ax1.plot(line_x, energy_respect_coreF, color=plot_colors[i_net], marker=markers[i_net],
            markersize=marker_size, markerfacecolor='none', linewidth=2, label=energy_label_name))

    box = ax1.get_position()
    ax1.set_position([box.x0, box.y0, box.width, box.height * 0.8])

    lines = sum(lines, [])
    legend_elements = []
    for i in range(len(lines)):
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
        save_file_name = 'inference_energy_{0}_coreF'.format(gpu)
        plt.savefig(os.path.join(OUTPUT_PATH, '%s.pdf' % save_file_name), bbox_inches='tight')
        plt.savefig(os.path.join(OUTPUT_PATH, '%s.png' % save_file_name), bbox_inches='tight')
    else:
        plt.show()




# draw_inference_2080_memF('auto', save=True)
# draw_inference_2080_coreF('auto', save=True)

# draw_inference_perf_power('v100', 'auto', nets, batch_size, v100_coreF, v100_memF[0], save=True)

# draw_inference_perf_power('p100', 'auto', nets, batch_size, p100_coreF, p100_memF[0], save=True)


# draw_inference_energy('v100', 'auto', nets, batch_size, v100_coreF, v100_memF[0], save=True)
# draw_inference_energy('p100', 'auto', nets, batch_size, p100_coreF, p100_memF[0], save=True)

# draw_inference_2080_energy_memF('auto', save=True)
# draw_inference_2080_energy_coreF('auto', save=True)s

