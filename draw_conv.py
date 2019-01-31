from draw import *
from cal_geo_mean import *




# ========================================================
def get_gpu_algo_geo_mean_power(gpu, algo):
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

    n = 0
    for i_net, net in enumerate(local_algo_nets[i_algo]):
        n += len(local_algo_batch_sizes_list[i_algo][i_net])

    for i_net, net in enumerate(local_algo_nets[i_algo]):
        for i_batch, batch in enumerate(local_algo_batch_sizes_list[i_algo][i_net]):
            powers_respect_coreF = np.array(data.get_power_respect(gpu, algo, 'caffe', net, batch, gpu_coreFs[i_gpu], gpu_memFs[i_gpu][0]), np.dtype('float'))
            powers_sqrt = np.power(powers_respect_coreF, 1/n)
            results = results * powers_sqrt

    return results

def get_gpu_algo_geo_mean_perf(gpu, algo):
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

    n = 0
    for i_net, net in enumerate(local_algo_nets[i_algo]):
        n += len(local_algo_batch_sizes_list[i_algo][i_net])

    for i_net, net in enumerate(local_algo_nets[i_algo]):
        for i_batch, batch in enumerate(local_algo_batch_sizes_list[i_algo][i_net]):
            perfs_respect_coreF = np.array(data.get_image_perf_respect(gpu, algo, 'caffe', net, batch, gpu_coreFs[i_gpu], gpu_memFs[i_gpu][0]), np.dtype('float'))
            perfs_sqrt = np.power(perfs_respect_coreF, 1/n)
            results = results * perfs_sqrt

    return results

def get_gpu_algo_geo_mean_energy(gpu, algo):
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

    n = 0
    for i_net, net in enumerate(local_algo_nets[i_algo]):
        n += len(local_algo_batch_sizes_list[i_algo][i_net])

    for i_net, net in enumerate(local_algo_nets[i_algo]):
        for i_batch, batch in enumerate(local_algo_batch_sizes_list[i_algo][i_net]):
            energys_respect_coreF = np.array(data.get_energy_respect(gpu, algo, 'caffe', net, batch, gpu_coreFs[i_gpu], gpu_memFs[i_gpu][0]), np.dtype('float'))
            energys_sqrt = np.power(energys_respect_coreF, 1/n)
            results = results * energys_sqrt

    return results


# ========================================================
# ========================================================
# ========================================================
# ========================================================
# ========================================================
def get_conv_2080_mean_power_coreF(gpu, algo):
    gpu = 'gtx2080ti'
    i_algo = algo_index[algo]
    i_gpu = gpu_index[gpu]

    local_algo_batch_sizes_list = gtx_algo_batch_sizes_list
    local_algo_nets = gtx_algo_nets
    results = np.array([1, 1, 1, 1], np.dtype('float'))

    n = 0
    for i_coreF, coreF in enumerate(gpu_coreFs[i_gpu]):
        for i_net, net in enumerate(local_algo_nets[i_algo]):
            n += len(local_algo_batch_sizes_list[i_algo][i_net])

    for i_coreF, coreF in enumerate(gpu_coreFs[i_gpu]):
        for i_net, net in enumerate(local_algo_nets[i_algo]):
            for i_batch, batch in enumerate(local_algo_batch_sizes_list[i_algo][i_net]):
                powers_respect_coreF = np.array(data.get_power_respect(gpu, algo, 'caffe', net, batch, coreF, gpu_memFs[i_gpu]), np.dtype('float'))
                powers_sqrt = np.power(powers_respect_coreF, 1/n)
                results = results * powers_sqrt

    return results

def get_conv_2080_mean_perf_coreF(gpu, algo):
    gpu = 'gtx2080ti'
    i_algo = algo_index[algo]
    i_gpu = gpu_index[gpu]

    local_algo_batch_sizes_list = gtx_algo_batch_sizes_list
    local_algo_nets = gtx_algo_nets
    results = np.array([1, 1, 1, 1], np.dtype('float'))

    n = 0
    for i_coreF, coreF in enumerate(gpu_coreFs[i_gpu]):    
        for i_net, net in enumerate(local_algo_nets[i_algo]):
            n += len(local_algo_batch_sizes_list[i_algo][i_net])
    for i_coreF, coreF in enumerate(gpu_coreFs[i_gpu]):  
        for i_net, net in enumerate(local_algo_nets[i_algo]):
            for i_batch, batch in enumerate(local_algo_batch_sizes_list[i_algo][i_net]):
                perfs_respect_coreF = np.array(data.get_image_perf_respect(gpu, algo, 'caffe', net, batch, coreF, gpu_memFs[i_gpu]), np.dtype('float'))
                perfs_sqrt = np.power(perfs_respect_coreF, 1/n)
                results = results * perfs_sqrt

    return results

def get_conv_2080_mean_energy_coreF(gpu, algo):
    gpu = 'gtx2080ti'
    i_algo = algo_index[algo]
    i_gpu = gpu_index[gpu]

    local_algo_batch_sizes_list = gtx_algo_batch_sizes_list
    local_algo_nets = gtx_algo_nets
    results = np.array([1, 1, 1, 1], np.dtype('float'))

    n = 0
    for i_coreF, coreF in enumerate(gpu_coreFs[i_gpu]):    
        for i_net, net in enumerate(local_algo_nets[i_algo]):
            n += len(local_algo_batch_sizes_list[i_algo][i_net])
    for i_coreF, coreF in enumerate(gpu_coreFs[i_gpu]):  
        for i_net, net in enumerate(local_algo_nets[i_algo]):
            for i_batch, batch in enumerate(local_algo_batch_sizes_list[i_algo][i_net]):
                energys_respect_coreF = np.array(data.get_energy_respect(gpu, algo, 'caffe', net, batch, coreF, gpu_memFs[i_gpu]), np.dtype('float'))
                energys_sqrt = np.power(energys_respect_coreF, 1/n)
                results = results * energys_sqrt

    return results

# ========================================================
# ========================================================
# ========================================================
# ========================================================
# ========================================================
def get_conv_2080_geo_mean_power_memF(gpu, algo):
    gpu = 'gtx2080ti'
    i_algo = algo_index[algo]
    i_gpu = gpu_index[gpu]

    local_algo_batch_sizes_list = gtx_algo_batch_sizes_list
    local_algo_nets = gtx_algo_nets
    results = np.array([1, 1, 1, 1, 1, 1], np.dtype('float'))

    n = 0
    for i_memF, memF in enumerate(gpu_memFs[i_gpu]): 
        for i_net, net in enumerate(local_algo_nets[i_algo]):
            n += len(local_algo_batch_sizes_list[i_algo][i_net])
    for i_memF, memF in enumerate(gpu_memFs[i_gpu]): 
        for i_net, net in enumerate(local_algo_nets[i_algo]):
            for i_batch, batch in enumerate(local_algo_batch_sizes_list[i_algo][i_net]):
                powers_respect_coreF = np.array(data.get_power_respect(gpu, algo, 'caffe', net, batch, gpu_coreFs[i_gpu], memF), np.dtype('float'))
                powers_sqrt = np.power(powers_respect_coreF, 1/n)
                results = results * powers_sqrt

    return results

def get_conv_2080_geo_mean_perf_memF(gpu, algo):
    gpu = 'gtx2080ti'
    i_algo = algo_index[algo]
    i_gpu = gpu_index[gpu]

    local_algo_batch_sizes_list = gtx_algo_batch_sizes_list
    local_algo_nets = gtx_algo_nets
    results = np.array([1, 1, 1, 1, 1, 1], np.dtype('float'))

    n = 0
    for i_memF, memF in enumerate(gpu_memFs[i_gpu]): 
        for i_net, net in enumerate(local_algo_nets[i_algo]):
            n += len(local_algo_batch_sizes_list[i_algo][i_net])
    for i_memF, memF in enumerate(gpu_memFs[i_gpu]): 
        for i_net, net in enumerate(local_algo_nets[i_algo]):
            for i_batch, batch in enumerate(local_algo_batch_sizes_list[i_algo][i_net]):
                perfs_respect_coreF = np.array(data.get_image_perf_respect(gpu, algo, 'caffe', net, batch, gpu_coreFs[i_gpu], memF), np.dtype('float'))
                perfs_sqrt = np.power(perfs_respect_coreF, 1/n)
                results = results * perfs_sqrt

    return results

def get_conv_2080_geo_mean_energy_memF(gpu, algo):
    gpu = 'gtx2080ti'
    i_algo = algo_index[algo]
    i_gpu = gpu_index[gpu]

    local_algo_batch_sizes_list = gtx_algo_batch_sizes_list
    local_algo_nets = gtx_algo_nets
    results = np.array([1, 1, 1, 1, 1, 1], np.dtype('float'))

    n = 0
    for i_memF, memF in enumerate(gpu_memFs[i_gpu]): 
        for i_net, net in enumerate(local_algo_nets[i_algo]):
            n += len(local_algo_batch_sizes_list[i_algo][i_net])
    for i_memF, memF in enumerate(gpu_memFs[i_gpu]): 
        for i_net, net in enumerate(local_algo_nets[i_algo]):
            for i_batch, batch in enumerate(local_algo_batch_sizes_list[i_algo][i_net]):
                energys_respect_coreF = np.array(data.get_energy_respect(gpu, algo, 'caffe', net, batch, gpu_coreFs[i_gpu], memF), np.dtype('float'))
                energys_sqrt = np.power(energys_respect_coreF, 1/n)
                results = results * energys_sqrt

    return results


















# ========================================================
# ========================================================
# ========================================================
# ========================================================
# ========================================================
def get_2080_conv_memF_geo_mean_perf_and_power(algo, net, batch_size):
    gpu = "gtx2080ti"
    i_algo = algo_index[algo]
    i_gpu = gpu_index[gpu]

    local_algo_batch_sizes_list = algo_batch_sizes_list
    local_algo_nets = algo_nets

    if gpu == 'gtx2080ti':
        local_algo_batch_sizes_list = gtx_algo_batch_sizes_list
        local_algo_nets = gtx_algo_nets
        perf_results = np.array([1, 1, 1, 1, 1, 1], np.dtype('float'))
        power_results = np.array([1, 1, 1, 1, 1, 1], np.dtype('float'))
    else:
        perf_results = np.array([1, 1, 1, 1, 1, 1, 1], np.dtype('float'))
        power_results = np.array([1, 1, 1, 1, 1, 1, 1], np.dtype('float'))

    n = 0
    n = len(gtx2080ti_memF)

    for memF in gtx2080ti_memF:
        perfs_respect_coreF = np.array(data.get_image_perf_respect(gpu, algo, 'caffe', net, batch_size, gpu_coreFs[i_gpu], memF), np.dtype('float'))
        perfs_sqrt = np.power(perfs_respect_coreF, 1/n)
        perf_results = perf_results * perfs_sqrt

        powers_respect_coreF = np.array(data.get_power_respect(gpu, algo, 'caffe', net, batch_size, gpu_coreFs[i_gpu], memF), np.dtype('float'))
        powers_sqrt = np.power(powers_respect_coreF, 1/n)
        power_results = power_results * powers_sqrt


    return power_results, perf_results

def get_2080_conv_coreF_geo_mean_energy(algo, net, batch_size):
    gpu = "gtx2080ti"
    i_algo = algo_index[algo]
    i_gpu = gpu_index[gpu]

    local_algo_batch_sizes_list = gtx_algo_batch_sizes_list
    local_algo_nets = gtx_algo_nets
    perf_results = np.array([1, 1, 1, 1], np.dtype('float'))
    power_results = np.array([1, 1, 1, 1], np.dtype('float'))


    n = 0
    n = len(gpu_coreFs[i_gpu])

    for coreF in gpu_coreFs[i_gpu]:
        perfs_respect_coreF = np.array(data.get_image_perf_respect(gpu, algo, 'caffe', net, batch_size, coreF, gpu_memFs[i_gpu]), np.dtype('float'))
        perfs_sqrt = np.power(perfs_respect_coreF, 1/n)
        perf_results = perf_results * perfs_sqrt

        powers_respect_coreF = np.array(data.get_power_respect(gpu, algo, 'caffe', net, batch_size, coreF, gpu_memFs[i_gpu]), np.dtype('float'))
        powers_sqrt = np.power(powers_respect_coreF, 1/n)
        power_results = power_results * powers_sqrt


    return power_results, perf_results






# ===================================draw ==============================================================

def draw_conv_energy(
    gpu, save=False):
    
    i_gpu = gpu_index[gpu]
    gpu_coreF = gpu_coreFs[i_gpu]
    

    #==============
    fig = plt.figure(figsize=(fig_x, fig_y))
    ax1 = fig.add_subplot(1, 1, 1)
    # ax2 = ax1.twinx()
    bar_i_width = 0

    bar_n = len(algos)
    bar_width = bar_total_width / bar_n
    if gpu == 'gtx2080ti':
        line_x = np.array([100, 200, 300, 400, 500, 600], np.dtype('int32'))
    else:
        line_x = np.array([100, 200, 300, 400, 500, 600, 700], np.dtype('int32'))
    # bar_x = line_x - (bar_total_width - bar_width) / 2

    #==============
    bars = []
    lines = []
    # max_power = 0
    max_energy = 0

    # for algo in algos:
    #     i_algo = algo_index[algo]

    #     powers_respect_coreF = get_gpu_algo_geo_mean_power(gpu, algo)
    #     power_label_name = '{0} Power'.format(algo)

    #     max_power = max(powers_respect_coreF) if max_power < max(powers_respect_coreF) else max_power
    #     bars.append(ax1.bar(bar_x + bar_i_width * bar_width, powers_respect_coreF, color=bar_colors[i_algo],
    #         width=bar_width, label=power_label_name))
    #     bar_i_width += 1


    for algo in algos:
        i_algo = algo_index[algo]
        energy_respect_coreF = get_gpu_algo_geo_mean_energy(gpu, algo)
        energy_label_name = '{0} Energy'.format(algo)
        max_energy = max(energy_respect_coreF) if max_energy < max(energy_respect_coreF) else max_energy
        lines.append(ax1.plot(line_x, energy_respect_coreF, color=plot_colors[i_algo], marker=markers[i_algo],
            linewidth=2, label=energy_label_name))    

    box = ax1.get_position()
    ax1.set_position([box.x0, box.y0, box.width, box.height * 0.8])

    # bars = list(bars)
    lines = sum(lines, [])
    legend_elements = []
    for i in range(len(lines)):
        # legend_elements.append(bars[i])
        legend_elements.append(lines[i])

    # print legend_elements
    ax1.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, 1.14), ncol=len(lines), prop=font_legend)
    #ax2.legend(loc='upper center', ncol=4, prop=font_legend)
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
    # ax2.set_yticklabels(np.linspace(0, max_energy * y_max_ratio, num=y_axis_interval).astype(np.int32), fontsize=size_ticks)
    ax1.grid(True, linestyle="--", color="0.5", linewidth=1)

    # ax2.set_ylabel("%s (J/image)" % 'energy', size=size_labels)

    ax1.set_xlabel("%s (Hz)" % 'coreF', size=size_labels)
    ax1.set_ylabel("%s (J/image)" % 'energy', size=size_labels)
    ax1.set_xticks(line_x)
    ax1.set_xticklabels([int(coreF) for coreF in gpu_coreF], size=size_ticks)

    # plt.xticks([int(coreF) for coreF in gpu_coreF], size=ticks_size)
    # plt.title("config [{0}, {1}], varient {2} ".format('net', 'batch', 'coreF'))
    if save == True:
        save_file_name = 'conv_energy_{0}'.format(gpu)
        plt.savefig(os.path.join(OUTPUT_PATH, '%s.pdf' % save_file_name), bbox_inches='tight')
        plt.savefig(os.path.join(OUTPUT_PATH, '%s.png' % save_file_name), bbox_inches='tight')
    else:
        plt.show()









def draw_conv_power_perf(
    gpu, save=False):

    i_gpu = gpu_index[gpu]
    gpu_coreF = gpu_coreFs[i_gpu]

    #==============
    fig = plt.figure(figsize=(fig_x, fig_y))
    ax1 = fig.add_subplot(1, 1, 1)
    ax2 = ax1.twinx()
    bar_i_width = 0

    bar_n = len(algos)
    bar_width = bar_total_width / bar_n
    if gpu == 'gtx2080ti':
        line_x = np.array([100, 200, 300, 400, 500, 600], np.dtype('int32'))        
    else:
        line_x = np.array([100, 200, 300, 400, 500, 600, 700], np.dtype('int32'))
    bar_x = line_x - (bar_total_width - bar_width) / 2

    #==============
    bars = []
    lines = []
    max_power = 0
    max_image_perf = 0

    for algo in algos:
        i_algo = algo_index[algo]

        powers_respect_coreF = get_gpu_algo_geo_mean_power(gpu, algo)
        power_label_name = '{0} Power'.format(algo)

        max_power = max(powers_respect_coreF) if max_power < max(powers_respect_coreF) else max_power
        bars.append(ax1.bar(bar_x + bar_i_width * bar_width, powers_respect_coreF, color=bar_colors[i_algo],
            width=bar_width, label=power_label_name))
        bar_i_width += 1

    for algo in algos:
        i_algo = algo_index[algo]

        image_perf_respect_coreF = get_gpu_algo_geo_mean_perf(gpu, algo)
        max_image_perf = max(image_perf_respect_coreF) if max_image_perf < max(image_perf_respect_coreF) else max_image_perf

        energy_label_name = '{0} image_per'.format(algo)
        lines.append(ax2.plot(line_x, image_perf_respect_coreF, color=plot_colors[i_algo], marker=markers[i_algo],
            linewidth=2, label=energy_label_name))
    
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
        save_file_name = 'conv_power_perf_{0}'.format(gpu)
        plt.savefig(os.path.join(OUTPUT_PATH, '%s.pdf' % save_file_name), bbox_inches='tight')
        plt.savefig(os.path.join(OUTPUT_PATH, '%s.png' % save_file_name), bbox_inches='tight')
    else:
        plt.show()


#====================================================================
#==============================================================================

#====================================================================
#==============================================================================

#====================================================================
#==============================================================================

#====================================================================
#==============================================================================

#====================================================================
#==============================================================================

def draw_conv_2080_energy_memF(save=False):
    gpu = 'gtx2080ti'
    i_gpu = gpu_index[gpu]
    gpu_coreF = gpu_coreFs[i_gpu]
    gpu_memF = gpu_memFs[i_gpu]

    #==============
    fig = plt.figure(figsize=(fig_x, fig_y))
    ax1 = fig.add_subplot(1, 1, 1)
    # ax2 = ax1.twinx()
    # bar_i_width = 0

    # bar_n = len(algos)
    # bar_width = bar_total_width / bar_n

    line_x = np.array([100, 200, 300, 400], np.dtype('int32'))

    bars = []
    lines = []
    max_energy = 0

    for algo in algos:
        i_algo = algo_index[algo]
        energy_respect_coreF = get_conv_2080_mean_energy_coreF(gpu, algo)
        energy_label_name = '{0} Energy'.format(algo)
        max_energy = max(energy_respect_coreF) if max_energy < max(energy_respect_coreF) else max_energy
        lines.append(ax1.plot(line_x, energy_respect_coreF, color=plot_colors[i_algo], marker=markers[i_algo],
            linewidth=2, label=energy_label_name))

    box = ax1.get_position()
    ax1.set_position([box.x0, box.y0, box.width, box.height * 0.8])

    # bars = list(bars)
    lines = sum(lines, [])
    legend_elements = []
    for i in range(len(lines)):
        # legend_elements.append(bars[i])
        legend_elements.append(lines[i])

    # print legend_elements
    ax1.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, 1.14), ncol=len(lines), prop=font_legend)
    #ax2.legend(loc='upper center', ncol=4, prop=font_legend)
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
    # ax2.set_yticklabels(np.linspace(0, max_energy * y_max_ratio, num=y_axis_interval).astype(np.int32), fontsize=size_ticks)
    ax1.grid(True, linestyle="--", color="0.5", linewidth=1)

    # ax2.set_ylabel("%s (J/image)" % 'energy', size=size_labels)

    ax1.set_xlabel("%s (Hz)" % 'coreF', size=size_labels)
    ax1.set_ylabel("%s (J/image)" % 'energy', size=size_labels)
    ax1.set_xticks(line_x)
    ax1.set_xticklabels([int(memF) for memF in gpu_memF], size=size_ticks)

    # plt.xticks([int(coreF) for coreF in gpu_coreF], size=ticks_size)
    # plt.title("config [{0}, {1}], varient {2} ".format('net', 'batch', 'coreF'))
    if save == True:
        save_file_name = 'conv_energy_{0}_memF'.format(gpu)
        plt.savefig(os.path.join(OUTPUT_PATH, '%s.pdf' % save_file_name), bbox_inches='tight')
        plt.savefig(os.path.join(OUTPUT_PATH, '%s.png' % save_file_name), bbox_inches='tight')
    else:
        plt.show()


def draw_conv_2080_power_perf_memF(save=False):
    gpu = 'gtx2080ti'
    i_gpu = gpu_index[gpu]
    gpu_coreF = gpu_coreFs[i_gpu]
    gpu_memF = gpu_memFs[i_gpu]
    #==============
    fig = plt.figure(figsize=(fig_x, fig_y))
    ax1 = fig.add_subplot(1, 1, 1)
    ax2 = ax1.twinx()
    bar_i_width = 0

    bar_n = len(algos)
    bar_width = bar_total_width / bar_n

    line_x = np.array([100, 200, 300, 400], np.dtype('int32'))        

    bar_x = line_x - (bar_total_width - bar_width) / 2

    #==============
    bars = []
    lines = []
    max_power = 0
    max_image_perf = 0

    for algo in algos:
        i_algo = algo_index[algo]

        powers_respect_coreF = get_conv_2080_mean_power_coreF(gpu, algo)
        power_label_name = '{0} Power'.format(algo)

        max_power = max(powers_respect_coreF) if max_power < max(powers_respect_coreF) else max_power
        bars.append(ax1.bar(bar_x + bar_i_width * bar_width, powers_respect_coreF, color=bar_colors[i_algo],
            width=bar_width, label=power_label_name))
        bar_i_width += 1

    for algo in algos:
        i_algo = algo_index[algo]

        image_perf_respect_coreF = get_conv_2080_mean_perf_coreF(gpu, algo)
        max_image_perf = max(image_perf_respect_coreF) if max_image_perf < max(image_perf_respect_coreF) else max_image_perf

        energy_label_name = '{0} image_per'.format(algo)
        lines.append(ax2.plot(line_x, image_perf_respect_coreF, color=plot_colors[i_algo], marker=markers[i_algo],
            linewidth=2, label=energy_label_name))
    
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
    ax1.set_xticklabels([int(memF) for memF in gpu_memF], size=size_ticks)
    # plt.xticks([int(coreF) for coreF in gpu_coreF])
    # plt.title("config [{0}, {1}], varient {2} ".format('net', 'batch', 'coreF'))
    if save == True:
        save_file_name = 'conv_power_perf_{0}_memF'.format(gpu)
        plt.savefig(os.path.join(OUTPUT_PATH, '%s.pdf' % save_file_name), bbox_inches='tight')
        plt.savefig(os.path.join(OUTPUT_PATH, '%s.png' % save_file_name), bbox_inches='tight')
    else:
        plt.show()





#====================================================================
#==============================================================================

#====================================================================
#==============================================================================

#====================================================================
#==============================================================================

#====================================================================
#==============================================================================

#====================================================================
#==============================================================================

def draw_conv_2080_energy_coreF(save=False):
    gpu = 'gtx2080ti'
    i_gpu = gpu_index[gpu]
    gpu_coreF = gpu_coreFs[i_gpu]
    

    #==============
    fig = plt.figure(figsize=(fig_x, fig_y))
    ax1 = fig.add_subplot(1, 1, 1)
    # ax2 = ax1.twinx()
    bar_i_width = 0

    bar_n = len(algos)
    bar_width = bar_total_width / bar_n
    if gpu == 'gtx2080ti':
        line_x = np.array([100, 200, 300, 400, 500, 600], np.dtype('int32'))
    else:
        line_x = np.array([100, 200, 300, 400, 500, 600, 700], np.dtype('int32'))
    bars = []
    lines = []
    max_energy = 0

    for algo in algos:
        i_algo = algo_index[algo]
        energy_respect_coreF = get_conv_2080_geo_mean_energy_memF(gpu, algo)
        energy_label_name = '{0} Energy'.format(algo)
        max_energy = max(energy_respect_coreF) if max_energy < max(energy_respect_coreF) else max_energy
        lines.append(ax1.plot(line_x, energy_respect_coreF, color=plot_colors[i_algo], marker=markers[i_algo],
            linewidth=2, label=energy_label_name))

    box = ax1.get_position()
    ax1.set_position([box.x0, box.y0, box.width, box.height * 0.8])

    # bars = list(bars)
    lines = sum(lines, [])
    legend_elements = []
    for i in range(len(lines)):
        # legend_elements.append(bars[i])
        legend_elements.append(lines[i])

    # print legend_elements
    ax1.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, 1.14), ncol=len(lines), prop=font_legend)
    #ax2.legend(loc='upper center', ncol=4, prop=font_legend)
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
    # ax2.set_yticklabels(np.linspace(0, max_energy * y_max_ratio, num=y_axis_interval).astype(np.int32), fontsize=size_ticks)
    ax1.grid(True, linestyle="--", color="0.5", linewidth=1)

    # ax2.set_ylabel("%s (J/image)" % 'energy', size=size_labels)

    ax1.set_xlabel("%s (Hz)" % 'coreF', size=size_labels)
    ax1.set_ylabel("%s (J/image)" % 'energy', size=size_labels)
    ax1.set_xticks(line_x)
    ax1.set_xticklabels([int(coreF) for coreF in gpu_coreF], size=size_ticks)

    # plt.xticks([int(coreF) for coreF in gpu_coreF], size=ticks_size)
    # plt.title("config [{0}, {1}], varient {2} ".format('net', 'batch', 'coreF'))
    if save == True:
        save_file_name = 'conv_energy_{0}_coreF'.format(gpu)
        plt.savefig(os.path.join(OUTPUT_PATH, '%s.pdf' % save_file_name), bbox_inches='tight')
        plt.savefig(os.path.join(OUTPUT_PATH, '%s.png' % save_file_name), bbox_inches='tight')
    else:
        plt.show()


def draw_conv_2080_power_perf_coreF(save=False):
    gpu = 'gtx2080ti'
    i_gpu = gpu_index[gpu]
    gpu_coreF = gpu_coreFs[i_gpu]

    #==============
    fig = plt.figure(figsize=(fig_x, fig_y))
    ax1 = fig.add_subplot(1, 1, 1)
    ax2 = ax1.twinx()
    bar_i_width = 0

    bar_n = len(algos)
    bar_width = bar_total_width / bar_n
    if gpu == 'gtx2080ti':
        line_x = np.array([100, 200, 300, 400, 500, 600], np.dtype('int32'))        
    else:
        line_x = np.array([100, 200, 300, 400, 500, 600, 700], np.dtype('int32'))
    bar_x = line_x - (bar_total_width - bar_width) / 2

    #==============
    bars = []
    lines = []
    max_power = 0
    max_image_perf = 0

    for algo in algos:
        i_algo = algo_index[algo]

        powers_respect_coreF = get_conv_2080_geo_mean_power_memF(gpu, algo)
        power_label_name = '{0} Power'.format(algo)

        max_power = max(powers_respect_coreF) if max_power < max(powers_respect_coreF) else max_power
        bars.append(ax1.bar(bar_x + bar_i_width * bar_width, powers_respect_coreF, color=bar_colors[i_algo],
            width=bar_width, label=power_label_name))
        bar_i_width += 1

    for algo in algos:
        i_algo = algo_index[algo]

        image_perf_respect_coreF = get_conv_2080_geo_mean_perf_memF(gpu, algo)
        max_image_perf = max(image_perf_respect_coreF) if max_image_perf < max(image_perf_respect_coreF) else max_image_perf

        energy_label_name = '{0} image_per'.format(algo)
        lines.append(ax2.plot(line_x, image_perf_respect_coreF, color=plot_colors[i_algo], marker=markers[i_algo],
            linewidth=2, label=energy_label_name))
    
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
        save_file_name = 'conv_power_perf_{0}_coreF'.format(gpu)
        plt.savefig(os.path.join(OUTPUT_PATH, '%s.pdf' % save_file_name), bbox_inches='tight')
        plt.savefig(os.path.join(OUTPUT_PATH, '%s.png' % save_file_name), bbox_inches='tight')
    else:
        plt.show()















































def draw_energy_fix_gpu_algo_config_net_varient_frequency(
    gpu, save=False):
    
    i_gpu = gpu_index[gpu]
    gpu_coreF = gpu_coreFs[i_gpu]

    i_algo = algo_index['ipc_gemm']
    i_gpu = gpu_index[gpu]

    local_algo_nets = algo_nets
    if gpu == 'gtx2080ti':
        local_algo_nets = gtx_algo_nets

    #==============
    fig = plt.figure(figsize=(fig_x, fig_y))
    ax1 = fig.add_subplot(1, 1, 1)
    # bar_i_width = 0

    # bar_n = len(algos)
    # bar_width = bar_total_width / bar_n
    if gpu == 'gtx2080ti':
        line_x = np.array([100, 200, 300, 400, 500, 600], np.dtype('int32'))        
    else:
        line_x = np.array([100, 200, 300, 400, 500, 600, 700], np.dtype('int32'))
    # bar_x = line_x - (bar_total_width - bar_width) / 2

    #==============
    bars = []
    lines = []
    max_power = 0
    max_energy = 0

    for i_net, net in enumerate(local_algo_nets[i_algo]):
        energy_respect_coreF = get_fix_gpu_algo_config_net_geo_mean_energy(gpu, 'ipc_gemm', net)
        energy_label_name = '{0} Energy'.format(net)
        max_energy = max(energy_respect_coreF) if max_energy < max(energy_respect_coreF) else max_energy
        # ax1.plot(line_x, energy_respect_coreF, color=plot_colors[i_algo], marker=markers[i_algo],
        #     linewidth=2, label=energy_label_name)
        lines.append(ax1.plot(line_x, energy_respect_coreF, color=plot_colors[i_net], marker=markers[i_net],
            linewidth=2, label=energy_label_name))
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
        save_file_name = 'power&perf_ex1_{0}_{1}_{2}'.format(gpu, algo, net)
        plt.savefig(os.path.join(OUTPUT_PATH, '%s.pdf' % save_file_name), bbox_inches='tight')
        plt.savefig(os.path.join(OUTPUT_PATH, '%s.png' % save_file_name), bbox_inches='tight')
    else:
        plt.show()



# for i_gpu, gpu in enumerate(gpus):

# #     draw_power_energy_fix_gpu_config_algos_varient_frequency(gpu)
#     draw_power_perf_fix_gpu_config_algos_varient_frequency(gpu)
























draw_conv_power_perf('p100', save=True)
draw_conv_power_perf('v100', save=True)
draw_conv_energy('p100', save=True)
draw_conv_energy('v100', save=True)
draw_conv_2080_energy_memF(save=True)
draw_conv_2080_power_perf_memF(save=True)
draw_conv_2080_energy_coreF(save=True)
draw_conv_2080_power_perf_coreF(save=True)

