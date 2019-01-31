import os, sys
import argparse
import re
class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

parser = argparse.ArgumentParser(description='GPU measure')
parser.add_argument('-d', '--data-dir', default='gtx2080ti', type=str,
                    help='sssss')
args = parser.parse_args()

# ==============================re
core_re = re.compile(r'(?<=core)\-?\d*')
mem_re = re.compile(r'(?<=mem)\-?\d*')

base_coreF = 1950
base_memF = 6800

data_dir = args.data_dir + '_extract'
file_list = os.listdir(data_dir)

out_dir = args.data_dir + '_out'
if not os.path.exists(out_dir):
    os.makedirs(out_dir)

for i, input_file_name in enumerate(file_list):

    file_name = input_file_name.split('.')[0]
    file_parameters = file_name.split('_')
    if not ((file_parameters[-2] == 'perf') & (file_parameters[1] == 'giexec')):
        continue

    coreF = base_coreF + int(core_re.search(file_parameters[3]).group())
    memF = base_memF + int(mem_re.search(file_parameters[4]).group())
    # print(core_re.search(file_parameters[3]).group())
    # print(int(core_re.search(file_parameters[3]).group()))
    # print(coreF)
    # print(mem_re.search(file_parameters[4]).group())
    # print(int(mem_re.search(file_parameters[4]).group()))
    # print(memF)

    out_file_name = file_parameters[1] + '_' + file_parameters[2].split('-')[0] + '_' + \
                    file_parameters[2].split('-')[1] + '_' + str(coreF) + '_' + \
                    str(memF) + '_perf' + '.log'

    out_file_path = os.path.join(out_dir, out_file_name)
    out_file = open(out_file_path, 'w')

    input_file_path = os.path.join(data_dir, input_file_name)
    input_file_f = open(input_file_path, 'r')
    input_file = input_file_f.readlines()

    meas_perf = AverageMeter()

    print("file_name*** :  %s \n" % input_file_name)
    for i in range(len(input_file)):
        if (i > 0.3*len(input_file)) & (i < 0.6*len(input_file)):
            # data_line = 
            meas_perf.update(float(input_file[i].split()[5]))
    try:
        speed = 1 / ( (meas_perf.avg / 1000) / (int(file_parameters[2].split('-')[1])))
        out_file.writelines(str(meas_perf.avg) + ' ' + str(speed))
    except:
        pass

    input_file_f.close()
    out_file.close()

for i, input_file_name in enumerate(file_list):

    file_name = input_file_name.split('.')[0]
    file_parameters = file_name.split('_')
    if file_parameters[-2] != 'power':
        continue

    coreF = base_coreF + int(core_re.search(file_parameters[3]).group())
    memF = base_memF +  int(mem_re.search(file_parameters[4]).group())

    out_file_name = file_parameters[1] + '_' + file_parameters[2].split('-')[0] + '_' + \
                    file_parameters[2].split('-')[1] + '_' + str(coreF) + '_' + \
                    str(memF) + '_power' + '.log'

    out_file_path = os.path.join(out_dir, out_file_name)
    out_file = open(out_file_path, 'w')

    input_file_path = os.path.join(data_dir, input_file_name)
    input_file_f = open(input_file_path, 'r')
    input_file = input_file_f.readlines()

    meas_perf = AverageMeter()

    print("file_name*** :  %s \n" % input_file_name)
    for i in range(len(input_file)):
        if (i > 0.7*len(input_file)) & (i < 0.8*len(input_file)):
            # data_line = input_file[i].split()
            meas_perf.update(int(input_file[i].split()[-1]))

    out_file.writelines(str(meas_perf.avg))

    input_file_f.close()
    out_file.close()



