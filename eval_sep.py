from modules import read_meta
import numpy as np

dataset_path = 'E:/DCASE2018/eval'

eval_data = read_meta(dataset_path + '/DCASE2018-task5-eval.meta/DCASE2018-task5-eval/evaluation_setup/evaluate.txt')
eval_map_data = read_meta(dataset_path + '/DCASE2018-task5-eval.meta/DCASE2018-task5-eval/evaluation_setup/map.txt')

# known mic
km = open(dataset_path + '/DCASE2018-task5-eval.meta/DCASE2018-task5-eval/known_mic_meta.txt', 'w')
# unknown mic
um = open(dataset_path + '/DCASE2018-task5-eval.meta/DCASE2018-task5-eval/unknown_mic_meta.txt', 'w')

for i in range(len(eval_data)-1):

    if eval_map_data[i][0][14:15] == '1' or eval_map_data[i][0][14:15] == '2' or eval_map_data[i][0][14:15] == '3' \
            or eval_map_data[i][0][14:15] == '4':
        for j in range(len(eval_data)):
            if eval_map_data[i][1] == eval_data[j][0]:
                km.write(eval_data[j][0] + '\t' + eval_data[j][1] + '\t' + eval_data[j][2] + '\n')
                break
            else:
                continue
    else:
        for j in range(len(eval_data)):
            if eval_map_data[i][1] == eval_data[j][0]:
                um.write(eval_data[j][0] + '\t' + eval_data[j][1] + '\t' + eval_data[j][2] + '\n')
                break
            else:
                continue
