"""Numeric summary pipeline extracted from eda/data_prepare_summary.ipynb."""
from src.config import DATA_ROOT, RAW_ROOT
import os
import sys
import pickle
import numpy as np
from datetime import datetime, timedelta
from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt

def time_to_stamp(date_time):
    return datetime.strptime(date_time, '%m/%d/%Y %H:%M')

def read_record(facility, id_):
    with open((DATA_ROOT + '/cleaned_records_{}/{}').format(facility, id_), 'rb') as f:
        record = pickle.load(f)
    return record
sum_keys = ['MAX_BLOOD_FLOW', 'MAX_DIALYSATE_FLOW', 'DIALYSATE_TEMP', 'DIAL_DURATION', 'PRE_DIAL_BP_SYST_SIT', 'POST_DIAL_BP_SYST_SIT', 'LOWEST_BP_SYST', 'HIGHEST_BP_SYST', 'ESTIMATED_DRY_WEIGHT', 'LAST_DIAL_WEIGHT_CHANGE', 'HOURS_LAST_DIAL', 'TOTAL_VOL_REMOVED', 'NET_VOL_REMOVED', 'VOL_INFUSED', 'PRE_DIAL_TEMP', 'POST_DIAL_TEMP', 'PRE_DIAL_WEIGHT_KG', 'POST_DIAL_WEIGHT_KG', 'BMI', 'TBW', 'LBW', 'BSA']
denos = {'MAX_BLOOD_FLOW': 512, 'MAX_DIALYSATE_FLOW': 512, 'DIALYSATE_TEMP': 64, 'DIAL_DURATION': 256, 'PRE_DIAL_BP_SYST_SIT': 128, 'POST_DIAL_BP_SYST_SIT': 128, 'LOWEST_BP_SYST': 128, 'HIGHEST_BP_SYST': 128, 'ESTIMATED_DRY_WEIGHT': 64, 'LAST_DIAL_WEIGHT_CHANGE': 2, 'HOURS_LAST_DIAL': 128, 'TOTAL_VOL_REMOVED': 2048, 'NET_VOL_REMOVED': 2048, 'VOL_INFUSED': 512, 'PRE_DIAL_TEMP': 128, 'POST_DIAL_TEMP': 128, 'PRE_DIAL_WEIGHT_KG': 64, 'POST_DIAL_WEIGHT_KG': 64, 'BMI': 32, 'TBW': 64, 'LBW': 64, 'BSA': 2}
sum_keys = sum_keys[:18]
denos = {k: denos[k] for k in sum_keys}
dia_sums = list()
root_path = os.path.join(RAW_ROOT, 'session_summaries')

def match_lab_test_and_dia_sum(id_, facility='all'):
    record = read_record(facility, id_)
    for s in record['sessions']:
        s_time = time_to_stamp(record['sessions'][s][0]['DATETIME'])
        curr_key = '{}_{}'.format(id_, record['sessions'][s][0]['SESSION_ID'])
        pad_clip_to = 64
        curr_sums = list()
        decays = list()
        if id_to_dia_sum.get(id_) is None:
            curr_sums = [[0 for _ in range(len(sum_keys))] for _ in range(pad_clip_to)]
            decays = [0.0 for _ in range(pad_clip_to)]
        else:
            for sum_i in id_to_dia_sum[id_]:
                if sum_times[sum_i] < s_time:
                    if sum_times[sum_i] == s_time:
                        print(curr_key, 'have current info.')
                    curr_sums.append(summaries[sum_i])
                    diff_days = (s_time - sum_times[sum_i]).days
                    decays.append(0.997 ** diff_days)
            if len(curr_sums) < pad_clip_to:
                curr_sums = [[0 for _ in range(len(sum_keys))] for _ in range(pad_clip_to - len(curr_sums))] + curr_sums
                decays = [0.0 for _ in range(pad_clip_to - len(decays))] + decays
        curr_summaries = {'summaries': curr_sums[-pad_clip_to:], 'decays': decays[-pad_clip_to:]}
        save_name = (DATA_ROOT + '/clean_summaries_{}/{}').format(facility, curr_key)
        with open(save_name, 'wb') as f:
            pickle.dump(curr_summaries, f)
if __name__ == '__main__':
    for fn in os.listdir(root_path):
        if fn[0] == '.':
            continue
        dia_sum = pd.read_csv(os.path.join(root_path, fn))
        dia_sum = dia_sum.dropna(subset=sum_keys)
        for k in denos:
            dia_sum[k] = dia_sum[k] / denos[k]
        dia_sums.append(dia_sum)
    summaries = list()
    sum_times = list()
    i_ = 1
    id_to_dia_sum = dict()
    for dia_sum in dia_sums:
        for (index, row) in tqdm(dia_sum.iterrows()):
            id_ = row['PATIENT_ID']
            try:
                ts = time_to_stamp(row['DIAL_STOP_TIME'])
            except:
                continue
            summaries.append(row[sum_keys].values)
            sum_times.append(ts)
            if id_to_dia_sum.get(id_) is None:
                id_to_dia_sum[id_] = [i_]
            else:
                id_to_dia_sum[id_].append(i_)
            i_ += 1
    summaries = [[0 for _ in range(len(summaries[-1]))]] + summaries
    sum_times = [timedelta(days=0)] + sum_times
    print('Num Patients:', len(id_to_dia_sum))
    print(np.array(summaries).shape, np.array(summaries).dtype)
    print(len(sum_times))
    for i in tqdm(range(1, len(summaries))):
        summaries[i] = summaries[i].tolist()
    print(np.array(summaries).shape, np.array(summaries).dtype)
    os.makedirs(os.path.join(DATA_ROOT, 'clean_summaries_all'), exist_ok=True)
    ids = sorted(os.listdir((DATA_ROOT + '/cleaned_records_{}').format('all')))
    for id_ in tqdm(ids):
        match_lab_test_and_dia_sum(int(id_))
