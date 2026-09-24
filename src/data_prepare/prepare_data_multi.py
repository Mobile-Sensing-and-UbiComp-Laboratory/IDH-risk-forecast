from src.config import DATA_ROOT, FIGURE_ROOT, SEQ_LENGTH, SAMPLE_EVERY
import os
import sys
import pickle
import numpy as np
from datetime import datetime, timedelta
from tqdm import tqdm
import pandas as pd

# denominator, the closest power of 2
DENOS = {
    "SYSBP": 128, 
    "DIASBP": 64,
    "PULSE": 64,
    "PATIENT_TEMP": 128,
    "BLOOD_FLOW": 256,
    "DIALYSATE_FLOW": 512,
    "DIALYSATE_TEMP": 32,
    "UFR": 1024,
    "NET_VOLUME_REMOVED": 1
}

# utils
def time_to_stamp(date_time):
    return datetime.strptime(date_time, '%m/%d/%Y %H:%M')

def save_record(path, record):
    with open(path, 'wb') as f:
        pickle.dump(record, f)

def read_record(facility, id_, data_root=DATA_ROOT):
    # facility id, patient id
    with open((data_root + '/cleaned_records_{}/{}').format(facility, id_), 'rb') as f:
        record = pickle.load(f)
    return record
        
def is_valid_val(val):
    return not (np.isnan(val) or np.isposinf(val) or np.isneginf(val) or np.isinf(val))

def is_valid_arr(arr):
    return not (np.isnan(arr).any() or np.isposinf(arr).any() or np.isneginf(arr).any() or np.isinf(arr).any())

def is_valid(sample):
    valid = True
    i = 0
    for arr in sample["body"]:
        i += 1
        if not is_valid_arr(arr):
            valid = False
    i = 0
    if valid:
        for arr in sample["machine"]:
            i += 1
            if not is_valid_arr(arr):
                valid = False
    return valid

# Imputation, Interpolation, and Resampling
def impute_with_prev(vals, key, timestamp, running_vals, fix_val=None):
    # take the either fix value or running average across previous session
    if not is_valid_val(vals[0]):
        if fix_val is not None:
            vals[0] = fix_val
        else:
            if len(running_vals[key]) > 0:
                vals[0] = np.mean(running_vals[key])
            else:
                if key == 'PATIENT_TEMP':
                    vals[0] = 98

    # take previous valid value
    for i in range(1, len(vals)):
        if not is_valid_val(vals[i]):
            vals[i] = vals[i-1]
    return vals, timestamp

def step_wise_impute(vals, key, timestamps, seconds, pad_clip_to, running_vals):
    # impute
    vals, timestamps = impute_with_prev(vals, key, timestamps, running_vals, fix_val=0)
    vals = [v/DENOS[key] for v in vals]

    # interpolation
    if len(vals) == 1:
        interp_vals = vals
    else:
        interp_vals = list()
        curr_time = timestamps[0]
        i = 1
        while curr_time <= timestamps[-1] + timedelta(seconds=seconds//2):
            if curr_time < timestamps[i]:
                interp_vals.append(vals[i-1])
            else:
                interp_vals.append(vals[i])
                i += 1
            curr_time += timedelta(seconds=seconds)
    
    # pad or clip
    if len(interp_vals) < pad_clip_to:
        interp_vals = [0]*(pad_clip_to-len(interp_vals)) + interp_vals
    return np.array(interp_vals[-pad_clip_to:]).astype(np.float16)

def impute_and_resample(vals, key, timestamps, seconds, pad_clip_to, running_vals):
    '''
    vals: 2D, CxL
    timestamps: list(datetime obj)
    seconds: control sampling rate
    pad_clip_to: total length, pad if less, clip if exceed
    '''
    # impute
    vals, timestamps = impute_with_prev(vals, key, timestamps, running_vals)
    vals = [v/DENOS[key] for v in vals]

    # interpolation and resampling
    if len(vals) == 1:
        interp_vals = vals
    else:
        curr_time = timestamps[0]
        time_to_interp = list()
        while curr_time <= timestamps[-1] + timedelta(seconds=seconds//2):
            time_to_interp.append(curr_time)
            curr_time += timedelta(seconds=seconds)
        time_to_interp = [datetime.timestamp(t) for t in time_to_interp]

        origin_time = [datetime.timestamp(t) for t in timestamps]
        interp_vals = np.interp(time_to_interp, origin_time, vals).tolist()

    # pad or clip
    if len(interp_vals) < pad_clip_to:
        interp_vals = [0]*(pad_clip_to-len(interp_vals)) + interp_vals
    return np.array(interp_vals[-pad_clip_to:]).astype(np.float16)

def determine_IDH(
    curr_bp,
    next_bp,
    curr_bf,
    next_bf,
    prev_ufr,
    curr_ufr,
    next_ufr,
    curr_saline,
    next_saline,
    next_symptoms,
    curr_supine,
    curr_temp,
    prev_temp,
    net_volume_removed,
    volume_back
):
    labels = {
        "fall20": False,
        "fall30": False,
        "nadir90": False,
        "nadir100": False,
        "hemo": False,
        "kdoqi": False,
        # for acute intervention
        "acute_interv": {
            "ufr": curr_ufr,
            'ufr_diff': 0 ,
            "flow_temp": 0,
            "saline": False,
            "supine": False,
            "net_volume_removed": net_volume_removed,
            "volume_back": volume_back
        }
    }
    if curr_bp - next_bp >= 20:
        labels["fall20"] = True
    if curr_bp - next_bp >= 30:
        labels["fall30"] = True
    if next_bp < 90:
        labels["nadir90"] = True
    if next_bp < 100:
        labels["nadir100"] = True
    
    if next_bp < curr_bp:
        if next_bf < curr_bf or next_ufr < curr_ufr or next_saline:
            labels["hemo"] = True
    if next_symptoms:
        labels["kdoqi"] = True
    
    # for acute intervention
    if curr_ufr < prev_ufr:
        labels['acute_interv']['ufr_diff'] = prev_ufr - curr_ufr
    if curr_temp < prev_temp:
        labels['acute_interv']['flow_temp'] = prev_temp - curr_temp
    if curr_saline:
        labels["acute_interv"]["saline"] = True
    if curr_supine:
        labels["acute_interv"]["supine"] = True

    return labels

# Main functions
def get_one_session_samples(session_records, running_vals, note_to_label, sample_every=5):
    # process data
    samples = list()
    session_level_label = {
        "fall20": False,
        "fall30": False,
        "nadir90": False,
        "nadir100": False,
        "hemo": False,
        "kdoqi": False
    }
    for i in range(len(session_records)-1):
        # edge case cleanup
        try:
            session_records[i]["NET_VOLUME_REMOVED"] = float(session_records[i]["NET_VOLUME_REMOVED"])
        except:
            session_records[i]["NET_VOLUME_REMOVED"] = np.nan

        try:
            next_saline = note_to_label[session_records[i+1]["COMMENTS"]]["saline"]
        except:
            next_saline = False
        try:
            next_symptoms = note_to_label[session_records[i+1]["COMMENTS"]]["symptoms"]
        except:
            next_symptoms = False
        try:
            curr_supine = note_to_label[session_records[i]["COMMENTS"]]["supine"]
        except:
            curr_supine = False
        try:
            curr_saline = note_to_label[session_records[i]["COMMENTS"]]["saline"]
        except:
            curr_saline = False
        try:
            curr_volume_back = note_to_label[session_records[i]["COMMENTS"]]["volume_back"]
        except:
            curr_volume_back = False

        # determine if next measurement is IDH
        prev_temp = 0 if i == 0 else session_records[i-1]["DIALYSATE_TEMP"]
        prev_ufr = 0 if i == 0 else session_records[i-1]["UFR"]
        labels = determine_IDH(
            session_records[i]["SYSBP"], 
            session_records[i+1]["SYSBP"],
            session_records[i]["BLOOD_FLOW"], 
            session_records[i+1]["BLOOD_FLOW"],
            prev_ufr,
            session_records[i]["UFR"], 
            session_records[i+1]["UFR"],
            curr_saline,
            next_saline,
            next_symptoms,
            curr_supine,
            session_records[i]["DIALYSATE_TEMP"],
            prev_temp,
            session_records[i]["NET_VOLUME_REMOVED"],
            curr_volume_back
        )
        for label_key in session_level_label:
            session_level_label[label_key] = session_level_label[label_key] or labels[label_key]

        # imputation required setting
        timestamps = [time_to_stamp(session_records[j]["DATETIME"]) for j in range(i+2)] # include next one
        seconds = int(60*sample_every) # resampling every x minute(s)

        pad_clip_length_map = {
            1: 256,
            3: 128,
            5: 64,
            10: 32,
            15: 24
        }
        pad_clip_to = pad_clip_length_map[sample_every]

        # impute and resample
        bodies = list()
        next_measure = list()
        for key in [
            'SYSBP', 
            'DIASBP', 
            'PULSE', 
            'PATIENT_TEMP', 
            # Feature-set variant: include pressure channels; add normalization divisors and adjust model inputs.
            # 'VENOUS_PRESSURE', 
            # 'ARTERIAL_PRESSURE'
        ]:
            vals = [session_records[j][key] for j in range(i+2)] # include next one
            bodies.append(impute_and_resample(vals[:-1], key, timestamps[:-1], seconds, pad_clip_to, running_vals))

            # next measurement
            vals_im_ = impute_and_resample(vals[-3:], key, timestamps[-3:], seconds, pad_clip_to, running_vals)
            next_measure.append(vals_im_[-1])
        
        machines = list()
        for key in ['BLOOD_FLOW', 'DIALYSATE_FLOW', 'DIALYSATE_TEMP', 'UFR', 'NET_VOLUME_REMOVED']:
            vals = [session_records[j][key] for j in range(i+1)]
            machines.append(step_wise_impute(vals, key, timestamps[:-1], seconds, pad_clip_to, running_vals))

        # save record
        samples.append({
            "body": bodies, 
            "machine": machines,
            "acute_interv": labels["acute_interv"],
            "fall20": labels["fall20"],
            "fall30": labels["fall30"],
            "nadir90": labels["nadir90"],
            "nadir100": labels["nadir100"],
            "hemo": labels["hemo"],
            "kdoqi": labels["kdoqi"],
            "next_measure": next_measure
        })
    
    # collect values for calculating running means
    for key in [
        'SYSBP', 
        'DIASBP', 
        'PULSE', 
        'PATIENT_TEMP', 
    ]:
        running_vals[key] += [session_records[j][key] for j in range(len(session_records)) if is_valid_val(session_records[j][key])]
    
    # add the very initial padding
    samples.append({
        "body": np.zeros(np.array(samples[0]["body"]).shape).tolist(), 
        "machine": np.zeros(np.array(samples[0]["machine"]).shape).tolist(),
        "acute_interv": {
            "ufr": 0,
            "ufr_diff": 0, 
            "flow_temp": 0,
            "saline": False,
            "supine": False,
            "net_volume_removed": 0,
            "volume_back": False
        },
        # labels:
        "fall20": session_level_label["fall20"],
        "fall30": session_level_label["fall30"],
        "nadir90": session_level_label["nadir90"],
        "nadir100": session_level_label["nadir100"],
        "hemo": session_level_label["hemo"],
        "kdoqi": session_level_label["kdoqi"], 
        "next_measure": np.array(samples[0]["body"])[:, -1].astype(np.float16).tolist()
    })
    return samples, running_vals

def main(facility="all", sample_every=5, remark="", root_predix=DATA_ROOT):
    os.makedirs("{}/clean_samples_{}{}".format(root_predix, facility, remark), exist_ok=True)
    # sample_every: unit in minutes
    # remark: remark string for the root folder
    '''
    # origin structure of entire data
    id_to_record[row['PATIENT_ID']] = {
        'dob': datetime.strptime(row["DATE_OF_BIRTH"], '%m/%d/%Y %H:%M'),
        'race': row["RACE"],
        'sex': row['SEX'],
        'med_records': list(),
        'lab_test_records': list(),
        'sessions': dict() # map: session_id -> list(row)
    }
    '''

    nurse_note = pd.read_csv(os.path.join(root_predix, "saline_symptoms.csv"))
    note_to_label = dict()
    for index, row in nurse_note.iterrows():
        note_to_label[row["Note"]] = {
            "saline": row["Saline"],
            "symptoms": row["Symptoms"],
            "supine": row["Supine"],
            "volume_back": row["Volume_back"]
        }

    ids = os.listdir("{}/cleaned_records_{}".format(root_predix, facility))
    labels_count = {
        "fall20": 0,
        "fall30": 0,
        "nadir90": 0,
        "nadir100": 0,
        "hemo": 0,
        "kdoqi": 0
    }

    # Start processing records.
    for id_ in tqdm(ids):
        # load cleaned records
        try:
            record = read_record(facility, id_, data_root=root_predix)
        except:
            print("Record fetch error:", id_)
            continue
        '''
        each sample: {
            "body": list(physiological signals) C_b x L,
            "machine": list(machine parameters) C_m x L,
            "label": True/False, whether IDH happens in next measurement (within 2 hour)
        }
        '''

        # get samples
        # for mean imputation
        running_vals = {
            'SYSBP': list(), 
            'DIASBP': list(), 
            'PULSE': list(), 
            'PATIENT_TEMP': list(), 
        }

        invalid_session = 0
        total_session = 0
        for session_id in record["sessions"]:
            total_session += 1
            is_valid_session = False
            curr_samples, running_vals = get_one_session_samples(record["sessions"][session_id], running_vals, note_to_label, sample_every=sample_every)
            for i in range(len(curr_samples)):
                if is_valid(curr_samples[i]):
                    is_valid_session = True
                    for k in labels_count:
                        labels_count[k] += int(curr_samples[i][k])
                    fname = "{}/clean_samples_{}{}/{}_{}_{}".format(root_predix, facility, remark, id_, session_id, i)
                    save_record(fname, curr_samples[i]) # if not save, comment this
            if not is_valid_session:
                invalid_session += 1
        print(id_, "{}% Invalid Sessions.".format(round(100 * invalid_session/total_session)))
    
    # verbose
    print("label counts:", labels_count)

if __name__ == '__main__':
    # python3 -m src.data_prepare.prepare_data_multi all
    facility = sys.argv[1]
    # Sampling variant: save one-minute samples under a separate suffix; configure consumers to read that folder.
    # main(facility=str(facility), sample_every=1, remark="_1min", root_predix='data')
    main(facility=str(facility), sample_every=SAMPLE_EVERY, remark="", root_predix=DATA_ROOT)
