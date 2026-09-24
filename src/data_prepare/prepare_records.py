"""Core ingestion extracted from eda/data_prepare.ipynb."""
from src.config import DATA_ROOT, RAW_ROOT
import pandas as pd
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

import numpy as np
from tqdm import tqdm
import os
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import pickle

# Helper functions
s_to_t = lambda s: datetime.strptime(s, '%m/%d/%Y %H:%M')

def check_distribution(arr, title=""):
    if len(arr) == 0:
        print("Empty List.")
        return
    print("Min", np.min(arr))
    print("Max", np.max(arr))
    print("Mean", np.mean(arr))

    plt.clf()
    plt.hist(np.array(arr), bins=100, label=title)
    plt.legend()
    plt.show()

def fetch_demo_lab_med(
    demo_path=os.path.join(RAW_ROOT, "demographics.csv"),
    lab_path=os.path.join(RAW_ROOT, "lab_results.csv"),
    lab_code_path=os.path.join(RAW_ROOT, "lab_codes.tsv")
):
    # fetch demographic
    demo = pd.read_csv(demo_path)
    demo = demo[demo['DATE_OF_BIRTH'].notna()]
    year_now = datetime.now().year
    ages = list()
    for index, row in demo.iterrows():
        ages.append(year_now - datetime.strptime(row["DATE_OF_BIRTH"], '%m/%d/%Y %H:%M').year)
    check_distribution(ages, title="Age")
    
    # fetch lab test results
    lab_code = pd.read_csv(lab_code_path, sep='\t', header=None)
    # construct dictionary
    id_to_test = dict()
    for index, row in lab_code.iterrows():
        id_to_test[row[1]] = row[0]
    
    lab_record = pd.read_csv(lab_path)
    id_to_test_records = dict()
    print("Fetching Lab Test Record")
    for index, row in tqdm(lab_record.iterrows()):
        curr_record = {
            "timestamp":datetime.strptime(row["LAB_DATE_TIME"], '%m/%d/%Y %H:%M'),
            'result': row['LAB_TEST_RESULT'],
            'test_type': id_to_test[row['PROCEDURE_CODE']]
        }
        if id_to_test_records.get(row["PATIENT_ID"]) is None:
            id_to_test_records[row["PATIENT_ID"]] = [curr_record]
        else:
            id_to_test_records[row["PATIENT_ID"]].append(curr_record)
    
    # fetch medication record
    # TBD
    id_to_med_records = dict()
    
    # integrate
    id_to_record = dict()
    print("Integrating")
    for index, row in tqdm(demo.iterrows()):
        # general
        id_to_record[row['PATIENT_ID']] = {
            'dob': datetime.strptime(row["DATE_OF_BIRTH"], '%m/%d/%Y %H:%M'),
            'race': row["RACE"],
            'sex': row['SEX'],
            'med_records': list(),
            'lab_test_records': list(),
            'sessions': list()
        }

        # medication history
        if id_to_med_records.get(row['PATIENT_ID']) is None:
            pass
        else:
            id_to_record[row['PATIENT_ID']]["med_records"] = id_to_med_records[row['PATIENT_ID']]

        # lab test history
        if id_to_test_records.get(row['PATIENT_ID']) is None:
            print("No lab test record for", row['PATIENT_ID'])
        else:
            id_to_record[row['PATIENT_ID']]["lab_test_records"] = id_to_test_records[row['PATIENT_ID']]
        
    print("Number of Patients:", len(id_to_record.keys()))
    return id_to_record


def fetch_raw_dial_details(
    detail_path=os.path.join(RAW_ROOT, "session_details")
):
    sessions = dict()
    print("Fetching Raw Data")
    for f_ in tqdm(os.listdir(detail_path)):
        if f_[0] == ".":
            continue
        # fetch all rows
        dialysis_info = pd.read_csv(os.path.join(detail_path, f_), encoding='cp1252')
        for index, row in dialysis_info.iterrows():
            try:
                timestamp = datetime.strptime(row["DATETIME"], '%m/%d/%Y %H:%M')
            except:
                continue
            id_ = row["PATIENT_ID"]
            if sessions.get(id_) is None:
                sessions[id_] = dict()
            if sessions[id_].get(row["SESSION_ID"]) is None:
                sessions[id_][row["SESSION_ID"]] = [row]
            else:
                sessions[id_][row["SESSION_ID"]].append(row)
    # verbose      
    num_patient = 0
    num_sessions = 0
    for id_ in sessions:
        num_patient += 1
        for session_id in sessions[id_]:
            num_sessions += 1
    print("Raw Data")
    print("# patients", num_patient)
    print("# sessions", num_sessions)
    print("=======")
    return sessions
######## CLEAN ###########

def clean_dialysis_details(sessions):
    del_id = list()
    bp_diffs = list()
    print("Cleaning")
    for id_ in tqdm(sessions):
        del_session = list()
        for session_id in sessions[id_]:
            # remove NaN in following column
            checklist = ["SYSBP", "DIASBP", "PULSE"]
            del_detail_id = list()
            prev_row = None
            for row in sessions[id_][session_id]:
                if "C" in str(row["PATIENT_TEMP_UNITS"]):
                    row["PATIENT_TEMP"] = (float(row["PATIENT_TEMP"]) * 9/5) + 32

                has_nan = False
                for item in checklist:
                    if np.isnan(row[item]):
                        del_detail_id.append(row["SESSION_DETAIL_ID"])
                        has_nan = True
                        break
                if row["SYSBP"] < 30 or row["SYSBP"] > 400 or row["PULSE"] < 30 or row["PULSE"] > 150:
                    del_detail_id.append(row["SESSION_DETAIL_ID"])
                    has_nan = True
                if has_nan:
                    continue

                # else check duplicated time
                if prev_row is None:
                    prev_row = row
                    continue
                else: # TODO check if time is same, if it is duplicated measure, or it is weird range
                    if (s_to_t(row["DATETIME"]) - s_to_t(prev_row["DATETIME"])).total_seconds() < 1:
                        del_detail_id.append(prev_row["SESSION_DETAIL_ID"])
                    prev_row = row

            sessions[id_][session_id] = [r for r in sessions[id_][session_id] if r["SESSION_DETAIL_ID"] not in del_detail_id]

            # sort
            sort_lam = lambda x: datetime.strptime(x["DATETIME"], '%m/%d/%Y %H:%M')
            sessions[id_][session_id] = sorted(sessions[id_][session_id], key=sort_lam)

            # calculate gaps
            gaps = [(s_to_t(sessions[id_][session_id][i]["DATETIME"]) - s_to_t(sessions[id_][session_id][i-1]["DATETIME"])).total_seconds() / 60 / 60 for i in range(1, len(sessions[id_][session_id]))]
            BPs = [sessions[id_][session_id][i]["SYSBP"] for i in range(len(sessions[id_][session_id]))]
            for i in range(len(gaps)):
                if gaps[i] <= 1e-8:
                    bp_diffs.append(abs(BPs[i+1] - BPs[i]))

            # remove less than 2 measurement
            if len(sessions[id_][session_id]) < 2:
                del_session.append(session_id)
            else:
                # calculate duration
                start = datetime.strptime(sessions[id_][session_id][0]["DATETIME"], '%m/%d/%Y %H:%M')
                end = datetime.strptime(sessions[id_][session_id][-1]["DATETIME"], '%m/%d/%Y %H:%M')
                duration = (end - start).total_seconds() / 60 / 60

                # remove within range [1, 10] hours
                check_conditions = np.array([
                    duration < 1, # session too short
    # Cohort-filter variants: exclude sessions longer than ten hours or with duplicate timestamps.
    #                     duration > 10,
    #                     np.min(gaps) < 1e-8, # dual measure
                    np.max(gaps) > 2 # measure gap too long
                ])
                if np.sum(check_conditions) >= 1:
                    del_session.append(session_id)

        for session in del_session:
            del sessions[id_][session]

        # remove empty patient
        if len(sessions[id_]) < 1:
            del_id.append(id_)

    for id_ in del_id:
        del sessions[id_]

    # verbose      
    num_patient = 0
    num_sessions = 0
    durations = list() # in hours
    measure_gaps = list() # in hours
    num_measures = list()
    print("Checking Stats")
    for id_ in tqdm(sessions):
        num_patient += 1
        for session_id in sessions[id_]:
            num_sessions += 1

            # calculate duration
            start = datetime.strptime(sessions[id_][session_id][0]["DATETIME"], '%m/%d/%Y %H:%M')
            end = datetime.strptime(sessions[id_][session_id][-1]["DATETIME"], '%m/%d/%Y %H:%M')
            duration = (end - start).total_seconds() / 60 / 60
            durations.append(duration)

            # calculate gaps
            gaps = [(s_to_t(sessions[id_][session_id][i]["DATETIME"]) - s_to_t(sessions[id_][session_id][i-1]["DATETIME"])).total_seconds() / 60 / 60 for i in range(1, len(sessions[id_][session_id]))]
            measure_gaps += gaps

            # calculate numbers
            num_measures.append(len(sessions[id_][session_id]))
    # versbose
    print("Clean Data")
    print("# patients", num_patient)
    print("# sessions", num_sessions)
    print("=========\n")
    # visual
    check_stat = [bp_diffs, durations, measure_gaps, num_measures]
    titles = ["BP Dual Measure Diff", "Duration Hours", "Measurement Gaps", "Number of BP Measure"]
    for i in range(len(check_stat)):
        check_distribution(check_stat[i], title=titles[i])
    
    return sessions


def main():
    records = fetch_demo_lab_med()
    sessions = clean_dialysis_details(fetch_raw_dial_details())
    output = os.path.join(DATA_ROOT, "cleaned_records_all")
    os.makedirs(output, exist_ok=True)
    for patient_id, record in records.items():
        if patient_id in sessions:
            record["sessions"] = sessions[patient_id]
            with open(os.path.join(output, str(patient_id)), "wb") as f:
                pickle.dump(record, f)

if __name__ == "__main__":
    main()
