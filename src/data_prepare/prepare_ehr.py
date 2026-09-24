"""ClinicalBERT/PCA care-record pipeline extracted from eda/Segment_Embedding.ipynb."""
from src.config import DATA_ROOT, RAW_ROOT

if __name__ == "__main__":
    import os
    import sys
    import pickle
    import numpy as np
    from datetime import datetime, timedelta
    from tqdm import tqdm
    import pandas as pd
    import matplotlib.pyplot as plt
    
    import h5py
    from sklearn.decomposition import PCA
    
    import torch
    from transformers import AutoTokenizer, AutoModelForMaskedLM
    
    tokenizer = AutoTokenizer.from_pretrained("medicalai/ClinicalBERT")
    model = AutoModelForMaskedLM.from_pretrained("medicalai/ClinicalBERT")
    
    def get_embedding(text):
        with torch.no_grad():
            ids = torch.tensor(tokenizer.encode(text)).unsqueeze(0)
            out = model(ids, output_hidden_states=True)
    # Embedding variant: use the CLS token instead of mean-pooling ClinicalBERT token states.
    #     return out.hidden_states[-1][0, 0, :]
        return torch.mean(out.hidden_states[-1], dim=1)[0, :]
    
    def time_to_stamp(date_time):
        return datetime.strptime(date_time, '%m/%d/%Y %H:%M')
    
    def read_record(facility, id_):
        # facility id, patient id
        with open((DATA_ROOT + '/cleaned_records_{}/{}').format(facility, id_), 'rb') as f:
            record = pickle.load(f)
        return record
    
    def save_record(facility, id_, record):
        # facility id, patient id
        with open((DATA_ROOT + '/cleaned_records_{}/{}').format(facility, id_), 'wb') as f:
            pickle.dump(record, f)
    
    def pca_down(data, dim=2):
        pca = PCA(n_components=dim)
        pca.fit(data)
        
        return pca, pca.transform(data)


    dss = pd.read_csv(os.path.join(RAW_ROOT, "diagnoses.csv"))
    id_to_dss = dict()
    for idx, row in tqdm(dss.iterrows()):
        curr_record = {
            "timestamp": time_to_stamp(row["ENTRY_DATE_TIME"]),
            "disease": row["DIAG_DESCRIPTION"].lower()
        }
        if id_to_dss.get(row["PATIENT_ID"]) is None:
            id_to_dss[row["PATIENT_ID"]] = [curr_record]
        else:
            id_to_dss[row["PATIENT_ID"]].append(curr_record)


    facility = "all"
    ids = os.listdir((DATA_ROOT + '/cleaned_records_{}').format(facility))
    for id_ in tqdm(ids):
        if id_[0] == '.':
            continue
        record = read_record(facility, id_)
        if id_to_dss.get(int(id_)) is None:
            record["disease"] = list()
        else:
            record["disease"] = id_to_dss[int(id_)]
        save_record(facility, id_, record)
    
    '''
    keys: timestamp
    
    lab_test_records: test_type, result
    disease: disease
    med_records: med_type, dose
    '''
    
    LAB_TEST_TEMP = """{test_type} test, result of {result}."""
    MED_TEMP = """{med_type} taken, dose of {dose}."""
    DISEASE_TEMP = """diagnosed as {disease}."""
    
    lab_tests_all, medication_all, dx_all = list(), list(), list()
    
    
    for id_ in tqdm(ids):
        if id_[0] == '.':
            continue
        record = read_record(facility, id_)
        
        # save record
        for l in record["lab_test_records"]:
            lab_tests_all.append(LAB_TEST_TEMP.format(**l).lower())
        for m in record["med_records"]:
            medication_all.append(MED_TEMP.format(**m).lower())
        for d in record["disease"]:
            dx_all.append(DISEASE_TEMP.format(**d).lower())
            
    
    all_labs = [d for d in set(lab_tests_all)]
    
    embeds = [get_embedding(d).detach().numpy().tolist() for d in all_labs]
    embeds = np.array(embeds)
    
    pca_labs, embed_down_lab = pca_down(embeds, dim=192)
    
    all_dx = [d for d in set(dx_all)]
    
    embeds = [get_embedding(d).detach().numpy().tolist() for d in all_dx]
    embeds = np.array(embeds)
    
    pca_dx, embed_down_dx = pca_down(embeds, dim=192)
    embed_ehr = np.concatenate((embed_down_lab, embed_down_dx), axis=0)
    all_ehr = all_labs + all_dx
    
    
    '''
    keys: timestamp
    
    lab_test_records: test_type, result
    disease: disease
    med_records: med_type, dose
    '''
    
    all_ehrs = dict()
    
    for id_ in tqdm(ids):
        if id_[0] == '.':
            continue
        record = read_record(facility, id_)
        
        # save record
        for l in record["lab_test_records"]:
            if all_ehrs.get(id_) is None:
                all_ehrs[id_] = list()
            try:
                all_ehrs[id_].append((all_ehr.index(LAB_TEST_TEMP.format(**l).lower()), l["timestamp"]))
            except:
                print("Lab sentence missing:", LAB_TEST_TEMP.format(**l).lower())
        for d in record["disease"]:
            if all_ehrs.get(id_) is None:
                all_ehrs[id_] = list()
            try:
                all_ehrs[id_].append((all_ehr.index(DISEASE_TEMP.format(**d).lower()), d["timestamp"]))
            except:
                print("Disease sentence missing:", DISEASE_TEMP.format(**d).lower())
        
        # sort
        if all_ehrs.get(id_) is not None:
            all_ehrs[id_] = sorted(all_ehrs[id_], key=lambda x: x[1])
    
    import h5py
    
    with h5py.File((DATA_ROOT + '/IDH_EHR.h5'), 'w') as h5f:
        h5f.create_dataset("EHR", data=embed_ehr, maxshape=(None, 192), chunks=True)
    
    def match_lab_test_and_dia_sum(id_, facility="all"):
        record = read_record(facility, id_)
        for s in record['sessions']:
            s_time = time_to_stamp(record['sessions'][s][0]["DATETIME"])
            curr_key = "{}_{}".format(id_, record['sessions'][s][0]["SESSION_ID"])
            pad_clip_to = 64 # 256, 128, 64
            
            # match past ehr summary
            curr_sums = list()
            decays = list()
    
            id_ = str(id_)
            if all_ehrs.get(id_) is None:
                print("No valid summary for", id_)
                curr_sums = [0 for _ in range(pad_clip_to)]
                decays = [0.0 for _ in range(pad_clip_to)]
            else:
                for rec in all_ehrs[id_]:
                    if rec[1] < s_time:
                        curr_sums.append(rec[0])
                        diff_days = (s_time - rec[1]).days
                        decays.append(0.997 ** diff_days)
    
                # paddinga
                if len(curr_sums) < pad_clip_to:
                    curr_sums = [0 for _ in range(pad_clip_to-len(curr_sums))] + curr_sums
                    decays = [0.0 for _ in range(pad_clip_to-len(decays))] + decays
            
            # saved [sum_idxs, decays], (2, 64)
            saved_data = np.array([curr_sums[-pad_clip_to:], decays[-pad_clip_to:]]).astype(float)
            
            with h5py.File((DATA_ROOT + '/IDH_EHR.h5'), "a") as h5f:
                h5f.create_dataset(curr_key, data=saved_data, maxshape=(2, 64), chunks=True)
                
    ids = sorted(os.listdir((DATA_ROOT + '/cleaned_records_{}').format("all")))
    for id_ in tqdm(ids):
        match_lab_test_and_dia_sum(int(id_))
    
    import h5py
    import numpy as np
    from tqdm import tqdm
    import pickle
    
    os.makedirs(os.path.join(DATA_ROOT, 'cleaned_ehr_all'), exist_ok=True)
    with h5py.File((DATA_ROOT + '/IDH_EHR.h5'), "r") as h5f:
        # get reference to all raw vectors
        ehrs_all = h5f["EHR"]
        
        for k in tqdm(h5f):
            if k == "EHR":
                continue
            
            # get relevant index information
            ehrs = h5f[k] # (2, 64), 0:idx, 1:decay_factors
            idxs = [i_ for i_ in ehrs[0].astype(int) if i_ > 0]
            if len(idxs) > 0:
                l, h = min(idxs), max(idxs)
            else:
                l, h = 0, 0
            
            # get minimum chunks and match to all vectors
            chunks = ehrs_all[l:h+1, :]
            ehr_vecs = chunks[np.maximum(0, ehrs[0].astype(int) - l), :]
    
            
            # update record
            curr_ehr = {
                "ehrs": ehr_vecs.astype(np.float16).tolist(),
                "ehr_decays": ehrs[1].astype(np.float16).tolist(),
            }
            save_name = (DATA_ROOT + '/cleaned_ehr_all/{}').format(k)
            with open(save_name, 'wb') as f:
                pickle.dump(curr_ehr, f)
