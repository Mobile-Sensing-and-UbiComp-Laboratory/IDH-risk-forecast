from src.config import DATA_ROOT, FIGURE_ROOT, SEQ_LENGTH
import pickle
import os
from tqdm import tqdm
from scipy.spatial.distance import cdist

from src.data_prepare.eval_split import *
from src.models.pre_real_all import *

def load_model(i, m_name="pre_real_all"):
    # construct model
    model = PreRealAll(
            emb_size=192, # 48, 192, 768, 2048, 4096
            num_layers=2, # 1, 2, 12, 24, 32
            num_heads=12, # 12, 32
            # data related:
            in_channel=8, # signal data size, 4 if .ph, 8 if .ph+.mp
            pre_size=18, # pre-session data size, 18 for pure num, 768 for llm embed
            seq_length=SEQ_LENGTH,
            # other setting
            use_cls=False,
            late_fuse=False,
            seg_embed=False
        ).to(DEVICE)

    # load trained weights
    model_record_name = (DATA_ROOT + '/exp_res/{}/{}_model_{}.pt').format(m_name, m_name, i)
    model.load_state_dict(torch.load(model_record_name, map_location=torch.device('cpu')))
    model.eval()

    return model

def get_output(model, fn, demos, chosen_def="fall20", facility='all'):
    # init
    record = dict()
    keys = ["bodies", "fall20", "next_measures", "summaries", "decays", "ehrs", "ehr_decays"]
    for k in keys:
        record[k] = list()

    # fetch data
    with open(os.path.join((DATA_ROOT + '/clean_samples_{}').format(facility), fn), "rb") as f:
        sample = pickle.load(f)

    summary_path = os.path.join((DATA_ROOT + '/clean_summaries_{}').format(facility), "_".join(fn.split("_")[:-1]))
    with open(summary_path, "rb") as f:
        summaries = pickle.load(f)

    ehr_path = os.path.join((DATA_ROOT + '/cleaned_ehr_{}').format(facility), "_".join(fn.split("_")[:-1]))
    with open(ehr_path, "rb") as f:
        ehrs = pickle.load(f)

    record["bodies"].append(sample["body"]+sample["machine"][:4])
    record[chosen_def].append(1 if sample[chosen_def] else 0)

    record["summaries"].append(summaries["summaries"])
    record["decays"].append(summaries["decays"])

    record["ehrs"].append(ehrs["ehrs"])
    record["ehr_decays"].append(ehrs["ehr_decays"])
    
    # forward pass
    with torch.no_grad():
        pred = model(
            torch.tensor(record["bodies"]).float().to(DEVICE),
            torch.tensor(record["summaries"]).float().to(DEVICE), 
            torch.tensor(record["decays"]).float().to(DEVICE),
            torch.tensor(record["ehrs"]).float().to(DEVICE), 
            torch.tensor(record["ehr_decays"]).float().to(DEVICE),
        )
    preds = pred[chosen_def][:, 1].cpu().detach().numpy().tolist()
    trues = record[chosen_def]
    
    curr_id = fn.split("_")[0]
    sex_g, age_g = demos[int(curr_id)]["SEX"], demos[int(curr_id)]["AGE_GROUP"]
    
    return {
        "SEX": {sex_g: {
            "preds": preds,
            "trues": trues
        }},
        "AGE_GROUP": {age_g: {
            "preds": preds,
            "trues": trues
        }}
    }

        
def plot_i(m_name="pre_real_all", chosen_i=0, chosen_def='fall20'):
    with open((DATA_ROOT + '/splits_5fold_all'), "rb") as f:
        splits = pickle.load(f)
    split = splits[chosen_i]["test_fnames"]
    
    with open((DATA_ROOT + '/demographics_groups.pkl'), 'rb') as f:
        demos = pickle.load(f)
    
    # collect results
    stats = {
        "SEX": dict(),
        "AGE_GROUP": dict()
    }
    for t in ["M", "F"]:
        stats["SEX"][t] = {
            "preds": list(),
            "trues": list()
        }
    for t in [0, 1, 2]:
        stats["AGE_GROUP"][t] = {
            "preds": list(),
            "trues": list()
        }
    
    model = load_model(chosen_i, m_name=m_name)
    
    for fn in tqdm(split):
        res = get_output(model, fn, demos, chosen_def=chosen_def)
        for g in res:
            for t in res[g]:
                stats[g][t]["preds"] += res[g][t]["preds"]
                stats[g][t]["trues"] += res[g][t]["trues"]
    
    return stats

if __name__ == "__main__":
    chosen_def = 'fall20'
    final_stats = None
    for i in tqdm(range(5)):
        stats = plot_i(m_name="pre_real_all", chosen_i=i, chosen_def=chosen_def)
        with open((DATA_ROOT + '/exp_res/err_{}_{}.pkl').format(chosen_def, i), 'wb') as f:
            pickle.dump(stats, f)
            
    # Aggregation variant: pool subgroup predictions across folds instead of keeping fold-specific records.
    #     if final_stats is None:
    #         final_stats = stats
    #     else:
    #         for g in stats:
    #             for t in stats[g]:
    #                 final_stats[g][t]["preds"] += stats[g][t]["preds"]
    #                 final_stats[g][t]["trues"] += stats[g][t]["trues"]
                    
    # Pooled-fold variant: save the accumulated subgroup record from the block above.
    # with open("data/exp_res/err_{}.pkl".format(chosen_def), 'wb') as f:
    #     pickle.dump(final_stats, f)
        
