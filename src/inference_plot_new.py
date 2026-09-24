from src.config import DATA_ROOT, FIGURE_ROOT, SEQ_LENGTH, SAMPLE_EVERY
import pickle
import os
import matplotlib.pyplot as plt
import gc
import time
import shutil

import tracemalloc
from torch.utils.data import Dataset
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

def plot_one(model, fn_, facility="all", test=False, icu_style=False, chosen_def='fall20', legend=True):
    # init
    record = dict()
    keys = ["bodies", chosen_def, "next_measures", "summaries", "decays", "ehrs", "ehr_decays"]
    for k in keys:
        record[k] = list()

    # fetch data
    i = 0
    while True:
        try:
            with open(os.path.join((DATA_ROOT + '/clean_samples_{}').format(facility), "{}_{}".format(fn_, i)), "rb") as f:
                sample = pickle.load(f)
            
            summary_path = os.path.join((DATA_ROOT + '/clean_summaries_{}').format(facility), fn_)
            with open(summary_path, "rb") as f:
                summaries = pickle.load(f)
            
            ehr_path = os.path.join((DATA_ROOT + '/cleaned_ehr_{}').format(facility), fn_)
            with open(ehr_path, "rb") as f:
                ehrs = pickle.load(f)
        except:
            break
        i += 1

        # update data
        # fetch label
        if chosen_def == 'fall20nadir90':
            record[chosen_def].append(sample['fall20'] and sample['nadir90'])
        else:
            record[chosen_def].append(sample[chosen_def])

        # fetch data
        record["bodies"].append(sample["body"]+sample["machine"][:4])
        record["next_measures"].append(sample["next_measure"])

        record["summaries"].append(summaries["summaries"])
        record["decays"].append(summaries["decays"])

        record["ehrs"].append(ehrs["ehrs"])
        record["ehr_decays"].append(ehrs["ehr_decays"])
    
    # forward pass
    with torch.no_grad():
        model.eval()
        pred = model(
            torch.tensor(record["bodies"]).float().to(DEVICE),
            torch.tensor(record["summaries"]).float().to(DEVICE), 
            torch.tensor(record["decays"]).float().to(DEVICE),
            torch.tensor(record["ehrs"]).float().to(DEVICE), 
            torch.tensor(record["ehr_decays"]).float().to(DEVICE),
        )
    
    # get prediction
    if chosen_def == 'fall20nadir90':
        fall20_pred = torch.softmax(pred['fall20'], dim=-1)
        nadir90_pred = torch.softmax(pred['nadir90'], dim=-1)
        preds = torch.argmax((fall20_pred*nadir90_pred), dim=1).cpu().detach().numpy().tolist()
    else:
        preds = torch.argmax(pred[chosen_def], dim=1).cpu().detach().numpy().tolist()
    next_measure_pred = pred["next_measures"].cpu()

    for task in pred:
        print(task, torch.softmax(pred[task][-1], dim=0))

    print("torch.cuda.max_memory_reserved: %fGB"%(torch.cuda.max_memory_reserved(0)/1024/1024/1024))

    # concatenate results
    # get ground truth blood pressures
    BPs, BPsx, measure_pred, measure_x, IDH_pred_x, IDH_ground_x = list(), [0], list(), list(), list(), list()
    prev_measure = 0
    prev_pred_IDH = 0
    prev_ground_IDH = False
    for i in range(1, len(preds)):
        # get measurement
        start_idx = 0 if i == 0 else len(BPs)+1
        curr_BPs = [n*128 for n in np.array(record["bodies"][i])[0, :] if n > 0][start_idx:]
        if len(curr_BPs) < 1:
            BPs += curr_BPs
        else:
            BPs.append(curr_BPs[-1])
        lastx = BPsx[-1]
        BPsx.append(lastx + ((len(curr_BPs))))

        # get predicted measurement
        measure_pred.append(prev_measure)
        measure_x.append(BPsx[-1])
        prev_measure = next_measure_pred[i][0] * 128

        # get predicted measurement
        if prev_pred_IDH > 0:
            IDH_pred_x.append(BPsx[-1])
        prev_pred_IDH = preds[i-1]
        if prev_ground_IDH:
            IDH_ground_x.append(BPsx[-1])
        prev_ground_IDH = record[chosen_def][i-1]
    
    # remove some of 1st idx
    BPsx = BPsx[1:]
    BPs = [BPs[0]] + BPs
    measure_pred = measure_pred[1:]
    measure_x = measure_x[1:]

    # append last data point
    BPs.append(record["next_measures"][-1][0] * 128)
    BPsx.append(int(BPsx[-1]+(30/SAMPLE_EVERY)))
    measure_pred.append(prev_measure)
    measure_x.append(BPsx[-1])

    if prev_pred_IDH > 0:
        IDH_pred_x.append(BPsx[-1])
    if prev_ground_IDH:
        IDH_ground_x.append(BPsx[-1])
    

    IDH_pred_x = [_ for _ in IDH_pred_x if _ != measure_x[-1]]
    IDH_ground_x = [_ for _ in IDH_ground_x if _ != BPsx[-1]]
    BPsx = BPsx[:-1]
    BPs = BPs[:-1]
    measure_x = measure_x[:-1]
    measure_pred = measure_pred[:-1]

    plt.clf()
    plt.rc('font', size=12)          # controls default text sizes
    plt.rc('axes', titlesize=15)     # fontsize of the axes title
    plt.rc('axes', labelsize=18)    # fontsize of the x and y labels
    plt.rc('xtick', labelsize=12)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=16)    # fontsize of the tick labels
    plt.rc('legend', fontsize=18)    # legend fontsize
    plt.rc('figure', titlesize=10)
    
    # color list
    # #08F7FE blue
    # #00ff41 green
    # #FE53BB pink
    # #F5D300 yellow
    
    # ICU dashboard like style setting
    if icu_style:
        plt.style.use("seaborn-darkgrid")
        plt.rcParams['grid.color'] = (0.5, 0.5, 0.5, 0.2)
        for param in ['figure.facecolor', 'axes.facecolor', 'savefig.facecolor']:
            plt.rcParams[param] = 'black'  # bluish dark grey
        plt.rcParams['figure.facecolor'] = 'k'
        for param in ['text.color', 'axes.labelcolor', 'xtick.color', 'ytick.color']:
            plt.rcParams[param] = '0.9'  # very light grey
        
        bp_color = '#00ff41'
        pred_bp_color = '#08F7FE'
    else:
        bp_color = 'b'
        pred_bp_color = 'g'
    # Construct the plot and draw ground-truth blood pressure.
    plt.plot(BPsx[1:-1], BPs[1:-1], label="SYSBP", color=bp_color, marker='o', zorder=1)
    # glow
    n_lines = 10
    diff_linewidth = 1.05
    alpha_value = 0.03
    if icu_style:
        for n in range(1, n_lines+1):
            plt.plot(BPsx[1:-1], BPs[1:-1],
                    linewidth=2+(diff_linewidth*n),
                    alpha=alpha_value,
                    color=bp_color)
    
    # plot pred bp
    for i in range(1, len(measure_pred) - 1):
        BP_ind = i
        if i == 1:
            plt.plot([BPsx[BP_ind], measure_x[i]], [BPs[BP_ind], measure_pred[i]], color=pred_bp_color, label='Forecast SYSBP', marker='o', zorder=-1)

            # glow for ICU style
            n_lines = 10
            diff_linewidth = 1.05
            alpha_value = 0.03
            if icu_style:
                for n in range(1, n_lines+1):
                    plt.plot([BPsx[BP_ind], measure_x[i]], [BPs[BP_ind], measure_pred[i]],
                            linewidth=2+(diff_linewidth*n),
                            alpha=alpha_value,
                            color=pred_bp_color)
        else:
            if measure_pred[i] == measure_pred[i-1]:
                continue
            plt.plot([BPsx[BP_ind], measure_x[i]], [BPs[BP_ind], measure_pred[i]], color=pred_bp_color, marker='o', zorder=-1)

            # glow for ICU style
            if icu_style:
                n_lines = 10
                diff_linewidth = 1.05
                alpha_value = 0.03
                for n in range(1, n_lines+1):
                    plt.plot([BPsx[BP_ind], measure_x[i]], [BPs[BP_ind], measure_pred[i]],
                            linewidth=2+(diff_linewidth*n),
                            alpha=alpha_value,
                            color=pred_bp_color)
                
    first_pred, first_ground = True, True
    for x_i in range(len(IDH_ground_x)):
        
        # draw ground truth
        x = IDH_ground_x[x_i]
        if first_ground:
            plt.axvline(x=x, color='red', label='Ground Truth '+chosen_def, linestyle='solid')
            first_ground = False
        else:
            plt.axvline(x=x, color='red', linestyle='solid')

    for x_i in range(len(IDH_pred_x)):
        x = IDH_pred_x[x_i]

        # draw pred
        if first_pred:
            plt.axvline(x=x, color='orange', label='Prediction', linestyle='--')
            first_pred = False
        else:
            plt.axvline(x=x, color='orange', linestyle='--')
    

    plt.ylim(65, 220)
    plt.xlabel("Time Steps")
    plt.ylabel("SYSBP")
    if legend:
        plt.legend(frameon=True)

    if not test:
        plt.savefig((FIGURE_ROOT + '/pre_real_all/{}_{}.pdf').format(chosen_def, fn_), format="pdf", bbox_inches="tight")
    else:
        plt.show()

def plot_i(m_name="pre_real_all", test=False, select_fn='', icu_style=False, chosen_def='fall20', legend=True):
    chosen_i = np.random.choice([i for i in range(5)], 1)[0]
    with open((DATA_ROOT + '/splits_5fold_all'), "rb") as f:
        splits = pickle.load(f)
    split = splits[chosen_i]["test_fnames"]

    if len(select_fn) == 0:
        chosen_fns = ["_".join(np.random.choice(split, 1)[0].split("_")[:-1])]
    else:
        chosen_fns = [select_fn]

    tracemalloc.start()
    start = time.time()

    for chosen_fn in tqdm(chosen_fns):
        matching_folds = [
            i for i, fold in enumerate(splits)
            if chosen_fn in {"_".join(n.split("_")[:-1]) for n in fold["test_fnames"]}
        ]
        if not matching_folds:
            raise ValueError("Selected session is absent from the test splits")
        chosen_i = matching_folds[0]

        model = load_model(chosen_i, m_name=m_name)
        plot_one(model, chosen_fn, test=test, icu_style=icu_style, chosen_def=chosen_def, legend=legend)

    end = time.time()
    mem_use = tracemalloc.get_traced_memory() # (current, peak)
    tracemalloc.stop()
    print("Total time cost:", end-start, "s")
    print("Max RAM require:", mem_use[1]/1e6, "M")

    return end-start, mem_use[1]/1e6, chosen_fns[0]

if __name__ == "__main__":
    os.makedirs(os.path.join(FIGURE_ROOT, "pre_real_all"), exist_ok=True)
    # Select a session from the held-out split; no embedded patient identifiers.
    plot_i(chosen_def="fall20nadir90", test=False)
