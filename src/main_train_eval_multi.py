from src.config import DATA_ROOT, FIGURE_ROOT, SEQ_LENGTH
import pickle
import os
import matplotlib.pyplot as plt
import gc

from torch.utils.data import Dataset
from src.data_prepare.eval_split import *

from src.models.transformer_base import *
from src.models.pre_real_all import *


class IDHDataset(Dataset):
    def __init__(
        self, 
        fnames, 
        facility="all",
        real_time=True,
        history=False,
        record=False,
        acute_interv=False,
        machine_param=False,
        data_root=DATA_ROOT
    ):  
        self.fnames = fnames
        self.facility = facility

        self.real_time = real_time
        self.history = history
        self.record = record
        self.acute_interv = acute_interv
        self.machine_param = machine_param
        self.data_root = data_root

    def __len__(self):
        return len(self.fnames)

    def __getitem__(self, idx):
        record = dict()

        if self.real_time:
            # Historical sampling variant: read a dedicated three-minute sample directory; match IDH_SAMPLE_EVERY accordingly.
            # with open(os.path.join("{}/clean_samples_{}_3min".format(self.data_root, self.facility), self.fnames[idx]), "rb") as f:
            with open(os.path.join((self.data_root + '/clean_samples_{}').format(self.facility), self.fnames[idx]), "rb") as f:
                sample = pickle.load(f)

            bodies = sample["body"] # if .ph only
            if self.machine_param:
                bodies = bodies + sample["machine"][:4] # core model uses four machine channels
            record = {
                'bodies': torch.tensor(bodies).float().to(DEVICE), 
                'fall20': torch.tensor(sample["fall20"]).long().to(DEVICE),
                'fall30': torch.tensor(sample["fall30"]).long().to(DEVICE),
                'nadir90': torch.tensor(sample["nadir90"]).long().to(DEVICE),
                'nadir100': torch.tensor(sample["nadir100"]).long().to(DEVICE),
                'hemo': torch.tensor(sample["hemo"]).long().to(DEVICE),
                'kdoqi': torch.tensor(sample["kdoqi"]).long().to(DEVICE),
                "next_measures": torch.tensor(sample["next_measure"]).float().to(DEVICE),
            }
        
        # add acute action if required
        if self.acute_interv:
            volume_back = False
            if sample['acute_interv']['ufr'] == 0:
                if sample['acute_interv']['net_volume_removed'] < 0:
                    volume_back = True
                if sample['acute_interv'].get('volume_back') is not None and sample['acute_interv']['volume_back']:
                    volume_back = True
                if bodies[0][-1]*128 < 90:
                    volume_back = True
            
            # pack up
            record['acute_interv'] = torch.tensor([
                # Action-encoding variants: include positive UFR status or raw temperature; coordinate downstream action dimensions.
                # sample['acute_interv']['ufr'] > 0,
                # sample['acute_interv']['flow_temp'], # return raw flow temp
                sample['acute_interv']['ufr'] == 0 and not volume_back, # showtdown UFR
                sample['acute_interv']['ufr_diff'] > 0, # decrease UFR
                # Action-encoding variant: include a negative-UFR indicator; coordinate downstream action dimensions.
                # sample['acute_interv']['ufr'] < 0, # increase UFR
                volume_back, # volume is given back
                sample['acute_interv']['flow_temp'] > 0, # decrease flow temperature
                # Action-encoding variant: include saline and supine indicators; coordinate downstream action dimensions.
                # sample['acute_interv']['saline'],
                # sample['acute_interv']['supine']
                sample['acute_interv']['ufr_diff'] < 0, # turn UFR back
                sample['acute_interv']['flow_temp'] < 0, # turn flow temperature back
            ]).long().to(DEVICE)
        record['fnames'] = self.fnames[idx]
        
        # fetch past dialysis summary
        id_, session_id, idx_ = self.fnames[idx].split("_")
        if self.history:
            summary_path = os.path.join("{}/clean_summaries_{}".format(self.data_root, self.facility), "{}_{}".format(id_, session_id))

            try:
                with open(summary_path, "rb") as f:
                    summaries = pickle.load(f)
                record["summaries"] = torch.tensor(summaries["summaries"]).float().to(DEVICE)
                record["decays"] = torch.tensor(summaries["decays"]).float().to(DEVICE)
            except:
                record["summaries"] = torch.zeros(64, 18).to(DEVICE)
                record["decays"] = torch.zeros(64).to(DEVICE)
            
            if len(record["summaries"].shape) < 2:
                record["summaries"] = torch.zeros(64, 18).to(DEVICE)

        # fetch medication and lab tests results
        if self.record:
            ehr_path = os.path.join("{}/cleaned_ehr_{}".format(self.data_root, self.facility), "{}_{}".format(id_, session_id))
            with open(ehr_path, "rb") as f:
                ehrs = pickle.load(f)
            record["ehrs"] = torch.tensor(ehrs["ehrs"]).float().to(DEVICE)
            record["ehr_decays"] = torch.tensor(ehrs["ehr_decays"]).float().to(DEVICE)
            
        return record

def load_model(model_name):
    if model_name == "baseline":
        model = TransformerBase(in_channel=8) # 4 if .ph, 8 if .ph+.mp
    elif model_name == "pre_real_all":
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
        )
    print("Num params:", sum(p.numel() for p in model.parameters() if p.requires_grad))


    return model.to(DEVICE)

def eval_one_fold(i, trail_name="baseline", data_root=DATA_ROOT):
    # get splits
    with open((data_root + '/splits_5fold_all'), "rb") as f:
        split = pickle.load(f)[i]

    print("Train Num:", len(split["train_fnames"]))
    print("Test Num:", len(split["test_fnames"]))

    # load data
    real_time = True
    history = True
    record_ = True
    machine_param = True
    train_dataset = IDHDataset(
        split["train_fnames"], 
        facility="all",
        real_time=real_time,
        history=history,
        record=record_,
        machine_param=machine_param,
        data_root=data_root
    )
    eval_dataset = IDHDataset(
        split["test_fnames"], 
        facility="all",
        real_time=real_time,
        history=history,
        record=record_,
        machine_param=machine_param,
        data_root=data_root
    )

    torch.cuda.empty_cache()
    gc.collect()
    del split

    # Architecture variant: select the model by experiment name; only names supported by load_model are valid.
    # construct model
    # model = load_model(model_name=trail_name)
    model = load_model(model_name='pre_real_all')

    save_model = True
    model_record_name = (DATA_ROOT + '/exp_res/{}_model_{}.pt').format(trail_name, i)
    # Initialization variant: load an existing checkpoint before fitting; set its path explicitly.
    # model.load_state_dict(torch.load(model_record_name))

    # fit model
    record = model.fit(
        train_dataset,
        eval_dataset,
        batch_size=256,
        epochs=5, # 10, 5?, 3
        lr=5e-4, # 5e-4
        weight_decay=1e-5, # 1e-5
    )

    os.makedirs((DATA_ROOT + '/exp_res/{}').format(trail_name), exist_ok=True)

    # plot the training losses
    fig_name = (DATA_ROOT + '/exp_res/{}/{}_{}.png').format(trail_name, trail_name, i)
    plt.clf()
    plt.plot(record["train_i"], record["train_losses"], label="Train")
    plt.plot(record["eval_i"], record["eval_losses"], label="Eval")
    plt.legend()
    plt.savefig(fig_name)

    # final evaluation and save the record
    curr_scores = eval_res_multi(record["last_pred"]["y_preds"], record["last_pred"]["y_trues"])
    for s in curr_scores:
        print(s, curr_scores[s])
    
    record_name = (DATA_ROOT + '/exp_res/{}/{}_record_{}').format(trail_name, trail_name, i)
    with open(record_name, "wb") as f:
        records = {
            "y_preds": record["last_pred"]["y_preds"],
            "y_trues": record["last_pred"]["y_trues"],
        }
        pickle.dump(records, f)

    if save_model:
        model_record_name = (DATA_ROOT + '/exp_res/{}/{}_model_{}.pt').format(trail_name, trail_name, i)
        torch.save(model.state_dict(), model_record_name)
    
    return fig_name, record_name

if __name__ == '__main__':

    # get args
    trail_name = sys.argv[1] # output experiment name
    fold_i = int(sys.argv[2]) # 0, 1, 2, 3, 4

    # run job
    torch.cuda.empty_cache()
    gc.collect()
    fig_name, record_name = eval_one_fold(fold_i, trail_name=trail_name, data_root=DATA_ROOT)


    # Sampling variants: set IDH_SAMPLE_EVERY and use a separate data root and experiment name.

# Example fold commands for the one-minute experiment (set IDH_SAMPLE_EVERY=1).
# python3 -m src.main_train_eval_multi pre_real_all_1min 0
# python3 -m src.main_train_eval_multi pre_real_all_1min 1
# python3 -m src.main_train_eval_multi pre_real_all_1min 2
# python3 -m src.main_train_eval_multi pre_real_all_1min 3
# python3 -m src.main_train_eval_multi pre_real_all_1min 4
