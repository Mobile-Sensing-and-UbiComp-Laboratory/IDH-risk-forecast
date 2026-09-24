from src.config import DATA_ROOT, FIGURE_ROOT, SEQ_LENGTH
import pickle
from src.data_prepare.eval_split import *

def calc_res(trail_name):
    # Historical single-task summary variant: requires flat prediction/target arrays and the matching accumulation/reporting blocks.
    # # single task
    # scores = {
    #     "confmats": list(),
    #     "f1": list(),
    #     "pr_auc": list(),
    #     "roc_auc": list(),
    #     "specificity": list(),
    #     "sensitivity": list(),
    #     "balance_acc": list()
    # }

    # multitask
    scores = dict()

    for i in tqdm([
        0,
        1,
        2,
        3,
        4
    ]):  
        record_path = (DATA_ROOT + '/exp_res/{}/{}_record_{}').format(trail_name, trail_name, i)
        with open(record_path, "rb") as f:
            record = pickle.load(f)
        
        # Single-task variant: paired with the flat score dictionary and reporting block.
        # single task
        # curr_scores = eval_res(record["y_preds"], record["y_trues"])
        # for s in curr_scores:
        #     scores[s].append(curr_scores[s])

        # add the criteria with "and"
        post_task = {
            'nadir90fall20': {
                "y_preds": [record["y_preds"]["nadir90"][j]*record["y_preds"]["fall20"][j] for j in range(len(record["y_preds"]["nadir90"]))],
                "y_trues": [record["y_trues"]["nadir90"][j] and record["y_trues"]["fall20"][j] for j in range(len(record["y_preds"]["nadir90"]))]
            },
            'nadir90fall30': {
                "y_preds": [record["y_preds"]["nadir90"][j]*record["y_preds"]["fall30"][j] for j in range(len(record["y_preds"]["nadir90"]))],
                "y_trues": [record["y_trues"]["nadir90"][j] and record["y_trues"]["fall30"][j] for j in range(len(record["y_preds"]["nadir90"]))]
            },
            'fall20kdoqi': {
                "y_preds": [record["y_preds"]["kdoqi"][j]*record["y_preds"]["fall20"][j] for j in range(len(record["y_preds"]["kdoqi"]))],
                "y_trues": [record["y_trues"]["kdoqi"][j] and record["y_trues"]["fall20"][j] for j in range(len(record["y_preds"]["kdoqi"]))]
            }
        }
        for t in post_task:
            record["y_preds"][t] = post_task[t]['y_preds']
            record["y_trues"][t] = post_task[t]['y_trues']

        # multitask
        curr_scores = eval_res_multi(record["y_preds"], record["y_trues"], verbose=False)
        if len(scores) == 0:
            for task in curr_scores:
                scores[task] = {
                    "confmats": list(),
                    "f1": list(),
                    "pr_auc": list(),
                    "roc_auc": list(),
                    "specificity": list(),
                    "sensitivity": list(),
                    "balance_acc": list()
                }
        for task in curr_scores:
            if task == "fall20":
                print(curr_scores[task]['f1'])
            for s in curr_scores[task]:
                scores[task][s].append(curr_scores[task][s])
    
    # Single-task variant: report the flat score dictionary instead of task-keyed metrics.
    # # single task
    # for s in scores:
    #     if s == "confmats":
    #         print(s)
    #         print(np.sum(scores[s], axis=0))
    #         continue
    #     print(s, np.mean(scores[s]), "+-", np.std(scores[s]))
    
    # multitask
    for task in scores:
        print(task)
        for s in scores[task]:
            if s == "confmats":
                print(s)
                print(np.sum(scores[task][s], axis=0))
                continue
            print(s, round(np.mean(scores[task][s]), 3), "$\pm$", round(np.std(scores[task][s]), 3))
        print("======\n")

if __name__ == '__main__':
    trail_name = sys.argv[1]
    calc_res(trail_name)

    # python3 -m src.res_summary pre_real_all
    # python3 -m src.res_summary pre_real_all_adjust_ehr
    # python3 -m src.res_summary pre_real_all_1min
