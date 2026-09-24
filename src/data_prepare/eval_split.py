from src.config import DATA_ROOT, FIGURE_ROOT, SEQ_LENGTH
import numpy as np
import os
import sys
import pickle
from tqdm import tqdm

from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix, f1_score, roc_auc_score, average_precision_score


def fetch_all_subjects(facility="all"):
    sample_names = sorted(os.listdir((DATA_ROOT + '/clean_samples_{}').format(facility)))
    # each sample is: patient-id_session-id_idx
    subjects = dict()
    for n in sample_names:
        curr_ids = n.split("_")
        if len(curr_ids) == 3:
            if subjects.get(curr_ids[0]) is None:
                subjects[curr_ids[0]] = [n]
            else:
                subjects[curr_ids[0]].append(n)
    print("Num Subjects:", len(subjects))
    return subjects

def kfold_split(subjects, n=5, seed=42):
    subject_names = np.array([s for s in subjects.keys()])
    kf = KFold(n_splits=n, shuffle=True, random_state=seed)

    splits = list()
    for i, (train_index, test_index) in enumerate(kf.split(subject_names)):
        curr_split = {
            "train_fnames": list(),
            "test_fnames": list()
        }
        for s in subject_names[train_index]:
            curr_split["train_fnames"] += subjects[s]
        for s in subject_names[test_index]:
            curr_split["test_fnames"] += subjects[s]
        splits.append(curr_split)
    return splits

def eval_res(y_preds, y_trues):
    pred_class = np.array(y_preds) > 0.5
    scores = {
        "confmats": confusion_matrix(y_trues, pred_class, labels=[0, 1]),
        "f1": f1_score(y_trues, pred_class, average='binary'),
        "pr_auc": average_precision_score(y_trues, y_preds),
    }
    try:
        scores["roc_auc"] = roc_auc_score(y_trues, y_preds)
    except:
        scores["roc_auc"] = 0.0

    # 
    cf = scores["confmats"]
    TN, TP, FP, FN = cf[0][0], cf[1][1], cf[0][1], cf[1][0]
    scores["specificity"] = TN / (TN + FP)
    scores["sensitivity"] = TP / (TP + FN)
    scores["balance_acc"] = (scores["specificity"]+scores["sensitivity"]) / 2
    return scores

def eval_res_multi(y_preds, y_trues, verbose=True):
    scores = dict()
    for task in y_preds:
        scores[task] = eval_res(y_preds[task], y_trues[task])

        if verbose:
            print(task)
            for s in scores[task]:
                print(s, scores[task][s])
            print("======\n")
    return scores

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="Create patient-disjoint cross-validation folds")
    parser.add_argument("facility", nargs="?", default="all")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    subjects = fetch_all_subjects(args.facility)
    splits = kfold_split(subjects, n=5, seed=args.seed)
    path = os.path.join(DATA_ROOT, "splits_5fold_{}".format(args.facility))
    # Exclusive creation protects an existing study split.
    with open(path, "xb") as f:
        pickle.dump(splits, f)
    print("Saved", path)
