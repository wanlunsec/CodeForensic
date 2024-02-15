import pandas as pd
import numpy as np
import argparse
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve


"""
config
"""
parser = argparse.ArgumentParser()
parser.add_argument("--gpu", type=str, default=1)
parser.add_argument("--dataset", type=str, default="apps")
parser.add_argument("--model", type=str, default="PolyCoder-160M")
parser.add_argument("--reference_model", type=str, default="incoder-1B")
parser.add_argument("--max_new_tokens", type=int, default=256)
parser.add_argument("--temperature", type=float, default=0.2)
parser.add_argument("--sample_num", type=int, default=1000)
opt = parser.parse_args()

df_file_path = f"./log/ppl_{opt.dataset}_train_{opt.model}_{opt.max_new_tokens}_tar_.csv"
df = pd.read_csv(df_file_path)
ppl_model_train = np.array(df["ppl"].to_list()[:opt.sample_num])
df_file_path = f"./log/ppl_{opt.dataset}_train_{opt.reference_model}_{opt.max_new_tokens}_ref_.csv"
df = pd.read_csv(df_file_path)
ppl_reference = df["ppl"].to_list()[:opt.sample_num]
log_likelihood_ratio_train = np.log(np.array(ppl_reference)/(np.array(ppl_model_train)+1e-7))
log_likelihood_ratio_train[np.argwhere(np.isnan(log_likelihood_ratio_train))] = 0
ppl_model_train[np.argwhere(np.isnan(ppl_model_train))] = 0

df_file_path = f"./log/ppl_{opt.dataset}_test_{opt.model}_{opt.max_new_tokens}_tar_.csv"
df = pd.read_csv(df_file_path)
ppl_model_test = np.array(df["ppl"].to_list()[:opt.sample_num])
df_file_path = f"./log/ppl_{opt.dataset}_test_{opt.reference_model}_{opt.max_new_tokens}_ref_.csv"
df = pd.read_csv(df_file_path)
ppl_reference = df["ppl"].to_list()[:opt.sample_num]
log_likelihood_ratio_test = np.log(np.array(ppl_reference)/(np.array(ppl_model_test)+1e-7))
log_likelihood_ratio_test[np.argwhere(np.isnan(log_likelihood_ratio_test))] = 0
ppl_model_test[np.argwhere(np.isnan(ppl_model_test))] = 0

test_labels_all = [0]*opt.sample_num+[1]*opt.sample_num
test_deviations_all = np.concatenate([log_likelihood_ratio_test, log_likelihood_ratio_train],0)
auc_score = roc_auc_score(test_labels_all, test_deviations_all)
print("auc_score :", auc_score)
