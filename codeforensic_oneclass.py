

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager
from sklearn import svm
import torch
import argparse
import os
from sklearn.utils import shuffle
from sklearn.metrics import roc_auc_score

from feature_extract import feature_extraction


parser = argparse.ArgumentParser()
parser.add_argument("--gpu", type=str, default=1)
parser.add_argument("--dataset", type=str, default="mbpp")
parser.add_argument("--model", type=str, default="unixcoder-base-nine")
parser.add_argument("--max_new_tokens", type=int, default=256)
parser.add_argument("--checkpoints", type=str, default="./checkpoints")
parser.add_argument("--repeat_time", type=int, default=5)
parser.add_argument("--temperature", type=float, default=0.2)
parser.add_argument("--batchsize", type=int, default=16)
parser.add_argument("--save_ckpt", type=bool, default=False)

opt = parser.parse_args()

# opt.model = "codebert-base"
os.environ["CUDA_VISIBLE_DEVICES"] = str(opt.gpu)
use_cuda = False
device = torch.device("cuda" if (use_cuda and torch.cuda.is_available()) else "cpu")

data_root1 = "../code_generation_eval"
data_root2 = "../code"
if os.path.exists(data_root1):
    data_root = data_root1
elif os.path.exists(data_root2):
    data_root = data_root2
else:
    raise Exception("error")

data_path = os.path.join(data_root, "checkpoints/data_{}_{}.pt".format(opt.model, opt.dataset))
if os.path.exists(data_path):
    data = torch.load(data_path, map_location=device)
    print("load data from {}".format(data_path))
else:
    data = feature_extraction(opt)

hidden_states = data["hidden_states"]
labels = data["ori_labels"]
hidden_states = hidden_states.view(hidden_states.size(0),-1)
hidden_states = np.array(hidden_states)
labels = np.array(labels)


TP_nu, FP_nu = [], []
F_score_nu = []


test_power_cls = []
for target_label in range(0,5):
    TP_k, FP_k, auc_k, test_power_k = [], [], [], []
    for k in range(opt.repeat_time):
        (hidden_states, labels) = shuffle(hidden_states, labels, random_state=3047+k)
        target_class = [target_label]

        # Generate train data
        X = hidden_states[labels==target_class]
        X_train = X[:int(X.shape[0]*0.8)]
        # Generate some regular novel observations
        X_test = X[int(X.shape[0]*0.8):]
        # Generate some abnormal novel observations
        X_outliers = hidden_states[labels!=target_class]

        # fit the model
        clf = svm.OneClassSVM(kernel="rbf",
                              nu=0.5,
                              )
        clf.fit(X_train)
        y_pred_train = clf.predict(X_train)
        y_pred_test = clf.predict(X_test)
        y_pred_outliers = clf.predict(X_outliers)
        n_error_train = y_pred_train[y_pred_train == -1].size / y_pred_train.size
        n_error_test = y_pred_test[y_pred_test == -1].size / y_pred_test.size
        n_error_outliers = y_pred_outliers[y_pred_outliers == 1].size / y_pred_outliers.size

        scores_test = clf.score_samples(X_test)
        scores_ood = clf.score_samples(X_outliers)
        score_threshold = np.percentile(scores_test, 5)  # FPR=0.05 i.e., Type-I error=0.05
        TP = scores_ood < score_threshold
        TPR = np.sum(TP)/len(scores_ood)  # TPR@0.05FPR i.e., test power at Type-I error=0.05
        test_power_k.append(TPR)
        test_power_mean = np.mean(np.array(test_power_k))
        print(f"test_power_k: {test_power_k}, \ntest_power_mean: {test_power_mean}")

    test_power_cls.append(test_power_mean)
    print(f"test_power_cls: {test_power_cls}")
