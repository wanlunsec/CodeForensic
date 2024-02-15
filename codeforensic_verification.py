"""
This code is modified from its original version:
https://github.com/fengliu90/DK-for-TST/blob/master/Deep_Kernel_HDGM.py
"Learning deep kernels for two-sample testing"
"""
import numpy as np
import torch
import argparse
import os
from sklearn.utils import shuffle
from tqdm import tqdm

from utils_HD import MatConvert, TST_MMD_adaptive_bandwidth, Pdist2
from feature_extract import feature_extraction

class ModelLatentF(torch.nn.Module):
    """Latent space for both domains."""
    def __init__(self, x_in, H, x_out):
        """Init latent features."""
        super(ModelLatentF, self).__init__()
        self.restored = False
        self.latent = torch.nn.Sequential(
            torch.nn.Linear(x_in, H, bias=True),
            torch.nn.Softplus(),
            torch.nn.Linear(H, H, bias=True),
            torch.nn.Softplus(),
            torch.nn.Linear(H, H, bias=True),
            torch.nn.Softplus(),
            torch.nn.Linear(H, x_out, bias=True),
        )
    def forward(self, input):
        """Forward the LeNet."""
        fealant = self.latent(input)
        return fealant


# parameters to generate data
parser = argparse.ArgumentParser()
parser.add_argument('--n', type=int, default=750)  # number of samples per mode
parser.add_argument('--d', type=int, default=768)  # dimension of samples (default value is 10)

parser.add_argument("--gpu", type=str, default=1)
parser.add_argument("--checkpoints", type=str, default="./checkpoints")
parser.add_argument("--dataset", type=str, default="mbpp")
parser.add_argument("--model", type=str, default="unixcoder-base-nine")
parser.add_argument("--max_new_tokens", type=int, default=256)
parser.add_argument("--temperature", type=float, default=0.2)
parser.add_argument("--batchsize", type=int, default=16)
parser.add_argument("--save_ckpt", type=bool, default=False)
parser.add_argument("--target_label", type=int, default=1)

opt = parser.parse_args()


# opt.model = "codebert-base"
os.environ["CUDA_VISIBLE_DEVICES"] = str(opt.gpu)
is_cuda = False #True
dtype = torch.float
device = torch.device("cuda" if torch.cuda.is_available() and is_cuda else "cpu")

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
(hidden_states, labels) = shuffle(hidden_states, labels)

test_power_N_te_all = np.zeros([5, 20])
for target_label in range(0, 5):
    opt.target_label = target_label
    all_class = list(np.arange(5))
    target_class = [opt.target_label]
    source_class = []
    for i in all_class:
        if i in target_class:
            continue
        else:
            source_class.append(i)
    sample_num = 150
    code_feature_target = hidden_states[np.where(labels==target_class[0])][0:sample_num]
    code_feature_test = hidden_states[np.where(labels!=target_class[0])][0:sample_num*3]

    # Setup seeds
    np.random.seed(1102)
    torch.manual_seed(1102)
    torch.cuda.manual_seed(1102)
    torch.backends.cudnn.deterministic = True
    # Setup for experiments
    N_per = 200 # permutation times
    alpha = 0.05 # test threshold
    d = opt.d # dimension of data
    n = 56  # number of samples in per mode for training DK
    print('n: '+str(n)+' d: '+str(d))
    x_in = d # number of neurons in the input layer, i.e., dimension of data
    H = 256  # number of neurons in the hidden layer
    x_out = 64  # number of neurons in the output layer
    learning_rate = 0.00005 # default learning rate for MMD-D on HDGM
    N_epoch = 1 # number of training epochs default: 1000
    K = 5 # number of trails default: 10
    N = 100 # # number of test sets
    N_f = N*1.0 # number of test sets (float)


    test_power_N_te = []
    for num_test_sample in range(1,9):
        # Naming variables
        Results = np.zeros([1,K])
        J_star_u = np.zeros([N_epoch])
        J_star_adp = np.zeros([N_epoch])
        ep_OPT = np.zeros([K])
        s_OPT = np.zeros([K])
        s0_OPT = np.zeros([K])
        # Repeat experiments K times (K = 10) and report average test power (rejection rate)
        for kk in range(K):
            torch.manual_seed(kk * 19 + n)
            torch.cuda.manual_seed(kk * 19 + n)
            np.random.seed(seed=kk * 19 + n)

            # Generate training data of P
            Ind_all = np.arange(len(code_feature_target))
            Ind_tr = np.random.choice(len(Ind_all), n, replace=False)
            Ind_te = np.delete(Ind_all, Ind_tr)

            # Generate training data of Q
            np.random.seed(seed=819 * (kk + 9) + n)
            Ind_v4_all = np.arange(len(code_feature_test))
            Ind_tr_v4 = np.random.choice(len(Ind_v4_all), n, replace=False)
            Ind_te_v4 = np.delete(Ind_v4_all, Ind_tr_v4)

            # Train deep kernel to maximize test power
            np.random.seed(seed=1102)
            torch.manual_seed(1102)
            torch.cuda.manual_seed(1102)

              # --------------------------------------------
            # Compute test power of deep kernel based MMD
            # --------------------------------------------
            H_u = np.zeros(N)
            T_u = np.zeros(N)
            M_u = np.zeros(N)
            np.random.seed(1102)
            count_u = 0
            for k in range(N):
                # Fetch test data
                np.random.seed(seed=1102 * (k + 1) + n)
                data_all_te = code_feature_target[Ind_te]
                data_trans_te = code_feature_test[Ind_te_v4]
                N_te = num_test_sample*5
                np.random.seed(seed=1102 * (k+1) + n)
                # print(f"len(data_all_te):{len(data_all_te)},  N_te*2:{N_te*2}")
                ind1 = np.random.choice(len(data_all_te), N_te*2, replace=False)
                np.random.seed(seed=819 * (k+2) + n)
                ind2 = np.random.choice(len(data_trans_te), N_te, replace=False)
                s1 = data_all_te[ind1[:N_te]]
                s2 = data_trans_te[ind2]
                # REPLACE above line with
                # s2 = data_all_te[ind1[N_te:]]
                # for validating type-I error (s1 ans s2 are from the same distribution)
                # print("test data: s1: {}, s2: {}".format(s1.size(), s2.size()))
                S = np.concatenate((s1, s2), axis=0)
                S = MatConvert(S, device, dtype)
                # Run two sample test on generated data
                Dxy = Pdist2(S[:N_te, :], S[N_te:, :])
                # baseline: MMD_adaptive
                sigma0 = Dxy.median()
                sigma = 0
                h_adaptive, threshold_adaptive, mmd_value_adaptive = TST_MMD_adaptive_bandwidth(S, N_per, N_te, S, sigma, sigma0, alpha, device, dtype)
                h_u = h_adaptive
                # Gather results
                count_u = count_u + h_u
                # print("kk:",kk, "MMD-DK:", count_u)
                H_u[k] = h_u
            # Print test power of MMD-D
            print(f"Test Power of MMD-D with alpha={alpha}: ", H_u.sum() / N_f)
            Results[0, kk] = H_u.sum() / N_f
            print("Test Power of MMD-D (K times): ", Results[0])
            print(f"target_label: {target_label}. N_te:{N_te}. Average Test Power of MMD-D with alpha={alpha}: ", Results[0].sum() / (kk + 1))
        ave_test_power = Results[0].sum() / (kk + 1)
        test_power_N_te.append(ave_test_power)
        print(f"test_power_N_te: {test_power_N_te}")
        test_power_N_te_all[target_label, num_test_sample-1] = ave_test_power
        if ave_test_power > 0.99:
            break
    # np.save(f'./Results_{opt.dataset}', test_power_N_te_all)


