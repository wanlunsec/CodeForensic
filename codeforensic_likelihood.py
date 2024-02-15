
import torch
from torch.nn import CrossEntropyLoss
import numpy as np
from scipy.special import softmax
from datasets import load_dataset, load_from_disk
from transformers import AutoTokenizer, AutoModelForCausalLM, CodeGenForCausalLM, T5ForConditionalGeneration
import argparse
import os
from tqdm import tqdm
import pandas
from sklearn.metrics import roc_auc_score

"""
config
"""
parser = argparse.ArgumentParser()
parser.add_argument("--gpu", type=str, default=1)
parser.add_argument("--dataset", type=str, default="mbpp")
parser.add_argument("--model", type=str, default="codegen-6B-mono")
parser.add_argument("--source_model", type=str, default="incoder-6B")
parser.add_argument("--max_new_tokens", type=int, default=256)
parser.add_argument("--temperature", type=float, default=0.2)
parser.add_argument("--num_sample", type=int, default=100)
opt = parser.parse_args()

# os.environ['TRANSFORMERS_CACHE'] = '../huggingface'
os.environ["CUDA_VISIBLE_DEVICES"] = str(opt.gpu)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"torch device info:{device}")

pretrained_model_name_or_path_dic = {
    "codegen-6B-mono": "Salesforce/codegen-6B-mono",
    "codegen-6B-multi": "Salesforce/codegen-6B-multi",
    "codegen-350M-mono": "Salesforce/codegen-350M-mono",
    "PolyCoder-2.7B": "NinedayWang/PolyCoder-2.7B",
    "gpt-neo-2.7B": "EleutherAI/gpt-neo-2.7B",
    "incoder-6B": "facebook/incoder-6B",
    "incoder-1B": "facebook/incoder-1B",
    "codet5-base": "Salesforce/codet5-base",
}

if "incoder" in opt.model:
    kwargs = dict()
else:
    kwargs = dict(low_cpu_mem_usage=True)
tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path_dic[opt.model])
if "codet5" in opt.model:
    model = T5ForConditionalGeneration.from_pretrained(pretrained_model_name_or_path_dic[opt.model], **kwargs).to(device)
else:
    model = AutoModelForCausalLM.from_pretrained(pretrained_model_name_or_path_dic[opt.model], **kwargs).to(device)

model.eval()

## datasets
if opt.dataset == 'mbxp-all':
    dataset_list = ['mbxp-java', 'mbxp-js', 'mbxp-go', 'mbxp-php', 'mbxp-ruby', 'mbxp-cs', 'mbxp-cpp'] #'mbpp'
else:
    dataset_list = [opt.dataset]

TPR_dataset = np.zeros([1, len(dataset_list)])
for dataset_index, dataset_name in enumerate(dataset_list):
    pkl_file_dir = f"./log/{dataset_name}/temp_{opt.temperature}"
    pkl_file_list = os.listdir(pkl_file_dir)
    pkl_file_list1 = [item for item in pkl_file_list if ("merged" not in item
                                                         # and str(opt.temperature) in item
                                                         )]
    column_name = "generated_code"
    df_list = {}
    for _, pklfile in enumerate(pkl_file_list1):
        pklfile_path = os.path.join(pkl_file_dir, pklfile)
        if "codegen" in pklfile_path:
            df = pandas.read_pickle(pklfile_path)
            df_list.update({"codegen":df})
        elif "PolyCoder" in pklfile_path:
            df = pandas.read_pickle(pklfile_path)
            df_list.update({"PolyCoder":df})
        elif "incoder" in pklfile_path:
            df = pandas.read_pickle(pklfile_path)
            df_list.update({"incoder":df})
        elif "gpt-neo" in pklfile_path:
            df = pandas.read_pickle(pklfile_path)
            df_list.update({"gpt":df})
        elif "codet5" in pklfile_path:
            df = pandas.read_pickle(pklfile_path)
            df_list.update({"codet5":df})
        else:
            raise Exception("invalid pklfile")

    TPR_list = []
    for k in range(5):
        ppl_in = []
        ppl_out = []
        for target_model in list(df_list.keys()):
            df = df_list[target_model]
            ppls = []
            num_sample = min(df.__len__(), opt.num_sample)
            if target_model not in str(opt.model):
                num_sample = num_sample // 4
            np.random.seed(seed=1102 * (k+1))
            df_indices = np.random.choice(df.__len__(), num_sample, replace=False)
            df_sub = df.loc[df_indices]
            for i in tqdm(range(0, num_sample)):
                index_column, prompt_column = "task_id", "prompt"
                # print("-"*50, i, "-"*50)
                df_index = df_indices[i]
                sample = df_sub.loc[df_index]
                task_id = sample[index_column]
                prompt = sample["prompt"]
                completion = sample["generated_code"]
                code = prompt + completion + "<|endoftext|>"
                # print("code:", code)
                # print("completion:", completion)

                prompt_input_ids = tokenizer(prompt, return_tensors="pt").input_ids
                start_loc = prompt_input_ids.size(1)
                encodings = tokenizer(code, return_tensors="pt")
                input_ids = encodings.input_ids
                # revert token_id to token
                indices = input_ids[0].detach().tolist()
                all_tokens = tokenizer.convert_ids_to_tokens(indices)
                # print("after tokenization: {}".format(all_tokens))

                stride = 512
                if opt.model in ["gpt-neo-2.7B", "incoder-6B", "PolyCoder-2.7B"]:
                    max_length = model.config.max_position_embeddings
                else:
                    max_length = model.config.n_positions

                seq_len = encodings.input_ids.size(1)
                # print("max_length:{}, \t seq_len:{}".format(max_length, seq_len))

                nlls = []
                prev_end_loc = 0
                for begin_loc in range(0, seq_len, stride):
                    # print("begin_loc: {}".format(begin_loc))
                    end_loc = min(begin_loc + max_length, seq_len)
                    trg_len = end_loc - prev_end_loc - start_loc  # may be different from stride on last loop
                    input_ids = encodings.input_ids[:, begin_loc:end_loc].to(device)
                    target_ids = input_ids.clone()
                    # the default value of ignore_index in torch.nn.CrossEntropyLoss() is -100
                    target_ids[:, :-trg_len] = -100
                    labels = target_ids
                    # print("input_ids:", input_ids)
                    # print("target_ids:", target_ids)
                    with torch.no_grad():
                        outputs = model(input_ids, labels=target_ids, output_hidden_states=True)
                        ce_loss = outputs.loss

                        # loss is calculated using CrossEntropyLoss which averages over input tokens.
                        # Multiply it with trg_len to get the summation instead of average.
                        # We will take average over all the tokens to get the true average
                        # in the last step of this example.
                        neg_log_likelihood = ce_loss * trg_len

                    nlls.append(neg_log_likelihood)

                    prev_end_loc = end_loc
                    if end_loc == seq_len:
                        break

                ppl = torch.exp(torch.stack(nlls).sum() / (end_loc-start_loc))
                # print("nlls: {}, \t ppl: {}".format(nlls, ppl))
                ppls.append([task_id, ppl.cpu().item()])
                # print(ppls)
                if target_model in str(opt.model):
                    ppl_in.append(ppl.cpu().item())
                else:
                    ppl_out.append(ppl.cpu().item())

                if i % 50 == 49:  # save results for every 50 samples
                    df_result = pandas.DataFrame(ppls, columns=["task_id", "ppl"])
                    ppl_save_dir = "./log/ppl"
                    if not os.path.exists(ppl_save_dir):
                        os.makedirs(ppl_save_dir)
                    ppl_save_path = os.path.join(ppl_save_dir, f"ppl_{dataset_name}_{opt.model}_{opt.max_new_tokens}_{target_model}_.csv")
                    df_result.to_csv(ppl_save_path)

        FPR_threshold = np.percentile(ppl_in, 95)  # FPR fixed at 0.05
        TPR = sum(np.array(ppl_out) > FPR_threshold) / len(ppl_out)
        print("TPR@0.5FPR:", TPR)
        TPR_list.append(TPR)
        TPR_mean = np.mean(TPR_list)
        print("TPR_list:", TPR_list, "TPR_mean:", TPR_mean)
        TPR_dataset[0, dataset_index] = TPR_mean
        print("dataset:", dataset_name, "TPR_dataset:", TPR_dataset)

