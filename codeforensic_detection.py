
import os
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
from datasets import load_dataset, load_from_disk
from transformers import get_scheduler
from transformers import RobertaTokenizer, RobertaModel, AutoTokenizer, AutoModel
import time
from tqdm.auto import tqdm
import pandas
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import numpy as np
import argparse


class CodeDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

class RobertaClassificationHead(nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(classifier_dropout)
        self.out_proj = nn.Linear(config.hidden_size, config.num_labels)

    def forward(self, features, **kwargs):
        x = features[:, 0, :]  # take <s> token (equiv. to [CLS])
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x


class TransformerModel(nn.Module):
    def __init__(self, num_classes=2, BaseModel_remote="microsoft/unixcoder-base-nine"):
        super(TransformerModel, self).__init__()
        BaseModel = AutoModel.from_pretrained(BaseModel_remote)
        self.config = BaseModel.config
        self.config.num_labels = num_classes
        self.f = BaseModel
        self.classifier = RobertaClassificationHead(self.config)

    def forward(self, *arg, **kwargs):
        bertoutput = self.f(*arg, **kwargs)
        sequence_output = bertoutput['last_hidden_state']
        out = self.classifier(sequence_output)
        return out

    def intermediate(self, *arg, **kwargs):
        bertoutput = self.f(*arg, **kwargs)
        sequence_output = bertoutput['last_hidden_state']
        return sequence_output[:, 0, :]



def train_step(classifier, optimizer, lr_scheduler, dataloader_train, epoch):
    print(f"\n {'-'*20} epoch:{epoch+1} {'-'*20} \n ")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    classifier.train()
    criterion = nn.BCEWithLogitsLoss()
    loss = 0.0
    total_correct = 0.0
    total_sample = 0.0
    avg_acc = 0.0
    for batch in tqdm(dataloader_train):
        batch = {k: v.to(device) for k, v in batch.items()}
        labels = batch.pop('labels')
        labels = labels.float()
        outputs = classifier(**batch)
        logits = torch.squeeze(outputs)
        loss = criterion(logits, labels)
        total_sample += labels.shape[0]
        predictions = torch.where(logits > 0.5, 1, 0)  # torch.argmax(logits, 1)
        correct = torch.sum(predictions == labels)
        total_correct += correct
        avg_acc = total_correct * 100.0 / total_sample
        loss.backward()
        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()
    print(f"train loss:{loss}, avg_acc:{avg_acc.item()}")


def eval_step(classifier, dataloader_test, opt, save_ckpt=False):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    classifier.eval()
    criterion = nn.BCEWithLogitsLoss()
    total_correct = 0.0
    total_sample = 0.0
    avg_acc = 0.0
    logits_test = []
    hidden_states = []
    ori_labels = []
    pred_labels = []
    for batch in tqdm(dataloader_test):
        batch = {k: v.to(device) for k, v in batch.items()}
        labels = batch.pop('labels')
        labels = labels.float()
        total_sample += labels.shape[0]
        with torch.no_grad():
            logits = classifier(**batch)
            logits = torch.squeeze(logits)
            loss = criterion(logits, labels)
            predictions = torch.where(logits > 0.5, 1, 0)  # torch.argmax(logits, 1)
            correct = torch.sum(predictions == labels)
            total_correct += correct
            avg_acc = total_correct * 100.0 / total_sample
            logits_test.append(logits)
            hidden_state = classifier.intermediate(**batch)
            hidden_states.append(hidden_state.to(torch.device('cpu')))
            ori_labels.append(labels.to(torch.device('cpu')))
            pred_labels.append(predictions.to(torch.device('cpu')))
    labs = torch.cat(ori_labels, 0).numpy()
    preds = torch.cat(pred_labels, 0).numpy()
    auc_score = roc_auc_score(labs, preds)
    avg_acc = avg_acc.item()
    print(f"\n test loss:{loss}, avg_acc:{avg_acc}, auc: {auc_score*100.0}")

    # save model ckpt
    if save_ckpt:
        state_dict = classifier.state_dict()
        ckpt_folder = os.path.join(opt.checkpoints)
        if not os.path.exists(ckpt_folder):
            os.makedirs(ckpt_folder)
        ckpt_path = os.path.join(ckpt_folder, "{}_{}_{}_{}_ckpt.pth.tar".format(opt.model, opt.dataset, opt.src_gen, opt.temperature))
        torch.save(state_dict, ckpt_path)
        print("save ckpt at {}".format(ckpt_path))
    return avg_acc, auc_score*100.0


if __name__ == "__main__":
    """
    config
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=str, default=0)
    parser.add_argument("--dataset", type=str, default="mbpp")
    parser.add_argument("--model", type=str, default="unixcoder-base-nine")
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--checkpoints", type=str, default="./checkpoints")
    parser.add_argument("--save_ckpt", type=bool, default=False)
    parser.add_argument("--eval_only", type=bool, default=False)
    parser.add_argument("--num_epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-6)
    parser.add_argument("--batchsize", type=int, default=16)
    parser.add_argument("--resume", type=bool, default=False)
    parser.add_argument("--finetune_ll", type=bool, default=False)
    parser.add_argument("--src_gen", type=str, default="codegen-6B-mono")
    parser.add_argument("--test_gen", type=str, default="PolyCoder-2.7B")
    parser.add_argument("--repeat_time", type=int, default=5)
    opt = parser.parse_args()

    # os.environ['TRANSFORMERS_CACHE'] = '../huggingface'
    os.environ["CUDA_VISIBLE_DEVICES"] = str(opt.gpu)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"torch device info:{device}")

    if os.path.exists("../huggingface"):
        TRANSFORMERS_CACHE = None #"../huggingface/hub/"
        DATASETS_CACHE = None # "../huggingface/datasets"
    else:
        TRANSFORMERS_CACHE = None
        DATASETS_CACHE = None


    auc_list_k = {"codegen": [], "PolyCoder": [], "incoder": [], "gpt-neo": [], "codet5": []}
    acc_list_k = {"codegen": [], "PolyCoder": [], "incoder": [], "gpt-neo": [], "codet5": []}
    for k in range(opt.repeat_time):
        ## datasets
        # real code
        train_text, train_label, test_text, test_label = [], [], [], []
        if opt.dataset == "humaneval":
            ds = load_dataset("openai_humaneval", split="test")
            index_column, prompt_column = "task_id", "prompt"
        elif opt.dataset == "mbpp":
            ds = load_from_disk("../huggingface/datasets/mbpp-merged")
            index_column, prompt_column = "task_id", "text"
        elif opt.dataset == "apps":
            ds = load_dataset("codeparrot/apps", split="test")
            index_column, prompt_column = "problem_id", "question"
        else:
            raise Exception("Invalid Dataset")

        if opt.dataset in ["humaneval", "mbxp-py"]:
            solution_column = "canonical_solution"
        elif opt.dataset in ["apps"]:
            solution_column = "solutions"
        else:
            solution_column = "code"

        task_ids = ds[index_column]
        code = ds[solution_column]
        label = [1] * len(code)
        X_train_r, X_test_r, y_train_r, y_test_r, train_task_ids, test_task_ids = train_test_split(code, label, task_ids, test_size=0.2, random_state=42*k)
        train_text.extend(X_train_r)
        train_label.extend(y_train_r)
        test_text.extend(X_test_r)
        test_label.extend(y_test_r)
        print("real code: ", len(train_text), len(test_text))

        # neural code
        # training set
        index_column, prompt_column = "task_id", "prompt"
        pkl_file_dir = f"./log/{opt.dataset}/temp_{opt.temperature}"  # f"./log/{opt.dataset}/top_p_{opt.top_p}"  #
        pkl_file_list = os.listdir(pkl_file_dir)
        pkl_file_list1 = [item for item in pkl_file_list if ("merged" not in item
                                                             # and "1001" in item # for HumanEval
                                                             )]
        column_name = "generated_code"
        for i, pklfile in enumerate(pkl_file_list1):
            print("load file: {}".format(pklfile))
            pklfile_path = os.path.join(pkl_file_dir, pklfile)
            if opt.src_gen not in pklfile_path:
                continue
            df = pandas.read_pickle(pklfile_path)
            task_ids = df[index_column].to_list()
            texts = df[column_name].to_list()
            labels = [0] * df.__len__()
            print("neural code train:", len(labels))
            # X_train, X_test, y_train, y_test = train_test_split(texts, labels, test_size=0.2, random_state=42)
            X_train, y_train = [], []
            for task_id, text, label in zip(task_ids, texts, labels):
                if task_id in train_task_ids:
                    X_train.append(text)
                    y_train.append(label)
            train_text.extend(X_train)
            train_label.extend(y_train)
            print("source: {}, len: {}".format(pklfile_path, df.__len__()))

        # neural code
        # test set
        test_temperature = 1.0  # opt.temperature
        test_dataset = opt.dataset
        pkl_file_dir = f"./log/{test_dataset}/temp_{test_temperature}"
        pkl_file_list = os.listdir(pkl_file_dir)
        pkl_file_list1 = [item for item in pkl_file_list if ("merged" not in item
                                                             # and "1001" in item
                                                             )]
        column_name = "generated_code"
        for i, pklfile in enumerate(pkl_file_list1):
            print("load file: {}".format(pklfile))
            pklfile_path = os.path.join(pkl_file_dir, pklfile)
            if opt.test_gen not in pklfile_path:
                continue
            df = pandas.read_pickle(pklfile_path)
            task_ids = df[index_column].to_list()
            texts = df[column_name].to_list()
            labels = [0] * df.__len__()
            print("neural code test:", len(labels))
            # X_train, X_test, y_train, y_test = train_test_split(texts, labels, test_size=0.2, random_state=42)
            X_test, y_test = [], []
            for task_id, text, label in zip(task_ids, texts, labels):
                if task_id in test_task_ids:
                    X_test.append(text)
                    y_test.append(label)
            test_text.extend(X_test)
            test_label.extend(y_test)
            print("test: {}, len: {}".format(pklfile_path, df.__len__()))
        print("train_text:{}, train_label:{}, test_text:{}, test_label:{}".format(len(train_text), len(train_label), len(test_text), len(test_label)))

        if opt.eval_only:
            opt.num_epochs = 1
            opt.resume = True
            opt.save_ckpt = False
            print("*************** eval_only mode ***************")
        print(opt)

        # exit()
        # tokenization
        ## models
        pretrained_model_name_or_path_dic = {
            "unixcoder-base-nine": "microsoft/unixcoder-base-nine",
            "bert-base-uncased": "bert-base-uncased",
        }

        tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path_dic[opt.model])

        time1 = time.time()
        max_length = 256
        encodings_train = tokenizer(train_text, padding="max_length", max_length=max_length, truncation=True)
        dataset_train = CodeDataset(encodings_train, train_label)
        encodings_test = tokenizer(test_text, padding="max_length", max_length=max_length, truncation=True)
        dataset_test = CodeDataset(encodings_test, test_label)
        time2 = time.time()
        print(f"data loading time: {time2-time1}s")
        batchsize = opt.batchsize
        dataloader_train = DataLoader(dataset_train, shuffle=True, batch_size=batchsize)
        dataloader_test = DataLoader(dataset_test, shuffle=True, batch_size=batchsize)

        classifier = TransformerModel(num_classes=1,
                                      BaseModel_remote=pretrained_model_name_or_path_dic[opt.model]).to(device)

        if opt.resume:
            ckpt_folder = os.path.join(opt.checkpoints)
            ckpt_path = os.path.join(ckpt_folder, "{}_{}_{}_{}_ckpt.pth.tar".format(opt.model, opt.dataset, opt.src_gen, opt.temperature))
            if os.path.exists(ckpt_path):
                classifier.load_state_dict(torch.load(ckpt_path))
                print(f"load pretrained classifier from {ckpt_path}")
            else:
                print(f"Do not find ckpt at {ckpt_path}")

        finetune = opt.finetune_ll
        if finetune == True:
            optmz_param = []
            for name, children in classifier.named_children():
                if name in ['f']:
                    for param in children.parameters():
                        param.requires_grad = False
                else:
                    for param in children.parameters():
                        optmz_param.append(param)
            optimizer = torch.optim.AdamW(optmz_param, lr=opt.lr)
            print('-'*50,'Finetune')
        else:
            optimizer = torch.optim.AdamW(classifier.parameters(), lr=opt.lr)

        num_epochs = opt.num_epochs
        num_training_steps = num_epochs * len(dataloader_train)
        lr_scheduler = get_scheduler("linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps)
        best_acc = 0.0
        auc_list = {"codegen": 0.0, "PolyCoder": 0.0, "incoder": 0.0, "gpt-neo": 0.0, "codet5": 0.0}
        acc_list = {"codegen": 0.0, "PolyCoder": 0.0, "incoder": 0.0, "gpt-neo": 0.0, "codet5": 0.0}
        for epoch in range(num_epochs):
            if not opt.eval_only:
                train_step(classifier, optimizer, lr_scheduler, dataloader_train, epoch)
            acc, auc = eval_step(classifier, dataloader_test, opt, save_ckpt=opt.save_ckpt)
            if acc < 99:
                best_acc = acc
                continue
            else:
                break
        for model_name in ["codegen", "PolyCoder", "incoder", "gpt-neo", "codet5"]:
            if model_name in opt.test_gen:
                auc_list[model_name] = auc
                acc_list[model_name] = acc

        # evaluation on other dataset
        # test_temperature = 1.0 #opt.temperature  #
        pkl_file_dir = f"./log/{opt.dataset}/top_p_{test_temperature}" #f"./log/{opt.dataset}/temp_{test_temperature}" #
        pkl_file_list = os.listdir(pkl_file_dir)
        pkl_file_list1 = [item for item in pkl_file_list if ("merged" not in item
                                                             # and "1001" in item # for test on HumanEval
                                                             )]
        column_name = "generated_code"
        for i, pklfile in enumerate(pkl_file_list1):
            print("load file: {}".format(pklfile))
            pklfile_path = os.path.join(pkl_file_dir, pklfile)
            if opt.test_gen in pklfile_path:
                continue
            df = pandas.read_pickle(pklfile_path)
            task_ids = df[index_column].to_list()
            texts = df[column_name].to_list()
            # label = [i] * df.__len__()
            labels = [0] * df.__len__()
            # X_train, X_test, y_train, y_test = train_test_split(texts, labels, test_size=0.2, random_state=42)
            X_test, y_test, test_text, test_label = [], [], [], []
            for task_id, text, label in zip(task_ids, texts, labels):
                if task_id in test_task_ids:
                    X_test.append(text)
                    y_test.append(label)
            test_text.extend(X_test)
            test_label.extend(y_test)
            test_text.extend(X_test_r)
            test_label.extend(y_test_r)
            print("test: {}, len: {}".format(pklfile_path, df.__len__()))
            encodings_test = tokenizer(test_text, padding="max_length", max_length=max_length, truncation=True)
            dataset_test = CodeDataset(encodings_test, test_label)
            dataloader_test = DataLoader(dataset_test, shuffle=True, batch_size=batchsize)
            acc, auc = eval_step(classifier, dataloader_test, opt)
            for model_name in ["codegen", "PolyCoder", "incoder", "gpt-neo", "codet5"]:
                if model_name in pklfile_path:
                    auc_list[model_name] = auc
                    acc_list[model_name] = acc
        print("acc:", acc_list, "\nauc:", auc_list)
        for model_name in ["codegen", "PolyCoder", "incoder", "gpt-neo", "codet5"]:
            auc_list_k[model_name].append(auc_list[model_name])
            acc_list_k[model_name].append(acc_list[model_name])
        print("acc_k:", acc_list_k, "\nauc_k:", auc_list_k)
        auc_list_mean = {"codegen": 0.0, "PolyCoder": 0.0, "incoder": 0.0, "gpt-neo": 0.0, "codet5": 0.0}
        acc_list_mean = {"codegen": 0.0, "PolyCoder": 0.0, "incoder": 0.0, "gpt-neo": 0.0, "codet5": 0.0}
        for model_name in ["codegen", "PolyCoder", "incoder", "gpt-neo", "codet5"]:
            auc_list_mean[model_name] = np.mean(np.array(auc_list_k[model_name]))
            acc_list_mean[model_name] = np.mean(np.array(acc_list_k[model_name]))
        print("acc_mean:", acc_list_mean, "\nauc_mean:", auc_list_mean)

# train
# python codeforensic_detection.py  --dataset mbpp --src_gen codegen-6B-mono  --test_gen PolyCoder-2.7B --gpu 0


