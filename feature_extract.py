
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
import argparse
import numpy as np

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
    criterion = nn.CrossEntropyLoss()
    loss = 0.0
    total_correct = 0.0
    total_sample = 0.0
    avg_acc = 0.0
    for batch in tqdm(dataloader_train):
        batch = {k: v.to(device) for k, v in batch.items()}
        labels = batch.pop('labels')
        # labels = labels.float()
        outputs = classifier(**batch)
        logits = torch.squeeze(outputs)
        loss = criterion(logits, labels)
        total_sample += labels.shape[0]
        predictions = torch.argmax(logits, 1)
        correct = torch.sum(predictions == labels)
        total_correct += correct
        avg_acc = total_correct * 100.0 / total_sample
        loss.backward()
        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()
    print(f"train loss:{loss}, avg_acc:{avg_acc.item()}")


def eval_step(classifier, dataloader_test, opt):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    classifier.eval()
    criterion = nn.CrossEntropyLoss()
    total_correct = 0.0
    total_sample = 0.0
    avg_acc = 0.0
    logits_test = []
    hidden_states = []
    ori_labels = []
    for batch in tqdm(dataloader_test):
        batch = {k: v.to(device) for k, v in batch.items()}
        labels = batch.pop('labels')
        # labels = labels.float()
        total_sample += labels.shape[0]
        with torch.no_grad():
            logits = classifier(**batch)
            logits = torch.squeeze(logits)
            loss = criterion(logits, labels)
            predictions = torch.argmax(logits, 1)
            correct = torch.sum(predictions == labels)
            total_correct += correct
            avg_acc = total_correct * 100.0 / total_sample
            logits_test.append(logits)
            hidden_state = classifier.intermediate(**batch)
            hidden_states.append(hidden_state.to(torch.device('cpu')))
            ori_labels.append(labels.to(torch.device('cpu')))
    avg_acc = avg_acc.item()
    print(f"\n test loss:{loss}, avg_acc:{avg_acc}")
    # save feature
    data = {"hidden_states": torch.cat(hidden_states, dim=0),
            "ori_labels": torch.cat(ori_labels, 0),
            }
    ckpt_folder = f"./checkpoints/"
    ckpt_path = os.path.join(ckpt_folder, "data_{}_{}.pt".format(opt.model, opt.dataset))
    torch.save(data, ckpt_path)
    print(f"save feature data at: {ckpt_path}")

    return avg_acc, data

def feature_extraction(opt):
    # os.environ['TRANSFORMERS_CACHE'] = '../huggingface'
    os.environ["CUDA_VISIBLE_DEVICES"] = str(opt.gpu)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"torch device info:{device}")

    train_text, train_label, test_text, test_label = [], [], [], []
    dataset_name = opt.dataset
    pkl_file_list = os.listdir(f"./log/{dataset_name}/temp_{opt.temperature}")
    pkl_file_list1 = [item for item in pkl_file_list if ("merged" not in item
                                                         # and str(opt.temperature) in item
                                                         )]
    column_name = "generated_code"
    for _, pklfile in enumerate(pkl_file_list1):
        pklfile_path = os.path.join("./log", dataset_name, "temp_{}".format(opt.temperature), pklfile)
        if "codegen" in pklfile_path:
            lb = 0
        elif "PolyCoder" in pklfile_path:
            lb = 1
        elif "incoder" in pklfile_path:
            lb = 2
        elif "gpt-neo" in pklfile_path:
            lb = 3
        elif "codet5" in pklfile_path:
            lb = 4
        else:
            raise Exception("invalid pklfile")
        label = lb
        df = pandas.read_pickle(pklfile_path)
        print("load file: {} , len:{}".format(pklfile, df.__len__()))
        texts = df[column_name].to_list()
        labels = [label] * df.__len__()
        X_train, X_test, y_train, y_test = train_test_split(texts, labels, test_size=0.2, random_state=42)
        train_text.extend(X_train)
        train_label.extend(y_train)
        test_text.extend(X_test)
        test_label.extend(y_test)
    # concatenate all code snippets for feature extraction
    test_text.extend(train_text)
    test_label.extend(train_label)
    print("train_text:{}, train_label:{}, test_text:{}, test_label:{}".format(len(train_text), len(train_label), len(test_text), len(test_label)))
    print(opt)

    # tokenization
    ## models
    pretrained_model_name_or_path_dic = {
        "codebert-base": "microsoft/codebert-base",
        "unixcoder-base-nine": "microsoft/unixcoder-base-nine",
        "bert-base-uncased": "bert-base-uncased",
    }
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path_dic[opt.model])

    time1 = time.time()
    max_length = 256
    encodings_test = tokenizer(test_text, padding="max_length", max_length=max_length, truncation=True)
    dataset_test = CodeDataset(encodings_test, test_label)
    time2 = time.time()
    print(f"data loading time: {time2-time1}s")
    batchsize = opt.batchsize
    dataloader_test = DataLoader(dataset_test, shuffle=True, batch_size=batchsize)

    num_classes = len(set(train_label))
    assert len(set(train_label)) == len(set(test_label))
    classifier = TransformerModel(num_classes=num_classes,
                                  BaseModel_remote=pretrained_model_name_or_path_dic[opt.model]).to(device)

    _, data = eval_step(classifier, dataloader_test, opt)

    return data

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
    parser.add_argument("--checkpoints", type=str, default="./checkpoints")
    parser.add_argument("--batchsize", type=int, default=16)
    opt = parser.parse_args()

    feature_extraction(opt)



