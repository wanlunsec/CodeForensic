
from datasets import load_dataset, Dataset
import pandas
import os


def mbpp_merge():
    """
    Under the hood, Dataset.set_format() changes the return format for the dataset’s __getitem__() dunder method.
    This means that when we want to create a new object like train_df from a Dataset in the "pandas" format,
    we need to slice the whole dataset to obtain a pandas.DataFrame.
    create a pandas.DataFrame for the whole training set by selecting all the elements of drug_dataset["train"]:
    train_df = drug_dataset["train"][:]
    """
    dataset = load_dataset("mbpp")

    data1 = dataset["prompt"]
    data1.set_format("pandas")  # changes the output format of the dataset
    df1 = data1[:]

    data2 = dataset["test"]
    data2.set_format("pandas")
    df2 = data2[:]

    data3 = dataset["validation"]
    data3.set_format("pandas")
    df3 = data3[:]

    data4 = dataset["train"]
    data4.set_format("pandas")
    df4 = data4[:]

    df = pandas.concat([df1, df2, df3, df4], keys=["prompt", "test", "validation", "train"])
    df_new = df.reset_index()  # save index as columns in df

    # converting from DataFrame to Dataset
    dataset_full = Dataset.from_pandas(df_new)
    # save dataset to disk in Arrow format

    dataset_full.save_to_disk("../huggingface/datasets/mbpp-merged")
    # reload for check
    dataset2 = Dataset.load_from_disk("../huggingface/datasets/mbpp-merged")


def mbxp_toHFdataset(ori_name, new_name):
    file_path = f"../mbxp-exec-eval-main/data/mbxp/{ori_name}_release_v1.jsonl"
    df_raw = pandas.read_json(path_or_buf=file_path, lines=True)
    # converting from DataFrame to Dataset
    dataset_full = Dataset.from_pandas(df_raw)
    # save dataset to disk in Arrow format
    dataset_full.save_to_disk(f"../huggingface/datasets/{new_name}")
    # reload for check
    dataset2 = Dataset.load_from_disk(f"../huggingface/datasets/{new_name}")

if __name__ == "__main__":

    HF_dataroot = "../huggingface"
    if not os.path.exists(HF_dataroot):
        os.makedirs(HF_dataroot)
    # pre-process mbpp dataset
    mbpp_merge()

    # pre-process mbxp datasets
    """
    please first DOWNLOAD the original datasets from:
    https://github.com/amazon-science/mbxp-exec-eval
    Then unzip the repo, and put the it in the same directory as ./CodeForensic
    """
    mbxp_namedict={"mbjp": "mbxp-java",
                   "mbjsp": "mbxp-js",
                   "mbphp": "mbxp-php",
                   "mbrbp": "mbxp-ruby",
                   "mbgp": "mbxp-go",
                   "mbcsp": "mbxp-cs",
                   "mbcpp": "mbxp-cpp",
                   }
    for ori_name in list(mbxp_namedict.keys()):
        new_name = mbxp_namedict[ori_name]
        mbxp_toHFdataset(ori_name, new_name)
