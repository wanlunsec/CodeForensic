
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, T5ForConditionalGeneration
from datasets import load_dataset, load_from_disk
import os
import argparse
import pickle
import pandas
from tqdm import tqdm
import time
"""
config
"""
parser = argparse.ArgumentParser()
parser.add_argument("--gpu", type=str, default=0)
parser.add_argument("--dataset", type=str, default="mbpp")
parser.add_argument("--model", type=str, default="codegen-6B-mono")
parser.add_argument("--max_new_tokens", type=int, default=256)
parser.add_argument("--temperature", type=float, default=0.2)
parser.add_argument("--top_p", type=float, default=0.95)
parser.add_argument("--max_generation", type=int, default=1000)

opt = parser.parse_args()

# os.environ['TRANSFORMERS_CACHE'] = '../huggingface'
if opt.model == "codegen-16B-mono":
    os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1"
else:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(opt.gpu)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"torch device info:{device}")

torch.random.seed()

def save_variable(data):
    path = f"log/"
    if not os.path.exists(path):
        os.makedirs(path)
    filename = f'solution_{opt.dataset}_{opt.model}_.pickle'
    filepath = os.path.join(path, filename)
    with open(filepath, 'wb') as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)
    return filepath

def load_variavle(filename):
    with open(filename, 'rb') as handle:
        r = pickle.load(handle)
    return r


if os.path.exists("../huggingface") :
    TRANSFORMERS_CACHE = "../huggingface/hub/"
    DATASETS_CACHE = None # "../huggingface/datasets"
else:
    TRANSFORMERS_CACHE = None
    DATASETS_CACHE = None


if opt.dataset == "humaneval":
    ds = load_dataset("openai_humaneval", split="test")
    index_column, prompt_column = "task_id", "prompt"
elif opt.dataset == "apps":
    ds = load_dataset("codeparrot/apps", split="test")
    index_column, prompt_column = "problem_id", "question"
elif opt.dataset == "mbpp":
    ds = load_from_disk("../huggingface/datasets/mbpp-merged")
    index_column, prompt_column = "task_id", "text"
elif opt.dataset == "mbxp-java":
    ds = load_from_disk("../huggingface/datasets/mbxp-java")
    index_column, prompt_column = "task_id", "prompt"
elif opt.dataset == "mbxp-js":
    ds = load_from_disk("../huggingface/datasets/mbxp-js")
    index_column, prompt_column = "task_id", "prompt"
elif opt.dataset == "mbxp-go":
    ds = load_from_disk("../huggingface/datasets/mbxp-go")
    index_column, prompt_column = "task_id", "prompt"
elif opt.dataset == "mbxp-py":
    ds = load_from_disk("../huggingface/datasets/mbxp-py")
    index_column, prompt_column = "task_id", "prompt"
elif opt.dataset == "mbxp-ruby":
    ds = load_from_disk("../huggingface/datasets/mbxp-ruby")
    index_column, prompt_column = "task_id", "prompt"
elif opt.dataset == "mbxp-php":
    ds = load_from_disk("../huggingface/datasets/mbxp-php")
    index_column, prompt_column = "task_id", "prompt"
elif opt.dataset == "mbxp-cs":
    ds = load_from_disk("../huggingface/datasets/mbxp-cs")
    index_column, prompt_column = "task_id", "prompt"
elif opt.dataset == "mbxp-cpp":
    ds = load_from_disk("../huggingface/datasets/mbxp-cpp")
    index_column, prompt_column = "task_id", "prompt"
else:
    raise Exception("Invalid Dataset")




pretrained_model_name_or_path_dic = {
    "codegen-6B-mono": "Salesforce/codegen-6B-mono",
    "codegen-6B-multi": "Salesforce/codegen-6B-multi",
    "codegen-16B-mono": "Salesforce/codegen-16B-mono",
    "codegen-2B-mono": "Salesforce/codegen-2B-mono",
    "codegen-350M-mono": "Salesforce/codegen-350M-mono",
    "PolyCoder-2.7B": "NinedayWang/PolyCoder-2.7B",
    "gpt-neo-2.7B": "EleutherAI/gpt-neo-2.7B",
    "codet5-base": "Salesforce/codet5-base",
}

tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path_dic[opt.model])
if opt.model == "codegen-16B-mono":
    model = AutoModelForCausalLM.from_pretrained(pretrained_model_name_or_path_dic[opt.model],
                                                 # revision="float16",
                                                 # torch_dtype=torch.float16,
                                                 low_cpu_mem_usage=True,
                                                 device_map="auto",
                                                 )
elif "codet5" in opt.model:
    model = T5ForConditionalGeneration.from_pretrained(pretrained_model_name_or_path_dic[opt.model], low_cpu_mem_usage=True)
else:
    model = AutoModelForCausalLM.from_pretrained(pretrained_model_name_or_path_dic[opt.model],
                                                 # revision="float16",
                                                 # torch_dtype=torch.float16,
                                                 low_cpu_mem_usage=True).to(device)

dataset_list = [opt.dataset] #['mbxp-java', 'mbxp-js', 'mbxp-go', 'mbxp-php', 'mbxp-ruby', 'mbxp-cs', 'mbxp-cpp'] # ['mbpp'] #
for dataset_name in dataset_list:
    opt.dataset = dataset_name
    for _sample_idx in range(0, 1):
        solution_code = []
        task_id_generated = []
        # resume from saved results
        save_dir = f"./log/{opt.dataset}/temp_{opt.temperature}"
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        file_name = f"solution_{opt.dataset}_{opt.model}_{opt.max_new_tokens}_{opt.temperature}_{opt.top_p}_100{_sample_idx+1}_.pkl"
        save_path = os.path.join(save_dir, file_name)
        if os.path.exists(save_path):
            print(f"resume from previous results at: {save_path}")
            df = pandas.read_pickle(save_path)
            for task_id, prompt, full_text, result in zip(df["task_id"].to_list(), df["prompt"].tolist(),
                                                          df["full_text"].to_list(), df["generated_code"].to_list()):
                solution_code.append([task_id, prompt, full_text, result])
                task_id_generated.append(task_id)
        else:
            print(f"Not found: {save_path}")
        for i in tqdm(range(0, ds.__len__())):
            torch.manual_seed(1102*_sample_idx + i)
            torch.cuda.manual_seed(1102*_sample_idx + i)
            # print("--"*50,i)
            sample = ds[i]
            # print(sample)
            problem_id = sample[index_column]
            if problem_id in task_id_generated:
                continue
            text = sample[prompt_column]
            text = text.replace("\n\n\n", "\n\n")
            # print(text)
            input_ids = tokenizer(text, return_tensors="pt").input_ids.to(device)
            start = input_ids.size(1)
            max_length = opt.max_new_tokens + start
            if max_length > 2048:
                print("warning: max_length {} is greater than the context window {}".format(max_length, 2048))
                continue
            generated_ids = model.generate(input_ids, do_sample=True,
                                           max_new_tokens=opt.max_new_tokens,
                                           top_p=opt.top_p,
                                           temperature=opt.temperature,
                                           )
            # print("="*50)
            full_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True, clean_up_tokenization_spaces=False)
            result = tokenizer.decode(generated_ids[0][start:], skip_special_tokens=True, clean_up_tokenization_spaces=False)
            # print(result)
            solution_code.append([problem_id, text, full_text, result])
            if i % 50 == 49:  # save results for every 50 samples
                df = pandas.DataFrame(solution_code, columns=["task_id", "prompt", "full_text", "generated_code"])
                df.to_pickle(save_path)
                time_string = time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime())
                print(time_string, f"total number of results:{df.__len__()}", f"save_path: {save_path}")
            if len(solution_code) > opt.max_generation:
                break

        ### pandas
        df = pandas.DataFrame(solution_code, columns=["task_id", "prompt", "full_text", "generated_code"])
        df.to_pickle(save_path)
        time_string = time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime())
        print(time_string, f"total number of results:{df.__len__()}", f"save_path: {save_path}")




# python -u codegen_test.py --dataset mbpp --model codegen-6B-mono --temperature 0.2 --top_p 1.0 --max_generation 51 --gpu 1 > output1.log 2>&1 &
# 16G
# python -u codegen_test.py --dataset mbpp --model PolyCoder-2.7B --temperature 1.0 --top_p 0.95 --max_generation 51 --gpu 0 > output2.log 2>&1 &
# 15g
# python -u codegen_test.py --dataset mbpp --model gpt-neo-2.7B --temperature 1.0 --top_p 0.95 --max_generation 51 --gpu 0 > output3.log 2>&1 &
# 15g
# python -u codet5_test.py --dataset mbpp --model codet5-base --temperature 0.2 --top_p 0.95 --max_generation 1000 --gpu 1 > output5.log 2>&1 &
# 2.4G


