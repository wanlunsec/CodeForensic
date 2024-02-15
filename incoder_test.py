
from typing import List
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset, load_from_disk
import tokenizers
import json
import os
import argparse
import csv
import pickle
import pandas
from tqdm import tqdm
import time


"""
config
"""
parser = argparse.ArgumentParser()
parser.add_argument("--gpu", type=str, default=1)
parser.add_argument("--dataset", type=str, default="mbpp")
parser.add_argument("--model", type=str, default="incoder-6B")
parser.add_argument("--max_new_tokens", type=int, default=256)
parser.add_argument("--temperature", type=float, default=0.2)
parser.add_argument("--top_p", type=float, default=0.95)
parser.add_argument("--max_generation", type=int, default=1000)
opt = parser.parse_args()

# os.environ['TRANSFORMERS_CACHE'] = '../huggingface'
os.environ["CUDA_VISIBLE_DEVICES"] = str(opt.gpu)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"torch device info:{device}")


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

def make_sentinel(i):
    # signals (1) a location to insert an infill and (2) the start of the infill generation
    return f"<|mask:{i}|>"

def generate(input: str, max_to_generate: int=128, temperature: float=0.2):
    """
    Do standard left-to-right completion of the prefix `input` by sampling from the model
    """
    input_ids = tokenizer(input, return_tensors="pt").input_ids
    if CUDA:
        input_ids = input_ids.cuda()
    max_length = max_to_generate + input_ids.flatten().size(0)
    if max_length > 2048:
        print("warning: max_length {} is greater than the context window {}".format(max_length, 2048))
    with torch.no_grad():
        output = model.generate(input_ids=input_ids, do_sample=True, top_p=opt.top_p, temperature=temperature, max_length=max_length)
    # pass clean_up_tokenization_spaces=False to avoid removing spaces before punctuation, e.g. "from ." -> "from."
    detok_hypo_str = tokenizer.decode(output.flatten(), clean_up_tokenization_spaces=False)
    if detok_hypo_str.startswith(BOS):
        detok_hypo_str = detok_hypo_str[len(BOS):]
    return detok_hypo_str

def infill(parts: List[str], max_to_generate: int=128, temperature: float=0.2, extra_sentinel: bool=True, max_retries: int=1):
    """
    Generate infills to complete a partial document, e.g.
    [A C E] -> [A B C D E], where B and D are infills that have been generated.
    parts: List[str]. list of parts of the document. One string will be
            inserted in between each element, i.e. infilling N-1 locations for a list
            of length N.
    max_to_generate: int. maximum number of tokens to generate. Keep in mind
            that the model context size is 2048.
    temperature: float. temperature parameter for sampling.
    extra_sentinel: bool. we recommend setting this to True, as it makes it
            easier for the model to end generated infills. See the footnote in
            section 2.2 of our paper for details.
    max_retries: int. if > 1, use rejection sampling to keep sampling infills until
            all infills sample a completion token.
    returns a dictionary containing the following:
        text:  str, the completed document (with infills inserted)
        parts:  List[str], length N. Same as passed to the method
        infills:  List[str], length N-1. The list of infills generated
        retries_attempted:  number of retries used (if max_retries > 1)
    """
    assert isinstance(parts, list)
    retries_attempted = 0
    done = False

    while (not done) and (retries_attempted < max_retries):
        retries_attempted += 1

        if VERBOSE:
            print(f"retry {retries_attempted}")

        ## (1) build the prompt
        if len(parts) == 1:
            prompt = parts[0]
        else:
            prompt = ""
            # encode parts separated by sentinel
            for sentinel_ix, part in enumerate(parts):
                prompt += part
                if extra_sentinel or (sentinel_ix < len(parts) - 1):
                    prompt += make_sentinel(sentinel_ix)

        infills = []
        complete = []

        done = True

        ## (2) generate infills
        for sentinel_ix, part in enumerate(parts[:-1]):
            complete.append(part)
            prompt += make_sentinel(sentinel_ix)
            # TODO: this is inefficient as it requires re-encoding prefixes repeatedly
            completion = generate(prompt, max_to_generate, temperature)
            completion = completion[len(prompt):]
            if EOM not in completion:
                if VERBOSE:
                    print(f"warning: {EOM} not found")
                completion += EOM
                done = False
            completion = completion[:completion.index(EOM) + len(EOM)]
            infilled = completion[:-len(EOM)]
            infills.append(infilled)
            complete.append(infilled)
            prompt += completion
        complete.append(parts[-1])
        text = ''.join(complete)

    if VERBOSE:
        print("generated text:")
        print(prompt)
        print()
        print("parts:")
        print(parts)
        print()
        print("infills:")
        print(infills)
        print()
        print("restitched text:")
        print(text)
        print()

    return {
        'text': text, # str, the completed document (with infills inserted)
        'parts': parts, # List[str], length N. Same as passed to the method
        'infills': infills, # List[str], length N-1. The list of infills generated
        'retries_attempted': retries_attempted, # number of retries used (if max_retries > 1)
    }

tokenizers_version = tuple(int(n) for n in tokenizers.__version__.split('.'))
if tokenizers_version < (0, 12, 1):
    print("warning: Your tokenizers version looks old and you will likely have formatting issues. We recommend installing tokenizers >= 0.12.1")

# set BIG_MODEL to use the 6.7B parameter model
if opt.model == "incoder-6B":
    BIG_MODEL = True
elif opt.model == "incoder-1B":
    BIG_MODEL = False
else:
    raise Exception(f"Invalid Model: {opt.model}")

# use a GPU
CUDA = True

# print intermediate outputs of infilling
VERBOSE = False

if BIG_MODEL:
    model_name = "facebook/incoder-6B"

    # the arguments added below will load a half precision version of the model,
    # which requires less RAM than loading the full float32 version.  this
    # should fit in ~16GB of RAM
    # NOTE: half precision should *not* be used if you plan to fine-tune the
    # model. You'll need full precision and a lot of GPU memory. We have not
    # tested fine-tuning in `transformers` (the model was trained in fairseq)
    if CUDA:
        kwargs = dict(
            revision="float16",
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
        )
    else:
        kwargs = dict(
            low_cpu_mem_usage=True,
        )
else:
    model_name = "facebook/incoder-1B"
    kwargs = {}

if os.path.exists("../huggingface") :
    TRANSFORMERS_CACHE = "../huggingface/hub/"
    DATASETS_CACHE = None # "../huggingface/datasets"
else:
    TRANSFORMERS_CACHE = None
    DATASETS_CACHE = None

print("loading model")
model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)
print("loading tokenizer")
tokenizer = AutoTokenizer.from_pretrained(model_name)
print("loading complete")

if CUDA:
    # if you plan to fine-tune the model, you should not use half precision.
    model = model.half().cuda()

# signals the start of a document
BOS = "<|endoftext|>"
# signals the end of a generated infill
EOM = "<|endofmask|>"


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


dataset_list = [opt.dataset] #['mbxp-java', 'mbxp-js', 'mbxp-go', 'mbxp-php', 'mbxp-ruby', 'mbxp-cs', 'mbxp-cpp']
for dataset_name in dataset_list:
    opt.dataset = dataset_name
    max_to_generate, temperature = opt.max_new_tokens, opt.temperature
    for _sample_idx in range(1):
        solution_code = []
        task_id_generated = []
        # resume from saved results
        save_dir = f"./log/{opt.dataset}/temp_{opt.temperature}"
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        file_name = f"solution_{opt.dataset}_{opt.model}_{opt.max_new_tokens}_{opt.temperature}_{opt.top_p}_100{_sample_idx+1}_.pkl"
        save_path = os.path.join(save_dir, file_name)
        if os.path.exists(save_path):
            df = pandas.read_pickle(save_path)
            for task_id, prompt, full_text, result in zip(df["task_id"].to_list(), df["prompt"].tolist(),
                                                          df["full_text"].to_list(), df["generated_code"].to_list()):
                solution_code.append([task_id, prompt, full_text, result])
                task_id_generated.append(task_id)
        for i in tqdm(range(0, ds.__len__())):
            torch.manual_seed(1102*_sample_idx + i)
            torch.cuda.manual_seed(1102*_sample_idx + i)
            # print("--"*50,i)
            sample = ds[i]
            # print(sample)
            problem_id = sample[index_column]
            if problem_id in task_id_generated:
                continue
            text = sample[prompt_column] + "\n    <insert>\n<|/ file |>"
            text = text.replace("\n\n\n", "\n\n")
            # print(text)
            parts = text.split("<insert>")
            result = infill(parts, max_to_generate, temperature)
            # print("completed document:")
            # print(result["text"])
            # print("="*50)
            # print(result["infills"][0])
            solution_code.append([problem_id, text, result["text"], result["infills"][0]])
            if i % 50 == 49:  # save results for every 50 samples
                df = pandas.DataFrame(solution_code, columns=["task_id", "prompt", "full_text", "generated_code"])
                df.to_pickle(save_path)
                time_string = time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime())
                print(time_string,f"total number of results:{df.__len__()}", f"save_path: {save_path}")
            if len(solution_code) > opt.max_generation:
                break

        ### pandas
        df = pandas.DataFrame(solution_code, columns=["task_id", "prompt", "full_text", "generated_code"])
        df.to_pickle(save_path)
        time_string = time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime())
        print(time_string,f"total number of results:{df.__len__()}", f"save_path: {save_path}")

# python -u incoder_test.py --dataset mbpp  --temperature 0.2 --top_p 0.95 --max_generation 51 --gpu 0 > output4.log 2>&1 &
# 14.6G

