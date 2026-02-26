import os
import numpy as np
from datasets import load_dataset, interleave_datasets
from tqdm import tqdm
from tokenizer import SimpleTokenizer, train_tokenizer
from config import config

def standardize_column(example):
    if 'content' in example and example['content']:
        return {'text': example['content']}
    elif 'problem' in example and 'solution' in example:
        return {'text': f"Problem: {example['problem']}\nSolution: {example['solution']}"}
    elif 'messages' in example:
        full_text = ""
        for msg in example['messages']:
            role, content = msg['role'], msg['content']
            full_text += f"<|im_start|>{role}\n{content}<|im_end|>\n"
        return {'text': full_text}
    elif 'text' in example:
        return {'text': example['text']}
    return {'text': ""}

def load_all_datasets():
    ds_edu = load_dataset("HuggingFaceTB/smollm-corpus", "fineweb-edu-dedup", split="train", streaming=True)
    ds_python = load_dataset("HuggingFaceTB/smollm-corpus", "python-edu", split="train", streaming=True)
    ds_math = load_dataset("HuggingFaceTB/finemath", "finemath-4plus", split="train", streaming=True) 
    ds_cosmo = load_dataset("HuggingFaceTB/smollm-corpus", "cosmopedia-v2", split="train", streaming=True)
    ds_talk = load_dataset("HuggingFaceTB/smoltalk", "all", split="train", streaming=True)

    datasets = [ds_edu, ds_python, ds_math, ds_cosmo, ds_talk]
    return [d.map(standardize_column, remove_columns=list(d.features.keys())) for d in datasets]

def create_dataset_iterators():
    ds_edu, ds_python, ds_math, ds_cosmo, ds_talk = load_all_datasets()
    return {
        "web": iter(ds_edu),
        "python": iter(ds_python),
        "cosmo": iter(ds_cosmo),
        "math": iter(ds_math),
        "talk": iter(ds_talk)
    }

def process_sequence(ids, labels, sample, tokenizer, seq_len, pad_token_id):
    is_chat = 'messages' in sample
    
    if len(ids) > seq_len:
        if is_chat:
            ids = ids[:seq_len]
            labels = labels[:seq_len]
            end_token = tokenizer.encode("<|im_end|>")[0]
            ids[-1] = end_token
            labels[-1] = end_token
        else:
            ids = ids[:seq_len]
            labels = labels[:seq_len]
    
    pad_len = seq_len - len(ids)
    if pad_len > 0:
        ids.extend([pad_token_id] * pad_len)
        labels.extend([-100] * pad_len)
    
    return ids, labels

def get_stage_mixture(stage_name):
    ds_edu, ds_python, ds_math, ds_cosmo, ds_talk = load_all_datasets()
    
    stage_map = {"STAGE_1": 1, "STAGE_2": 2, "STAGE_3": 3, "STAGE_4": 4}
    stage_num = stage_map.get(stage_name)
    
    if stage_num is None:
        raise ValueError(f"Unknown stage: {stage_name}")
    
    ratios = config['stage_ratios'][stage_num]
    
    datasets_list = []
    probabilities = []
    dataset_mapping = {
        "web": ds_edu,
        "python": ds_python,
        "math": ds_math,
        "cosmo": ds_cosmo,
        "talk": ds_talk
    }
    
    for domain, ratio in ratios.items():
        if ratio > 0.0:
            datasets_list.append(dataset_mapping[domain])
            probabilities.append(ratio)
    
    seeds = {"STAGE_1": 42, "STAGE_2": 43, "STAGE_3": 44, "STAGE_4": 45}
    current_seed = seeds.get(stage_name, 42)
    
    return interleave_datasets(datasets_list, probabilities=probabilities, seed=current_seed)

def get_tokenizer_mixture():
    ds_edu, ds_python, ds_math, ds_cosmo, ds_talk = load_all_datasets()
    
    return interleave_datasets(
        [ds_edu, ds_python, ds_math, ds_cosmo, ds_talk],
        probabilities=[0.40, 0.20, 0.15, 0.15, 0.10], 
        seed=999
    )

def format_for_sft(example, tokenizer):
    all_ids = []
    all_labels = []
    
    if example.get('messages'):
        for msg in example['messages']:
            role, content = msg['role'], msg['content']
            prefix = f"<|im_start|>{role}\n"
            suffix = "<|im_end|>\n"
            
            prefix_ids = tokenizer.encode(prefix)
            content_ids = tokenizer.encode(content)
            suffix_ids = tokenizer.encode(suffix)
            
            msg_ids = prefix_ids + content_ids + suffix_ids
            
            if role == "assistant":
                msg_labels = [-100] * len(prefix_ids) + content_ids + suffix_ids
            else:
                msg_labels = [-100] * len(msg_ids)
            
            all_ids.extend(msg_ids)
            all_labels.extend(msg_labels)
    
    else:
        text = example.get('text', '')
        if not text: return [], []
        
        all_ids = tokenizer.encode(text + "<|endoftext|>")
        all_labels = all_ids.copy()
    
    return all_ids, all_labels

def prepare_validation_data(batch_size=16):
    dest = config['dataset_path']
    tok_json = os.path.join(config['tokenizer_path'], "tokenizer.json")
    tokenizer = SimpleTokenizer(tok_json)
    
    seq_len = config['block_size'] + 1 
    pad_token_id = tokenizer.encode("<|endoftext|>")[0]
    
    print("Loading datasets for validation batches...")
    iters = create_dataset_iterators()

    print(f"Generating validation batches (Batch Size: {batch_size}, Seq Len: {seq_len})...")
    
    for stage, ratios in config['stage_ratios'].items():
        bin_data_path = os.path.join(dest, f"val_stage_{stage}_data.bin")
        bin_labels_path = os.path.join(dest, f"val_stage_{stage}_labels.bin")
        
        print(f"  -> Building Stage {stage} validation batch...")
        
        with open(bin_data_path, "wb") as f_data, open(bin_labels_path, "wb") as f_labels:
            sequences_collected = 0
            
            for domain, ratio in ratios.items():
                if ratio == 0.0:
                    continue
                
                n_seqs = int(batch_size * ratio)
                
                for _ in range(n_seqs):
                    sample = next(iters[domain])
                    
                    ids, labels = format_for_sft(sample, tokenizer)
                    ids, labels = process_sequence(ids, labels, sample, tokenizer, seq_len, pad_token_id)
                    
                    np.array(ids, dtype=np.uint16).tofile(f_data)
                    np.array(labels, dtype=np.int32).tofile(f_labels)
                    
                    sequences_collected += 1
                    
            while sequences_collected < batch_size:
                majority_domain = max(ratios, key=ratios.get)
                sample = next(iters[majority_domain])
                
                ids, labels = format_for_sft(sample, tokenizer)
                ids, labels = process_sequence(ids, labels, sample, tokenizer, seq_len, pad_token_id)
                
                np.array(ids, dtype=np.uint16).tofile(f_data)
                np.array(labels, dtype=np.int32).tofile(f_labels)
                
                sequences_collected += 1
                
    print(f"All 5 validation batches generated in {dest}!")

def prepare_sft_data():
    dest = config['dataset_path']
    bin_data_path = os.path.join(dest, "sft_data.bin")
    bin_labels_path = os.path.join(dest, "sft_labels.bin")
    tok_json = os.path.join(config['tokenizer_path'], "tokenizer.json")
    
    sft_target_tokens = config['stf_target_tokens']
    
    print(f"Loading SFT datasets from HuggingFace...")
    ds_edu, ds_python, ds_math, ds_cosmo, ds_talk = load_all_datasets()
    
    ratios = config['stage_ratios'][5]
    
    datasets_list = []
    probabilities = []
    dataset_mapping = {
        "web": ds_edu,
        "python": ds_python,
        "math": ds_math,
        "cosmo": ds_cosmo,
        "talk": ds_talk
    }
    
    for domain, ratio in ratios.items():
        if ratio > 0.0:
            datasets_list.append(dataset_mapping[domain])
            probabilities.append(ratio)
    
    ds_mixed = interleave_datasets(datasets_list, probabilities=probabilities, seed=46)
    tokenizer = SimpleTokenizer(tok_json)
    
    seq_len = config['block_size'] + 1 
    pad_token_id = tokenizer.encode("<|endoftext|>")[0]
    
    print(f"SFT binary encoding in {bin_data_path}...")
    count = 0
    pbar = tqdm(total=sft_target_tokens, unit='tok')
    
    with open(bin_data_path, "wb") as f_data, open(bin_labels_path, "wb") as f_labels:
        for ex in ds_mixed:
            ids, labels = format_for_sft(ex, tokenizer)
            
            if len(ids) < 10: 
                continue
            
            if (np.array(labels) != -100).sum() == 0: 
                continue
            
            is_chat = 'messages' in ex
            
            if len(ids) > seq_len:
                if is_chat:
                    continue 
                else:
                    ids = ids[:seq_len]
                    labels = labels[:seq_len]
                    
                    end_token = tokenizer.encode("<|im_end|>")[0]
                    ids[-1] = end_token
                    labels[-1] = end_token

            pad_len = seq_len - len(ids)
            if pad_len > 0:
                ids.extend([pad_token_id] * pad_len)
                labels.extend([-100] * pad_len)
            
            np.array(ids, dtype=np.uint16).tofile(f_data)
            np.array(labels, dtype=np.int32).tofile(f_labels)
            
            count += seq_len
            pbar.update(seq_len)
            
            if count >= sft_target_tokens: 
                break
            
    pbar.close()
    print(f"SFT dataset and labels are encoded !")

def get_data():
    dest = config['dataset_path']
    tmp_txt = os.path.join(dest, "temp_sample.txt")
    bin_path = os.path.join(dest, "data.bin")
    tok_json = os.path.join(config['tokenizer_path'], "tokenizer.json")

    if not os.path.exists(tok_json):
        print(f"Collecting a sample of all datasets (Web, Python, Math, Code, Cosmo) for Tokenizers in a temp file... (~1.5GB)")
        with open(tmp_txt, "w", encoding="utf-8") as f:
            ds_sample = get_tokenizer_mixture()
            bytes_written = 0
            for ex in tqdm(ds_sample, total=1.5*10**9, unit='B', unit_scale=True):
                line = ex['text'] + "<|endoftext|>\n"
                f.write(line)
                bytes_written += len(line.encode('utf-8'))
                if bytes_written > 1.5 * 1024**3: 
                    break 

        print("Training tokenizer...")
        train_tokenizer(tmp_txt, tok_json, vocab_size=config['vocab_size'])
        os.remove(tmp_txt)
        print("Temp file was deleted !")

    tokenizer = SimpleTokenizer(tok_json)
    
    target_tokens = config['pre_training_target_tokens']
    
    stages = [
        ("STAGE_1", 0.4 * target_tokens),  
        ("STAGE_2", 0.3 * target_tokens),  
        ("STAGE_3", 0.2 * target_tokens), 
        ("STAGE_4", 0.1 * target_tokens),  
    ]

    print(f"Pre training binary encoding in {bin_path}...")
    with open(bin_path, "wb") as f_bin:
        for stage_name, target in stages:
            ds_mixed = get_stage_mixture(stage_name)
            count = 0
            
            pbar = tqdm(total=target, unit='tok', desc=f"Processing {stage_name}")
            
            for ex in ds_mixed:
                text = ex['text'] + "<|endoftext|>"
                
                if len(text) < 20: 
                    continue 
                
                ids = tokenizer.encode(text)
                if ids:
                    np.array(ids, dtype=np.uint16).tofile(f_bin)
                    count += len(ids)
                    pbar.update(len(ids))
                
                if count >= target: 
                    break
            pbar.close()
    
    print(f"Encoding was finished ! Total : {target_tokens/1e9:.3f}B tokens over 4 stages.")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepare-sft", action="store_true", help="Preparing SFT datasets")
    parser.add_argument("--validation", action="store_true", help="Generate 5 validation batches encoded in .bin")
    args = parser.parse_args()
    
    os.makedirs(config['dataset_path'], exist_ok=True)
    os.makedirs(config['tokenizer_path'], exist_ok=True)
    
    if args.prepare_sft:
        prepare_sft_data()
        os._exit(0)
    elif args.validation:
        prepare_validation_data() 
        os._exit(0)
    else:
        get_data()
        os._exit(0)