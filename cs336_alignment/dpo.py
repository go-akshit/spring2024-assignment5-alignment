import torch
import gzip
import json

def hh_dataset(file_paths, output_file_path):
    dataset = []
    for file_path in file_paths:
        category = file_path.split('/')[-1]
        with gzip.open(file_path, 'rt', 'utf-8') as f:
            for line in f:
                example = json.loads(line)
                chosen = example['chosen']
                reject = example['rejected']
                if chosen.count('\n\nHuman') > 1:
                    continue
                else: 
                    len_human_title = len('\n\nHuman: ')
                    len_assistant_title = len('\n\nAssistant: ')
                    
                    chosen_human_index = chosen.index('\n\nHuman:')
                    chosen_assistant_index = chosen.index('\n\nAssistant:')
                    reject_human_index = reject.index('\n\nHuman:')
                    reject_assistant_index = reject.index('\n\nAssistant:')
    
                    human = chosen[chosen_human_index + len_human_title:chosen_assistant_index]
                    
                    chosen_assistant = chosen[chosen_assistant_index + len_assistant_title:]
                    reject_assistant = reject[reject_assistant_index + len_assistant_title:]
                    entry = {}
                    entry['instruction'] = human
                    entry['chosen'] = chosen_assistant
                    entry['rejected'] = reject_assistant
                    entry['category'] = category
                    dataset.append(entry)
    
    with open(output_file_path, 'w') as f:
        for entry in dataset:
            f.write(json.dumps(entry) + '\n')


# def dpo_loss(lm, lm_ref, tokenizer, beta, prompt, response_chosen, response_rejected):
#     yw_x = f"{tokenizer.bos_token}Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\n{prompt}\n\n### Response:\n{response_chosen}"
#     yl_x = f"{tokenizer.bos_token}Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\n{prompt}\n\n### Response:\n{response_rejected}"
#     tokenized_yw_x = tokenizer.encode(yw_x, add_special_tokens=False)
#     tokenized_yl_x = tokenizer.encode(yl_x, add_special_tokens=False)

    


def main():
    input_file_paths = ['/home/shared/hh/helpful-base.jsonl.gz',
                        '/home/shared/hh/helpful-rejection-sampled.jsonl.gz',
                        '/home/shared/hh/helpful-online.jsonl.gz',
                        '/home/shared/hh/harmless-base.jsonl.gz']
    output_file_path = 'hh_dataset.jsonl'
    hh_dataset(input_file_paths, output_file_path)  

if __name__ == '__main__':
    main()

