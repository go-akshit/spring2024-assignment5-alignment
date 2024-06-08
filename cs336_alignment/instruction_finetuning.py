import torch
import json
import random
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch.nn.functional as F
import math
import logging
import argparse
import gzip
from memory_profiler import profile
import wandb
from tqdm import tqdm

logger = logging.getLogger(__name__)


class finetuning_dataset(Dataset):
    #@profile
    def __init__(self, tokenizer, dataset_path, seq_length, shuffle):
        dataset_path = str(dataset_path)
        self.tokenizer = tokenizer
        self.dataset_path = dataset_path
        self.seq_length = seq_length
        self.shuffle = shuffle
        self.mydata = []
        all_documents = []
        tokenized_all_documents = []
        if dataset_path.endswith('.gz'):
            open_func = gzip.open
            mode = 'rt'
            encoding = 'utf-8'
        else:
            open_func = open
            mode = 'r'
            encoding = 'utf-8'
        with open_func(dataset_path, mode, encoding=encoding) as file:
            for line in file:
                prompt_response_pair = json.loads(line.strip())
                prompt = prompt_response_pair['prompt']
                response = prompt_response_pair['response']
                document = f"<|begin_of_text|>Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\n{prompt}\n\n### Response:\n{response}"
                all_documents.append(document)

                
        if shuffle:
            random.shuffle(all_documents)
        
        #tokenized_text = tokenizer.encode(concatenated_documents, add_special_tokens=False)
        tokenized_all_documents = []
        for idx, document in enumerate(all_documents):
            if idx != len(all_documents) - 1:
                document += "<|end_of_text|>"
            tokenized_document = tokenizer.encode(document, add_special_tokens=False)
            tokenized_all_documents.extend(tokenized_document)
            
        for i in range(0, len(tokenized_all_documents) - seq_length - 1, seq_length):
            input_ids = tokenized_all_documents[i:i + seq_length]
            input_ids = torch.tensor(input_ids)
            labels = tokenized_all_documents[i + 1:i + seq_length + 1]
            labels = torch.tensor(labels)
            self.mydata.append({"input_ids": input_ids, "labels": labels})

    
    def __len__(self):
        return len(self.mydata)
    
    def __getitem__(self, i):
        return self.mydata[i]
    
#@profile
def get_dataloader(dataset, batch_size, shuffle):
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

def get_cosine_lr(
    it: int,
    max_learning_rate: float,
    min_learning_rate: float,
    warmup_iters: int,
    cosine_cycle_iters: int,
):
    """Cosine with warmup learning rate scheduler."""
    # First, we linearly warmup for warmup_iters steps.
    if it < warmup_iters:
        return max_learning_rate * it / warmup_iters
    # Then, if it > cosine_cycle_iters, we return min learning rate.
    if it > cosine_cycle_iters:
        return min_learning_rate
    # Else, we use cosine decay down to min learning rate.
    decay_ratio = (it - warmup_iters) / (cosine_cycle_iters - warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_learning_rate + coeff * (max_learning_rate - min_learning_rate)

def train_finetuning(args):
    if args.model_name == 'base':
        model_path = '/data/Meta-Llama-3-8B'
    elif args.model_name == 'instruct':
        model_path = '/home/shared/Meta-Llama-3-70B-Instruct'

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype = torch.bfloat16,
        attn_implementation = "flash_attention_2",
    )
    model.to(device)
    logger.info(f"CS336-Assn5: Model loaded from {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)

    train_dataset = finetuning_dataset(tokenizer, args.train_dataset_path, args.seq_length, args.shuffle)
    logger.info(f"CS336-Assn5: Training dataset loaded from {args.train_dataset_path}")
    train_dataloader = get_dataloader(train_dataset, args.batch_size, args.shuffle)
    test_dataset = finetuning_dataset(tokenizer, args.test_dataset_path, args.seq_length, args.shuffle)
    logger.info(f"CS336-Assn5: Test dataset loaded from {args.test_dataset_path}")
    test_dataloader = get_dataloader(test_dataset, args.batch_size, args.shuffle)

    logger.info(f"CS336-Assn5: Training started")
    for epoch in range(args.epochs):
        for idx, batch in enumerate(tqdm(train_dataloader, desc="Training Epoch Progress")):
            total_iter = epoch * len(train_dataloader) + idx
            if args.lr_scheduler.lower() == "cosine":
                lr = get_cosine_lr(
                    total_iter,
                    max_learning_rate=args.learning_rate,
                    min_learning_rate=args.learning_rate * 0.1,
                    warmup_iters=int(args.train_steps * args.warmup_ratio),
                    cosine_cycle_iters=args.train_steps,
                )
                for param_group in optimizer.param_groups:
                    param_group["lr"] = lr
            else:
                lr = args.learning_rate

            input_ids = batch['input_ids'].to(device)
            labels = batch['labels'].to(device)
            logits = model(input_ids).logits
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1))

            loss.backward()

            if (idx + 1) % args.gradient_accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
            if (idx + 1) % (args.gradient_accumulation_steps * 10) == 0:
                logger.info(f"CS336-Assn5: Epoch {epoch}, Iteration {idx}, train_loss: {loss.item()}, LR: {lr}")
                wandb.log({"train_loss": loss.item(), "lr": lr, "epoch": epoch, "iteration": idx})
            if((idx + 1) % args.eval_steps == 0):
                test_loss = estimate_test_loss(model, test_dataloader, device)
                logger.info(f"CS336-Assn5: Epoch {epoch}, Iteration {idx}, test_loss: {test_loss}")
                wandb.log({"test_loss": test_loss.item(), "lr": lr, "epoch": epoch, "iteration": idx})

    model.save_pretrained(save_directory = args.output_dir)
    tokenizer.save_pretrained(save_directory = args.output_dir)
            
def estimate_test_loss(model, test_dataloader, device):
    model.eval()
    total_loss = 0
    total_samples = 0
    with torch.no_grad():
        # Wrap the test_dataloader with tqdm for a progress bar
        for batch in tqdm(test_dataloader, desc="Evaluating Test Set"):
            input_ids = batch['input_ids'].to(device)
            labels = batch['labels'].to(device)
            logits = model(input_ids).logits
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1))
            total_loss += loss.item() * input_ids.size(0)  # Adjust for batch size in the calculation
            total_samples += input_ids.size(0)
    model.train()
    return total_loss / total_samples


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_dataset_path", type=str, default='/home/shared/safety_augmented_ultrachat_200k_single_turn/train.jsonl.gz')
    parser.add_argument("--test_dataset_path", type=str, default='/home/shared/safety_augmented_ultrachat_200k_single_turn/test.jsonl.gz')
    parser.add_argument("--model_name", type=str, default='base')
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--output_dir", type=str, default='./finetuning_output')
    parser.add_argument("--seq_length", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--shuffle", type=bool, default=True)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--train_steps", type=int, default=6726)
    parser.add_argument("--warmup_ratio", type=float, default=0.03)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=16)
    parser.add_argument("--eval_steps", type=int, default=1600)
    parser.add_argument("--lr_scheduler", type=str, default='regular')
    return parser.parse_args()


if __name__ == "__main__":
    #import pdb; pdb.set_trace()
    logging.basicConfig(
        format="%(asctime)s - %(module)s - %(levelname)s - %(message)s",
        level=logging.INFO,
    )
    wandb.init(project="cs336-assn5-SFT")
    arguments = get_args()
    train_finetuning(arguments)
    # model_path = '/data/Meta-Llama-3-8B'
    #tokenizer = AutoTokenizer.from_pretrained(model_path)
    # tokenizer = AutoTokenizer.from_pretrained('./tests/fixtures/Meta-Llama-3-8B')
    #train_dataset_path = '/home/shared/safety_augmented_ultrachat_200k_single_turn/train.jsonl.gz'
    # train_dataset_path = './tests/fixtures/sft_sample.jsonl'
    #test_dataset_path = '/home/shared/safety_augmented_ultrachat_200k_single_turn/test.jsonl.gz'
    
    # train_dataset = finetuning_dataset(tokenizer, train_dataset_path, seq_length=512, shuffle = True)
    #train_dataloader = get_dataloader(train_dataset, batch_size = 2, shuffle = True)
    #test_dataset = finetuning_dataset(tokenizer, test_dataset_path, seq_length=512, shuffle = True)
    #test_dataloader = get_dataloader(test_dataset, batch_size = 2, shuffle = True)

    logger.info("finished running")
    wandb.finish()


            
