import os
import gc
import torch
import json
from tqdm import tqdm
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from peft import LoraConfig, get_peft_model, get_peft_model_state_dict
from torch.utils.data import DataLoader, SequentialSampler
from accelerate import Accelerator

# Add parent directory to path to make imports work
import sys
sys.path.append('/home/robert/temp-lora-code/temp-lora-decision')

# Now we can import the modules
from data_modules.pg19 import PG19SlowRawDataset, PG19Dataset
from trainer.acc_pg19_trainer import train_once, inference_once, evaluate
from helper import create_lr_scheduler, write_json

# Check if GPU is available and set device
accelerator = Accelerator()
accelerator.wait_for_everyone()
device = accelerator.device
#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# Load GPT-2 and tokenizer
model = GPT2LMHeadModel.from_pretrained('gpt2').to(device)
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

# Configure LoRA with peft
lora_config = LoraConfig(
    r=8,  # Rank of LoRA
    lora_alpha=32,
    target_modules=['c_attn', 'c_proj'],
    lora_dropout=0.1,
    task_type="CAUSAL_LM"  # Specify the task type
)

# Create the LoRA model
lora_model = get_peft_model(model, lora_config).to(device)
temp_lora_states = []  # List to store LoRA states

file_path = "data/pg19/349.txt"
prefix_length_training = 128
prefix_length_eval = 128
stride_size = 512
num_train_epochs = 1

# Check if the data file exists
if not os.path.exists(file_path):
    print(f"Error: File {file_path} does not exist.")
    exit(1)

try:
    with open(file_path, "r", encoding="utf-8") as f:
        text = "".join(f.readlines()).strip()
        input_ids = torch.tensor(data=[tokenizer.encode(text)], dtype=torch.int64)
except Exception as e:
    print(f"Error loading file: {e}")
    exit(1)

# Create train dataset
_dataset = PG19SlowRawDataset(
    fp=file_path, tokenizer=tokenizer, prefix_length=prefix_length_training,
    stride_size=stride_size
)
_dataset.load_from_input_ids(input_ids=input_ids)
train_dataset = PG19Dataset(dataset=_dataset)
train_dataloader = DataLoader(
    dataset=train_dataset, batch_size=1, shuffle=False, sampler=SequentialSampler(data_source=train_dataset),
    num_workers=0, collate_fn=lambda i: i[0], pin_memory=True
)
num_training_steps = len(train_dataloader)

# Create eval dataset
_dataset = PG19SlowRawDataset(
    fp=file_path, tokenizer=tokenizer, prefix_length=prefix_length_eval,
    stride_size=stride_size
)
_dataset.load_from_input_ids(input_ids=input_ids)
eval_dataset = PG19Dataset(dataset=_dataset)
eval_dataloader = DataLoader(
    dataset=eval_dataset, batch_size=1, shuffle=False, sampler=SequentialSampler(data_source=eval_dataset),
    num_workers=0, collate_fn=lambda i: i[0], pin_memory=True
)

# Setup optimizer and scheduler
optimizer = torch.optim.AdamW(lora_model.parameters(), lr=1e-4)
loss_fn = torch.nn.CrossEntropyLoss()
lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer= optimizer, lr_lambda=lambda step: 1.0 - (step / num_training_steps))

# Initialize tracking variables
pbr = tqdm(range(0, num_training_steps), desc="Training", unit="batch")
total_tokens, base_ppl, lora_ppl = 0, 0, 0
step_record = []

eval_last_index = -1
eval_total_tokens = 0
eval_base_ppl = 0
eval_lora_ppl = 0
lora_decision_ppl_boundary_list = [1.5, 2.0, 2.5, 3.0]

output_dir = "output_dir"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)


accelerator.state.deepspeed_plugin.deepspeed_config["train_micro_batch_size_per_gpu"] = 1
lora_model, optimizer, train_dataloader, lr_scheduler = accelerator.prepeare(lora_model, optimizer, train_dataloader, lr_scheduler)

# Main training loop
for i in range(0, len(lora_decision_ppl_boundary_list)):
    lora_decision_ppl_boundary = lora_decision_ppl_boundary_list[i]
    boundary_dir = os.path.join(output_dir, str(lora_decision_ppl_boundary))
    if not os.path.exists(boundary_dir):
        os.makedirs(boundary_dir)
    
    for b_idx, batch in enumerate(train_dataloader):
        new_tokens = batch["new_tokens"]
        serial_id = batch["serial_id"]
        total_tokens += new_tokens

        step_base_ppl, step_base_loss = inference_once(model=model, batch=batch, disable_lora=True)
        step_lora_ppl, step_lora_loss = inference_once(model=lora_model, batch=batch, disable_lora=False)
        base_ppl += step_base_ppl
        lora_ppl += step_lora_ppl

        _record = {
            "b_idx": b_idx,
            "inference": {
                "step_base_ppl": step_base_ppl,
                "step_lora_ppl": step_lora_ppl,
                "step_base_loss": step_base_loss.cpu().item(),
                "step_lora_loss": step_lora_loss.cpu().item(),
            },
            "train": {},
            "lr": lr_scheduler.get_last_lr(),
            "total_tokens": total_tokens,
            "new_tokens": new_tokens,
            "serial_id": serial_id,
            "eval_step_record": {}
        }

        last_infer_input_ids = batch["input_ids"][:, -stride_size:].detach().contiguous() if stride_size > 0 else batch["input_ids"].detach().contiguous()
        eval_base_ppl, eval_lora_ppl, eval_total_tokens, eval_last_index = evaluate(
            eval_dataloader=eval_dataloader,
            model=model,
            lora_model=lora_model,
            device=device,
            total_tokens=total_tokens,
            eval_total_tokens=eval_total_tokens,
            eval_base_ppl=eval_base_ppl,
            eval_lora_ppl=eval_lora_ppl,
            eval_last_idx=eval_last_index,
            last_infer_input_ids=last_infer_input_ids
        )

        _record["eval_step_record"][str(prefix_length_eval)] = {
            "base_ppl": eval_base_ppl,
            "lora_ppl": eval_lora_ppl,
            "mean_base_ppl": eval_base_ppl / eval_total_tokens if eval_total_tokens > 0 else 0,
            "mean_lora_ppl": eval_lora_ppl / eval_total_tokens if eval_total_tokens > 0 else 0,
            "total_tokens": eval_total_tokens
        }
        step_record.append(_record)
        step_lora_loss = 0
        
        # Decide whether to train LoRA based on perplexity
        if lora_ppl / total_tokens > lora_decision_ppl_boundary:
            for epoch in range(0, num_train_epochs):
                _step_lora_ppl, step_lora_loss = train_once(model=lora_model, batch=batch, optimizer=optimizer)
                step_record[-1]["train"][f"train_lora_{epoch + 1}"] = {
                    "ppl": _step_lora_ppl,
                    "loss": step_lora_loss.detach().cpu().item(),
                }
        else:
            for epoch in range(0, num_train_epochs):
                step_record[-1]["train"][f"train_lora_{epoch + 1}"] = {
                    "ppl": None,
                    "loss": None,
                }
            
        lr_scheduler.step()
        pbr.update(n=1)
        
        # Periodic memory cleanup
        if b_idx % 500 == 0:
            gc.collect()
            torch.cuda.empty_cache()
            
        # Periodic save
        if b_idx % 10 == 0:
            output_file = os.path.join(boundary_dir, os.path.basename(file_path).replace(".txt", ".json"))
            write_json(
                fp=output_file,
                obj=[{
                    "done": False,
                    "file": os.path.basename(file_path),
                    "base_ppl": base_ppl,
                    "lora_ppl": lora_ppl,
                    "num_tokens": total_tokens,
                    "mean_base_ppl": base_ppl / total_tokens if total_tokens > 0 else 0,
                    "mean_lora_ppl": lora_ppl / total_tokens if total_tokens > 0 else 0,
                    "steps": step_record
                }]
            )
    pbr.close()
    
    # Final save for this boundary
    output_file = os.path.join(boundary_dir, os.path.basename(file_path).replace(".txt", ".json"))
    write_json(
        fp=output_file,
        obj=[{
            "done": True,
            "file": os.path.basename(file_path),
            "base_ppl": base_ppl,
            "lora_ppl": lora_ppl,
            "num_tokens": total_tokens,
            "mean_base_ppl": base_ppl / total_tokens if total_tokens > 0 else 0,
            "mean_lora_ppl": lora_ppl / total_tokens if total_tokens > 0 else 0,
            "steps": step_record
        }]
    )

print("Training complete!")