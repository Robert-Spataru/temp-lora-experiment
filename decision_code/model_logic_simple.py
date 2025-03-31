import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer, GenerationConfig
from peft import LoraConfig, get_peft_model, get_peft_model_state_dict

# Check if GPU is available and set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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

# Tokenize input for generation
#input_text = "Write a python program "
#input_ids = tokenizer.encode(input_text, return_tensors='pt').to(device)
file_path = "data/pg19/349.txt"
with open(file_path, "r", encoding="utf-8") as f:
    text = "".join(f.readlines()).strip()
    input_ids = torch.tensor(data=[tokenizer.encode(text)], dtype=torch.int64).to(device)


# Function to fine-tune LoRA on a chunk
def train_lora_on_chunk(lora_model, input_ids, epochs=1):
    lora_model.train()
    optimizer = torch.optim.AdamW(lora_model.parameters(), lr=1e-4)
    loss_fn = torch.nn.CrossEntropyLoss()

    for epoch in range(epochs):
        optimizer.zero_grad()
        outputs = lora_model(input_ids, labels=input_ids)
        loss = outputs.loss
        loss.backward()
        optimizer.step()


# Parameters
chunk_size = 128  # Number of tokens per chunk
max_length = 500  # Total desired length
generated_ids = input_ids.clone()
temp_lora_boundary_list = [1.5, 2.0, 2.5, 3.0]

for boundary in temp_lora_boundary_list:
    # Generate text in chunks
    while generated_ids.shape[-1] < max_length:
        # Generate next chunk
        with torch.no_grad():
            outputs = lora_model.generate(
                generated_ids,
                max_new_tokens=chunk_size,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
                use_cache=True  # Enable KV cache
            )
        new_tokens = outputs[:, generated_ids.shape[-1]:]

        # Append new tokens to generated_ids
        generated_ids = torch.cat([generated_ids, new_tokens], dim=-1)

        # Fine-tune LoRA on the new chunk
        train_lora_on_chunk(lora_model, generated_ids[:, -chunk_size:], epochs=1)

        # Save LoRA state if needed (optional)
        # state_dict = {k: v.cpu() for k, v in get_peft_model_state_dict(lora_model).items()}
        # temp_lora_states.append(state_dict)

# Final output
generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
print(generated_text)

# Cleanup
del lora_model