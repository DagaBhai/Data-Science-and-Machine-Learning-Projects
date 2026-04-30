import json
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, DataCollatorForLanguageModeling

import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
import os
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

model_id = "openai-community/gpt2"
model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.float32, device_map="auto")
tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.pad_token = tokenizer.eos_token
model.resize_token_embeddings(len(tokenizer))

def tokenize_function(examples):
    full_texts = []
    prompt_lengths = []

    for inp, out in zip(examples["input"], examples["output"]):
        prompt = f"### Input:\n{inp}\n\n### Response:\n"
        full = prompt + out
        full_texts.append(full)

        prompt_len = len(tokenizer(prompt, add_special_tokens=False)["input_ids"])
        prompt_lengths.append(prompt_len)

    # Pad and truncate everything to the same length here
    tokenized = tokenizer(
        full_texts,
        truncation=True,
        max_length=512,
        padding="max_length",  # Uniform length — avoids the tensor shape mismatch
    )

    labels = []
    for i, ids in enumerate(tokenized["input_ids"]):
        label = list(ids)  # copy

        prompt_len = prompt_lengths[i]

        # Mask prompt tokens
        for j in range(prompt_len):
            label[j] = -100

        # Mask padding tokens (eos used as pad)
        for j in range(len(label)):
            if ids[j] == tokenizer.pad_token_id:
                label[j] = -100

        labels.append(label)

    tokenized["labels"] = labels
    return tokenized

print("Script started...") # If you don't see this, the script isn't running at all.
raw_data = load_dataset("json",data_files="examples.json")
print("Data loaded successfully!")

tokenized_dataset = raw_data.map(tokenize_function, batched=True, remove_columns=["input", "output"])

data_collector = DataCollatorForLanguageModeling(tokenizer=tokenizer,mlm=False, pad_to_multiple_of=16)

train_dataset = DataLoader(
    tokenized_dataset['train'],
    batch_size=2,
    shuffle=True,
    collate_fn=data_collector,
)

optimizer = torch.optim.AdamW(model.parameters(),lr=2e-5)

model.train()
device = model.device

for epoch in range(3):
    for batch in train_dataset:
        #moving to device
        batch = {k: v.to(device) for k, v in batch.items()}

        #model predits next token probablities
        output = model(input_ids=batch['input_ids'],attention_mask=batch['attention_mask'],labels=batch["labels"],)
        logits = output.logits

        '''
        Input sequence (Tokens):
        [A, B, C, D]

        Model outputs raw predictions (Logits) at each position:
        Logit 0: (Trying to predict what follows nothing) -> Should be B
        Logit 1: (Trying to predict what follows A)       -> Should be B 
        Logit 2: (Trying to predict what follows B)       -> Should be C
        Logit 3: (Trying to predict what follows C)       -> Should be D

        Wait! In a standard forward pass, Logit 0 actually corresponds to the prediction 
        made AFTER seeing Input A. 

        So the raw output looks like this:
        Position 0 (A) -> predicts Logit 0
        Position 1 (B) -> predicts Logit 1
        Position 2 (C) -> predicts Logit 2
        Position 3 (D) -> predicts Logit 3
        '''

        # THE SHIFT
        # We remove the last prediction because we don't have a label for what comes after D
        logits = logits[:, :-1, :] 

        # We remove the first token because it's never a "target," it's always the starting input
        labels = batch["labels"][:, 1:] 

        # Mask padding (-100 tells cross_entropy to skip those positions)
        
        #labels[labels == tokenizer.pad_token_id] = -100

        '''
        Now they are perfectly aligned for the loss function:

        Index    Input used      Logit (Prediction)    Label (Correct Answer)
        0        A               Logit 0               B
        1        B               Logit 1               C
        2        C               Logit 2               D

        Result:
        Position 0 predicts B, Label is B.
        Position 1 predicts C, Label is C.
        Position 2 predicts D, Label is D.
        '''
        
        #reshape for cross_entropy loss
        B, T, C = logits.shape
        logits = logits.reshape(B*T,C)
        labels = labels.reshape(B * T)
        
        #for each to token loss = -log(probability of correct token)
        loss = F.cross_entropy(logits, labels,ignore_index=-100)

        #Computes gradients: ∂loss / ∂weights
        loss.backward()

        #Model slightly improves its predictions
        optimizer.step()
        optimizer.zero_grad()

    print(f"Epoch {epoch} loss: {loss.item()}")

def test(text):
    model.to('cuda')
    inputs_dict = tokenizer(text, return_tensors='pt').to('cuda')
    inputs_dict = inputs_dict.to('cuda')
    output_tokens = model.generate(
        **inputs_dict,
        max_new_tokens=50,
        pad_token_id=tokenizer.pad_token_id,
        do_sample=False,
        temperature=0.7,
        top_p=0.9,
        repetition_penalty=1.2
        )
    return tokenizer.decode(output_tokens[0], skip_special_tokens=True)

print(test("### Input:\nSolve: 441 + 628\n\n### Response:\n"))