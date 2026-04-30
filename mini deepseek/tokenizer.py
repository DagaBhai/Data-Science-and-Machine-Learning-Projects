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
    texts=[]
    for inp, out in zip(examples["input"], examples["output"]):
        prompt = f"""### Input:
            {inp}

            ### Response:
            {out}
        """
        
        texts.append(prompt)

    tokenized =tokenizer(
        texts,
        truncation=True,
        max_length=512,      
        padding=True      
    )

    tokenized["labels"] = tokenized["input_ids"].copy()

    return tokenized

import json
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

optimizer = torch.optim.AdamW(model.parameters(),lr=5e-6)

model.train()
device = model.device

for epoch in range(10):
    for batch in train_dataset:
        #moving to device
        batch = {k: v.to(device) for k, v in batch.items()}

        #model predits next token probablities
        output = model(input_ids=batch['input_ids'],attention_mask=batch['attention_mask'])
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
        labels = batch["input_ids"][:, 1:] 

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
        loss = F.cross_entropy(logits, labels,ignore_index=tokenizer.pad_token_id)

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
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
        repetition_penalty=1.2
        )
    return tokenizer.decode(output_tokens[0], skip_special_tokens=True)

print(test("What is 15 * 7?"))