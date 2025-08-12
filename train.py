from data import loading_dataset
from model import ImageCaptions
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import random
import torch.nn as nn
from PIL import Image
import json
import os
import time
# from transformers.optimization import get_cosine_schedule_with_warmup

def create_collate_fn(tokenizer, processor):
    def collate_fn(batch):
        raw_images = []
        captions = []
        
        for sample in batch:
            img = Image.open(sample['file_name'])
            if img.mode != 'RGB':
                img = img.convert('RGB')
            raw_images.append(img)
            captions.append(random.choice(sample['captions']))

        image_inputs = processor(
            images=raw_images,
            return_tensors="pt"
        )

        text_inputs = tokenizer(
            captions,
            padding='longest',
            truncation=True,
            max_length=50,
            return_tensors="pt"
        )
        return image_inputs.pixel_values, text_inputs.input_ids, text_inputs.attention_mask
    return collate_fn

def train(model, dataset, epochs=5, batch_size=16, lr=1e-4):
    device = torch.device('cuda:0')
    print(f'----Start Training----')
    model.to(device)
    model.train()
    if os.path.exists('models/log.json') and os.path.getsize('models/log.json') > 0:
        with open('models/log.json', 'r') as f:
            log = json.load(f)
    else:
        log = {
            "all_train_loss": [],
            "all_eval_loss": []
        }

    collate_fn = create_collate_fn(model.tokenizer, model.pre_processor)
    train_dataloader = DataLoader(
        dataset['train'], 
        batch_size=batch_size, 
        shuffle=True, 
        collate_fn=collate_fn
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    # total_steps   = len(train_dataloader) * epochs
    # warmup_steps  = int(0.05 * total_steps)
    # scheduler = get_cosine_schedule_with_warmup(
    #     optimizer,
    #     num_warmup_steps=warmup_steps,
    #     num_training_steps=total_steps
    # )
    
    pad_token_id = model.tokenizer.pad_token_id if model.tokenizer.pad_token_id is not None else 50256
    criterion = nn.CrossEntropyLoss(ignore_index=pad_token_id)

    for epoch in range(epochs):
        print(f"\n---- Epoch {epoch+1}/{epochs} ----")
        start_time = time.time()
        total_loss = 0
        for image_pixels, input_ids, attention_mask in tqdm(train_dataloader, desc=f"Epoch {epoch+1}"):
            image_pixels = image_pixels.to(device)
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            
            optimizer.zero_grad()
            logits = model(image_pixel=image_pixels, decoder_input_ids=input_ids, decoder_mask=attention_mask)
            pred_logits = logits[:, :-1, :].contiguous()
            target_labels = input_ids[:, 1:].contiguous()
            loss = criterion(pred_logits.view(-1, model.tokenizer.vocab_size), target_labels.view(-1))
            loss.backward()
            optimizer.step()
            # scheduler.step()
            
            total_loss += loss.item()
        
        end_time = time.time()
        avg_loss = total_loss / len(train_dataloader)
        print(f"Epoch {epoch+1} Average Loss: {avg_loss:.4f} Time: {(end_time-start_time)/60:.1f}mins")
        torch.save(model.state_dict(), f'models/epoch{epoch+1}.pth')
        print(f"Save model Successfully")
        eval_loss = evaluate(model, dataset)

        log["all_train_loss"].append(avg_loss)
        log["all_eval_loss"].append(eval_loss)
        with open('models/log.json', 'w') as f:
            json.dump(log, f)
        
    print("\n---- Training Finished ----")

def evaluate(model, dataset, batch_size=64):
    device = torch.device('cuda:0')
    print(f'----Evaluating----')
    model.to(device)
    model.eval()

    collate_fn = create_collate_fn(model.tokenizer, model.pre_processor)
    val_dataloader = DataLoader(
        dataset['validation'],
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn
    )

    pad_token_id = model.tokenizer.pad_token_id if model.tokenizer.pad_token_id is not None else 50256
    criterion = nn.CrossEntropyLoss(ignore_index=pad_token_id)
    total_loss = 0

    with torch.no_grad():
        for batch in tqdm(val_dataloader, desc="Evaluating"):
            image_pixels, input_ids, attention_mask = batch
            image_pixels = image_pixels.to(device)
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)

            logits = model(image_pixel=image_pixels, decoder_input_ids=input_ids, decoder_mask=attention_mask)
            pred_logits = logits[:, :-1, :].contiguous()
            target_labels = input_ids[:, 1:].contiguous()
            loss = criterion(pred_logits.view(-1, model.tokenizer.vocab_size), target_labels.view(-1))
            total_loss += loss.item()

    avg_loss = total_loss / len(val_dataloader)
    print(f"Validation Loss: {avg_loss:.4f}")
    return avg_loss

if __name__ =='__main__':
    dataset = loading_dataset()
    model = ImageCaptions()
    model.load_state_dict(torch.load('models/42.pth'))
    train(model, dataset, epochs=3, batch_size=48, lr=1e-7)