import torch
from PIL import Image
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
from tqdm import tqdm
from model import ImageCaptions
from data import loading_dataset
from torch.utils.data import DataLoader
import torch.nn.functional as F

def calculate_bleu(model, dataset, batch_size=32, top_p=0.8, temperature=0.8, num_batches=None):
    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    print(f'\n----Calculating BLEU-4 Score on {device}----')
    model.to(device)
    model.eval()

    def collate_for_bleu(batch):
        raw_images, references_batch = [], []
        for sample in batch:
            try:
                img = Image.open(sample['file_name']).convert('RGB')
                raw_images.append(img)
                references_batch.append([ref.split() for ref in sample['captions']])
            except (FileNotFoundError, TypeError): continue
        if not raw_images: return None, None
        return raw_images, references_batch

    val_dataloader = DataLoader(dataset['validation'], batch_size=batch_size, shuffle=False, collate_fn=collate_for_bleu)

    hypotheses, references = [], []

    with torch.no_grad():
        for batch_idx, (batch_images, batch_refs) in enumerate(tqdm(val_dataloader, desc="Generating Captions for BLEU")):
            if num_batches is not None and batch_idx >= num_batches:
                print(f"\nStopping after {num_batches} batches as requested for debugging.")
                break

            if batch_images is None: continue

            image_pixels_batch = model.pre_processor(images=batch_images, return_tensors="pt").pixel_values.to(device)

            for i in range(image_pixels_batch.size(0)):
                image_pixel_single = image_pixels_batch[i].unsqueeze(0)
                
                generated_ids = [model.tokenizer.bos_token_id]
                for _ in range(50):
                    input_ids = torch.tensor([generated_ids]).to(device)
                    
                    logits = model(image_pixel=image_pixel_single, decoder_input_ids=input_ids)
                    
                    next_token_logits = logits[0, -1, :]
                    next_token_logits = next_token_logits / temperature
                    
                    probs = torch.nn.functional.softmax(next_token_logits, dim=-1)
                    sorted_probs, sorted_indices = torch.sort(probs, descending=True)
                    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0
                    indices_to_remove = sorted_indices[sorted_indices_to_remove]
                    probs[indices_to_remove] = 0
                    next_token_id = torch.multinomial(probs, num_samples=1).item()
                    
                    generated_ids.append(next_token_id)
                    if next_token_id == model.tokenizer.eos_token_id: break
                
                hypothesis = model.tokenizer.decode(generated_ids, skip_special_tokens=True)
                hypotheses.append(hypothesis.split())
            
            references.extend(batch_refs)

    print("\n--- First Sample Generated Caption for Debugging ---")
    if hypotheses:
        generated_sentence = " ".join(hypotheses[0])
        reference_sentences = [" ".join(ref) for ref in references[0]]
        print(f"Sample 1:\n  -> Generated:  '{generated_sentence}'\n  -> Reference 1: '{reference_sentences[0]}'\n" + "-"*20)
        
    if not hypotheses:
        print("Could not generate any captions. BLEU calculation aborted.")
        return 0.0

    chencherry = SmoothingFunction()
    bleu4_score = corpus_bleu(references, hypotheses, smoothing_function=chencherry.method1)
    
    print(f"\nCorpus BLEU-4 Score (on {len(hypotheses)} samples): {bleu4_score:.4f}")
    return bleu4_score

def calculate_bleu_reduce_repeat(model, dataset, batch_size=32, top_p=0.8, temperature=0.8, repetition_penalty=1.2, num_batches=None):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f'\n----Calculating BLEU-4 Score on {device}----')
    model.to(device)
    model.eval()

    def collate_for_bleu(batch):
        raw_images, references_batch = [], []
        for sample in batch:
            try:
                img = Image.open(sample['file_name']).convert('RGB')
                raw_images.append(img)
                references_batch.append([ref.split() for ref in sample['captions']])
            except (FileNotFoundError, TypeError): continue
        if not raw_images: return None, None
        return raw_images, references_batch

    val_dataloader = DataLoader(dataset['validation'], batch_size=batch_size, shuffle=False, collate_fn=collate_for_bleu)

    hypotheses, references = [], []

    with torch.no_grad():
        for batch_idx, (batch_images, batch_refs) in enumerate(tqdm(val_dataloader, desc="Generating Captions for BLEU")):
            if num_batches is not None and batch_idx >= num_batches:
                print(f"\nStopping after {num_batches} batches as requested for debugging.")
                break

            if batch_images is None: continue

            image_pixels_batch = model.pre_processor(images=batch_images, return_tensors="pt").pixel_values.to(device)

            for i in range(image_pixels_batch.size(0)):
                image_pixel_single = image_pixels_batch[i].unsqueeze(0)
                
                generated_ids = [model.tokenizer.bos_token_id]
                for _ in range(50):
                    input_ids = torch.tensor([generated_ids]).to(device)
                    
                    logits = model(image_pixel=image_pixel_single, decoder_input_ids=input_ids)
                    
                    next_token_logits = logits[0, -1, :]
                    
                    if repetition_penalty != 1.0:
                        for token_id in set(generated_ids):
                            next_token_logits[token_id] /= repetition_penalty

                    next_token_logits = next_token_logits / temperature
                    
                    probs = F.softmax(next_token_logits, dim=-1)
                    sorted_probs, sorted_indices = torch.sort(probs, descending=True)
                    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0
                    indices_to_remove = sorted_indices[sorted_indices_to_remove]
                    probs[indices_to_remove] = 0
                    next_token_id = torch.multinomial(probs, num_samples=1).item()
                    
                    generated_ids.append(next_token_id)
                    if next_token_id == model.tokenizer.eos_token_id: break
                
                hypothesis = model.tokenizer.decode(generated_ids, skip_special_tokens=True)
                hypotheses.append(hypothesis.split())
            
            references.extend(batch_refs)

    print("\n--- First Sample Generated Caption for Debugging ---")
    if hypotheses:
        generated_sentence = " ".join(hypotheses[0])
        reference_sentences = [" ".join(ref) for ref in references[0]]
        print(f"Sample 1:\n  -> Generated:  '{generated_sentence}'\n  -> Reference 1: '{reference_sentences[0]}'\n" + "-"*20)
        
    if not hypotheses:
        print("Could not generate any captions. BLEU calculation aborted.")
        return 0.0

    chencherry = SmoothingFunction()
    bleu4_score = corpus_bleu(references, hypotheses, smoothing_function=chencherry.method1)
    
    print(f"\nCorpus BLEU-4 Score c(on {len(hypotheses)} samples): {bleu4_score:.4f}")
    return bleu4_score

if __name__ == '__main__':
    coco_dataset = loading_dataset()
    model = ImageCaptions()
    model.load_state_dict(torch.load('models/42.pth'))
    calculate_bleu(model, coco_dataset, num_batches=None)