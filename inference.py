import torch
from PIL import Image
import os
import argparse
from model import ImageCaptions

def generate_caption(model, image_path, max_length=50, top_p=0.9, temperature=1.0):
    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()
    
    if not os.path.exists(image_path):
        print(f"Error: Image file '{image_path}' not found.")
        return None
    
    image = Image.open(image_path).convert("RGB")
    image_pixel = model.pre_processor(images=image, return_tensors="pt").pixel_values.to(device)
    generated_ids = [model.tokenizer.eos_token_id]

    with torch.no_grad():
        for _ in range(max_length):
            input_ids = torch.tensor([generated_ids]).to(device)
            
            logits = model(image_pixel=image_pixel, decoder_input_ids=input_ids)
            next_token_logits = logits[0, -1, :]

            next_token_logits = next_token_logits / temperature
            probs = torch.nn.functional.softmax(next_token_logits, dim=-1)

            sorted_probs, sorted_indices = torch.sort(probs, descending=True)
            cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
            
            # Mask
            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = False

            indices_to_remove = sorted_indices[sorted_indices_to_remove]
            probs[indices_to_remove] = 0
            
            next_token_id = torch.multinomial(probs, num_samples=1).item()
            generated_ids.append(next_token_id)

            if next_token_id == model.tokenizer.eos_token_id:
                break
    
    caption = model.tokenizer.decode(generated_ids, skip_special_tokens=True)
    return caption


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--files', nargs='+', required=True,
                        help='Path(s) to image file(s) for caption generation')
    parser.add_argument('--top_p', type=float, default=0.9,
                        help='Top-p value for nucleus sampling (default: 0.9)')
    parser.add_argument('--temperature', type=float, default=1.2,
                        help='Temperature for sampling (default: 1.2)')
    args = parser.parse_args()
    
    model = ImageCaptions()
    model.load_state_dict(torch.load('model/epoch10.pth'))
    
    for i, image_path in enumerate(args.files, 1):
        print("=" * 50)
        print(f"\n[{i}/{len(args.files)}] Processing: {image_path}")
        
        caption = generate_caption(
            model=model,
            image_path=image_path,
            top_p=args.top_p,
            temperature=args.temperature
        )
        print(f"Caption: {caption}")
        
if __name__ == '__main__':
    model = ImageCaptions()
    model.load_state_dict(torch.load('models/epoch11.pth'))
    images_list = [
        'data/train2017/000000581880.jpg',
        'data/train2017/000000000078.jpg',
        'data/train2017/000000581921.jpg'
    ]
    for i, image in enumerate(images_list):
        generated_text = generate_caption(
            model, 
            image,
            max_length=50,
            top_p=0.9,
            temperature=1.2
        )
        print(f'\nImage: {images_list[i]}')
        print(f'\nCaption: {generated_text}')
    # main()