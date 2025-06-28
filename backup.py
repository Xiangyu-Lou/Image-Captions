import torch
import torch.nn as nn
import transformers
from transformers.models.clip import CLIPProcessor, CLIPModel
from transformers.models.gpt2 import GPT2Model, GPT2Tokenizer
from datasets import load_dataset
from PIL import Image
import random

def load_encoder_model(model_name='openai/clip-vit-large-patch14', isprint=False):
    print('----Loading vision encoder and vision processor----')
    
    vision_model = CLIPModel.from_pretrained(model_name)
    vision_encoder = vision_model.vision_model
    vision_processor = CLIPProcessor.from_pretrained(model_name)
    
    if isprint == True:
        print(vision_encoder)
    
    return vision_encoder, vision_processor
    
def load_decoder_model(model_name='gpt2-medium', isprint=False):
    print('----Loading language decoder and tokenizer----')
    
    language_decoder = GPT2Model.from_pretrained(model_name)
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        
    if isprint == True:            
        print(language_decoder)
        
    return language_decoder, tokenizer

class CrossAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        
        self.d_model = d_model
        self.n_heads = n_heads
        
        self.mha = nn.MultiheadAttention(
            embed_dim=d_model, 
            num_heads=n_heads, 
            dropout=0.1,
            batch_first=True
        )
        
    def forward(self, query, key, value):
        output, _ = self.mha(
            query=query, 
            key=key, 
            value=value
        )
        
        return output
    
class AttentionBlock(nn.Module):
    def __init__(self, gpt2_block, d_model, n_heads):
        super().__init__()
        
        self.ln_1 = gpt2_block.ln_1
        self.self_attn = gpt2_block.attn
        self.ln_2 = gpt2_block.ln_2
        
        self.cross_attn = CrossAttention(d_model, n_heads)
        self.ln_3 = nn.LayerNorm(d_model)
        
        self.fc = gpt2_block.mlp

    def forward(self, decoder_stat, encoder_stat, attn_mask=None):
        extended_attn_mask = None
        if attn_mask is not None:
            extended_attn_mask = attn_mask.unsqueeze(1).unsqueeze(2)
            extended_attn_mask = extended_attn_mask.to(dtype=decoder_stat.dtype)
            extended_attn_mask = (1 - extended_attn_mask) * -1e9
            
        outputs_ln_1 = self.ln_1(decoder_stat)
        outputs_attn, _ = self.self_attn(outputs_ln_1, attention_mask=extended_attn_mask)
        residual_1 = outputs_attn[0]+ decoder_stat
        
        outputs_ln_2 = self.ln_2(residual_1)
        outputs_crossattn = self.cross_attn(
            outputs_ln_2,
            encoder_stat,
            encoder_stat
            )
        residual_2 = outputs_crossattn + residual_1
        
        outputs_ln_3 = self.ln_3(residual_2)
        outputs_fc = self.fc(outputs_ln_3)
        outputs = outputs_fc + residual_2
        
        return outputs
        
class ImageCaptions(nn.Module):
    def __init__(self, d_model=1024, n_heads=16):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        
        self.encoder, self.pre_processor = load_encoder_model()
        self.decoder, self.tokenizer = load_decoder_model()
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            # self.decoder.config.pad_token_id = self.tokenizer.eos_token_id
        
        self.decoder = self._inject_cross_attention()
        
        self.fc_head = nn.Linear(in_features=d_model, out_features=self.tokenizer.vocab_size)
        
        self._freeze_parameters()
        
    def _freeze_parameters(self):
        
        for param in self.parameters():
            param.requires_grad = False
        
        for block in self.decoder.h:
            for param in block.ln_2.parameters():   # type: ignore
                param.requires_grad = True
            for param in block.cross_attn.parameters(): # type: ignore
                param.requires_grad = True
            for param in block.ln_3.parameters():   # type: ignore
                param.requires_grad = True
            for param in block.fc.parameters(): # type: ignore
                param.requires_grad = True
        
        for param in self.fc_head.parameters():
            param.requires_grad = True

    def _inject_cross_attention(self):
        blocks = nn.ModuleList()
        for block in self.decoder.h:
            blocks.append(AttentionBlock(block, self.d_model, self.n_heads))
        
        self.decoder.h = blocks
        return self.decoder
    
    def forward(self, image_pixel, decoder_input_ids, decoder_mask=None):
        
        outputs_encoder = self.encoder(image_pixel).last_hidden_state
        input_embeds = self.decoder.wte(decoder_input_ids)
        positions = torch.arange(0, decoder_input_ids.size(-1), dtype=torch.long, device=decoder_input_ids.device)
        position_embed = self.decoder.wpe(positions)
        hidden_stat = input_embeds + position_embed
        
        for block in self.decoder.h:
            hidden_stat = block(
                hidden_stat,
                outputs_encoder,
                decoder_mask
            )
        
        logits = self.fc_head(hidden_stat)
        
        return logits
    
def train(model, dataset, epochs=10, batch_size=32, lr=1e-5):
    device = torch.device('cuda:0')
    model.to(device)
    model.train()
    
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