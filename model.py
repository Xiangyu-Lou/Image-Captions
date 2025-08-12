import torch
import torch.nn as nn
import transformers
from transformers.models.clip import CLIPProcessor, CLIPModel
from transformers.models.gpt2 import GPT2Model, GPT2Tokenizer
from datasets import load_dataset
from PIL import Image

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
    def __init__(self, d_model=1024, n_head=16):
        super().__init__()

        self.d_model = d_model
        self.n_heads = n_head
        self.d_k = d_model // n_head
        
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

    def forward(self, query, key, value, attention_mask=None):
        batch_size = query.size(0)
        seq_len_q = query.size(1)
        seq_len_k = key.size(1)
        q = self.W_q(query)
        k = self.W_k(key)
        v = self.W_v(value)

        q = q.view(batch_size, seq_len_q, self.n_heads, self.d_k).transpose(1, 2)
        k = k.view(batch_size, seq_len_k, self.n_heads, self.d_k).transpose(1, 2)
        v = v.view(batch_size, seq_len_k, self.n_heads, self.d_k).transpose(1, 2)

        sims = torch.matmul(q, k.transpose(-1, -2))
        scaled_sims = sims / (self.d_k ** 0.5)

        if attention_mask is not None:
            scaled_sims = scaled_sims.masked_fill(attention_mask, -1e9)

        attention_score = nn.functional.softmax(scaled_sims, dim=-1)
        output = torch.matmul(attention_score, v)
        
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len_q, self.d_model)
        output = self.W_o(output)
        
        return output

class AttentionBlock(nn.Module):
    def __init__(self, gpt2_block, d_model, n_heads):
        super().__init__()
        
        self.ln_1 = gpt2_block.ln_1
        self.attn = gpt2_block.attn
        self.ln_2 = gpt2_block.ln_2
        self.cross_attention = CrossAttention(d_model, n_heads)
        self.ln_3 = nn.LayerNorm(d_model)
        self.mlp = gpt2_block.mlp
        
    def forward(self, x, encoder_hidden_states, decoder_attention_mask=None):
        extended_attention_mask = None
        if decoder_attention_mask is not None:
            extended_attention_mask = decoder_attention_mask.unsqueeze(1).unsqueeze(2)
            extended_attention_mask = extended_attention_mask.to(dtype=x.dtype)
            extended_attention_mask = (1.0 - extended_attention_mask) * -1e9
        
        ln_1_output = self.ln_1(x)
        attn_output, _ = self.attn(ln_1_output, layer_past=None, attention_mask=extended_attention_mask, head_mask=None, use_cache=False)
        residual_1 = attn_output[0] + x
        
        ln_2_output = self.ln_2(residual_1)
        cross_attn_output = self.cross_attention(
            query=ln_2_output,
            key=encoder_hidden_states,
            value=encoder_hidden_states,
            attention_mask=None
        )
        residual_2 = residual_1 + cross_attn_output
        ln_3_output = self.ln_3(residual_2)
        mlp_output = self.mlp(ln_3_output)
        output = mlp_output + residual_2
        
        return output
        
class ImageCaptions(nn.Module):
    def __init__(self, d_model=1024, n_head=16):
        super().__init__()
        
        self.pre_encoder, self.pre_processor = load_encoder_model()
        self.pre_decoder, self.tokenizer = load_decoder_model()
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.pre_decoder.config.pad_token_id = self.tokenizer.eos_token_id
        
        self.d_model = d_model
        self.n_head = n_head

        self.decoder = self._inject_cross_attention()
        
        self.fc_connect = nn.Linear(d_model, d_model)
        
        self.fc_head = nn.Linear(in_features=d_model, out_features=self.tokenizer.vocab_size)
        
        self._freeze_parameters()
        
    def _inject_cross_attention(self):
        blocks = nn.ModuleList()
        for block in self.pre_decoder.h:
            blocks.append(AttentionBlock(block, self.d_model, self.n_head))
        
        self.pre_decoder.h = blocks
        return self.pre_decoder
    
    def _freeze_parameters(self):
        for param in self.parameters():
            param.requires_grad = False
            
        for param in self.fc_connect.parameters():
            param.requires_grad = True

        for block in self.decoder.h:
            for param in block.ln_2.parameters():   # type: ignore
                param.requires_grad = True
            for param in block.cross_attention.parameters():    # type: ignore
                param.requires_grad = True
            for param in block.ln_3.parameters():   # type: ignore
                param.requires_grad = True
            for param in block.mlp.parameters():    # type: ignore
                param.requires_grad =True

        for param in self.fc_head.parameters():
            param.requires_grad = True

    def forward(self, image_pixel, decoder_input_ids, decoder_mask=None):

        encoder_output = self.pre_encoder(image_pixel).last_hidden_state
        encoder_output = self.fc_connect(encoder_output)
        
        input_embeds = self.pre_decoder.wte(decoder_input_ids)
        position_ids = torch.arange(0, decoder_input_ids.size(1), dtype=torch.long, device=decoder_input_ids.device)
        position_embed = self.pre_decoder.wpe(position_ids)
        hidden_state = input_embeds + position_embed
        
        for block in self.pre_decoder.h:
            hidden_state = block(
                hidden_state, 
                encoder_output, 
                decoder_attention_mask=decoder_mask
            )
        
        logits = self.fc_head(hidden_state)

        return logits
    
if __name__ =='__main__':

    model = ImageCaptions()
    total_params = sum(param.numel() for param in model.parameters())
    trainable_params = sum(param.numel() for param in model.parameters() if param.requires_grad)
    
    print("="*50)
    print(f"Model total parameters: {total_params / 1_000_000:.2f} Million")
    print(f"Trainable parameters: {trainable_params / 1_000_000:.2f} Million")
    print("="*50)