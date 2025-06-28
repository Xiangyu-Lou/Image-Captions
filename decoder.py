import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm

token_to_id = {
    'what': 0,
    'is': 1,
    'statquest': 2,
    'awesome': 3,
    '<EOS>': 4
}
id_to_token = dict((v, k) for k, v in token_to_id.items())

inputs = torch.tensor([[0, 1, 2, 4, 3], [2, 1, 0, 4, 3]])
labels = torch.tensor([[1, 2, 4, 3, 4], [1, 0, 4, 3, 4]])

dataset = TensorDataset(inputs, labels)
data_loader = DataLoader(dataset, batch_size=2)

class PositionEncoding(nn.Module):
    pe: torch.Tensor
    def __init__(self, d_model=2, max_len=6):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(start=0, end=max_len, step=1).float().unsqueeze(1)
        embedding_index = torch.arange(start=0, end=d_model, step=2).float()
        div_term = 1/torch.tensor(10000.0)**(embedding_index / d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)
        
    def forward(self, word_embeddings):
        seq_len = word_embeddings.size(1)
        return word_embeddings + self.pe[:seq_len, :]
    
class Attention(nn.Module):
    def __init__(self, d_model=2):
        super().__init__()
        self.d_model = d_model
        
        self.W_qkv = nn.Linear(d_model, 3*d_model, bias=False)
        
        self.row_dim = 0
        self.col_dim = 1
        
    def forward(self, encodings_for_qkv, mask=None):
        qkv = self.W_qkv(encodings_for_qkv)
        q, k, v = qkv.chunk(3, dim=-1)
        
        sims = torch.matmul(q, k.transpose(-2, -1))
        scaled_sims = sims / torch.tensor(k.size(-1) ** 0.5)
        
        if mask is not None:
            scaled_sims = scaled_sims.masked_fill(mask, -1e9)
            
        attention_percents = nn.functional.softmax(scaled_sims, dim=self.col_dim)
        output = torch.matmul(attention_percents, v)
        return output
    
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model=4, num_heads=2):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = int(d_model / num_heads)
        
        self.W_qkv = nn.Linear(d_model, 3 * d_model)
        self.W_o = nn.Linear(d_model, d_model, bias=False)
        
    def forward(self, encodings_for_qkv, mask=None):
        batch_size, seq_length, _ = encodings_for_qkv.shape
        qkv = self.W_qkv(encodings_for_qkv)
        q, k, v = qkv.chunk(3, dim=-1)
        
        q = q.view(batch_size, seq_length, self.num_heads, self.d_k).transpose(1, 2)
        k = k.view(batch_size, seq_length, self.num_heads, self.d_k).transpose(1, 2)
        v = v.view(batch_size, seq_length, self.num_heads, self.d_k).transpose(1, 2)
        
        sims = torch.matmul(q, k.transpose(-2, -1))
        scaled_sims = sims / torch.tensor(k.size(-1) ** 0.5)
        
        if mask is not None:
            scaled_sims = scaled_sims.masked_fill(mask, -1e9)
        
        attention_percents = nn.functional.softmax(scaled_sims, dim=-1)
        output = torch.matmul(attention_percents, v)
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_length, self.d_model)
        
        return output
        
        

    
class DecoderonlyTransformer(nn.Module):
    def __init__(self, num_tokens=5, d_model=2, max_len=6):
        super().__init__()

        self.we = nn.Embedding(num_embeddings=num_tokens, embedding_dim=d_model)
        self.pe = PositionEncoding(d_model=d_model, max_len=max_len)
        self.self_attention = MultiHeadAttention(d_model=d_model)
        self.fc = nn.Linear(in_features=d_model, out_features=num_tokens)

    def forward(self, token_to_ids):
        word_embedding = self.we(token_to_ids)
        position_encoded = self.pe(word_embedding)

        seq_len = position_encoded.size(1)
        
        mask = torch.tril(torch.ones(seq_len, seq_len, device=token_to_ids.device))
        mask = (mask == 0)
        
        self_attention_values = self.self_attention(position_encoded, mask=mask)
        
        residual_connection_values = position_encoded + self_attention_values
        
        fc_layer_output = self.fc(residual_connection_values)
        
        return fc_layer_output
    
def train(model, data_loader, criterion, optimizer, num_epochs, device):
    model.train()
    model.to(device)
    
    for epoch in range(num_epochs):
        total_loss = 0.0
        for input_tokens, labels in tqdm(data_loader, desc=f'Epoch: {epoch+1}'):
            input_tokens = input_tokens.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            output = model(input_tokens)
            output = output.permute(0, 2, 1)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
        avg_loss = total_loss / len(data_loader)
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}")
    
    print('----Training Finished----')
    
def generate_sequence(model, initial_input_ids, max_length, id_to_token, token_to_id, device):
    model.eval()
    model = model.to(device)
    
    model_input = initial_input_ids.unsqueeze(0).to(device)
    
    generated_ids = initial_input_ids.tolist()

    with torch.no_grad():
        for i in range(initial_input_ids.size(0), max_length):
            predictions = model(model_input)
            
            last_token_logits = predictions[:, -1, :]
            
            predicted_id = torch.argmax(last_token_logits, dim=-1)

            if predicted_id.item() == token_to_id['<EOS>']:
                generated_ids.append(predicted_id.item()) 
                break
            
            generated_ids.append(predicted_id.item())
            model_input = torch.cat((model_input, predicted_id.unsqueeze(0)) , dim=1)
            
    generated_tokens_str = [id_to_token[id] for id in generated_ids]
    
    return generated_tokens_str
    
if __name__ == '__main__':
    model = DecoderonlyTransformer(d_model=4)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
    device = torch.device('cuda:0')
    train(model, data_loader, criterion, optimizer, 10, device)
    
    print("----Start Generating----")

    initial_input_tokens_1 = torch.tensor([token_to_id["what"], token_to_id["is"], token_to_id["statquest"], token_to_id["<EOS>"]])
    initial_input_tokens_2 = torch.tensor([token_to_id["statquest"], token_to_id["is"], token_to_id["what"], token_to_id["<EOS>"]])
    generated_sequence_1 = generate_sequence(model, initial_input_tokens_2, max_length=6, 
                                             id_to_token=id_to_token, token_to_id=token_to_id, device=device)

    print(f"Input: {' '.join([str(id_to_token.item()) for id_to_token in initial_input_tokens_1])}")
    print(f"Predicted Sequence: {' '.join(generated_sequence_1)}")