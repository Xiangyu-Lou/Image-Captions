# Image Caption Generation

A deep learning project that generates natural language descriptions for images.

## Model Architecture

```
                         Generated Caption
                                ▲
                                │
                     Language Modeling Head (Trainable)
                                ▲
                                │
                    ┌─────────────────────────────┐
                    │     MLP (Trainable)         │
                    │           ▲                 │
                    │           │                 │
                    │   LayerNorm (Trainable)     │
                    │           ▲                 │
                    │           │                 │
                    │  Cross-Attention ◄──────────┤
                    │    (Trainable)              │
                    │           ▲                 │
                    │           │                 │
                    │   LayerNorm (Trainable)     │
                    │           ▲                 │
                    │           │                 │
                    │  Self-Attention (Frozen)   │
                    │           ▲                 │
                    │           │                 │
                    │   LayerNorm (Frozen)        │
                    │    GPT-2 Transformer Layer  │
                    │         (× 24 layers)       │
                    └─────────────────────────────┘
                               ▲
                               │
                        GPT-2 Embedding ──────────┐
                        (Word + Position)         │
                               ▲                  │
                               │                  │
Input Text ────────────────────┘                 │
                                                 │
                          Visual Features        │
                           [B, 257, 1024] ───────┘
                                ▲
                                │
                            CLIP ViT-Large
                            (Vision Encoder)
                                ▲
                                │
Input Image ───────────────────────────────────────
```

The model combines two pre-trained components:

**CLIP ViT-Large-Patch14**
- Encodes images into 1024-dimensional feature vectors
- All parameters frozen during training

**GPT-2 Medium** 
- Generates text based on visual features
- Original self-attention layers frozen

**Connection Method**
- Custom cross-attention blocks injected into each GPT-2 transformer layer
- Query: from GPT-2 hidden states
- Key/Value: from CLIP visual features

**Parameters**:
- Total: 810.33 Million
- Trainable: 353.82 Million (44%)
- Frozen: CLIP encoder + GPT-2 self-attention
- Trained: Cross-attention layers + Layer norms + MLP layers

## Installation & Setup

Requirements: Python 3.10, PyTorch, CUDA(Optional, but recommended)

```bash
pip install torch torchvision torchaudio transformers datasets pillow tqdm numpy accelerate
```

**Dataset Setup** (Required for training):
```bash
# Download COCO 2017 images
mkdir -p data/train2017 data/val2017

# Training images (18GB)
wget http://images.cocodataset.org/zips/train2017.zip
unzip train2017.zip -d data/ && rm train2017.zip

# Validation images (1GB)
wget http://images.cocodataset.org/zips/val2017.zip  
unzip val2017.zip -d data/ && rm val2017.zip

## Usage

**Generate captions:**
```bash
# Single image
python inference.py -f path/to/image.jpg

# Multiple images
python inference.py -f image1.jpg image2.jpg image3.jpg

# Custom sampling parameters
python inference.py -f image.jpg --top_p 0.8 --temperature 1.5
```

**Training:**
Need at least 32GB GPU memory.
```bash
python train.py
```

**Programmatic usage:**
```python
from model import ImageCaptions
from inference import generate_caption

model = ImageCaptions()
model.load_state_dict(torch.load('models/best_model.pth'))
caption = generate_caption(model, 'image.jpg')
```

## Examples

<!-- Add your example images and captions here -->
*Example images and generated captions will be displayed here*

## Training Results

<!-- Add your training loss curves and metrics here -->
*Training loss curves and evaluation metrics will be displayed here*

## Technical Details

- **Training Strategy**: Parameter-efficient fine-tuning with frozen backbone
- **Generation**: Nucleus sampling (top-p) with temperature control
- **Training Time**: ~0.5 hours/epoch on 5090
- **Memory**: ~32GB GPU memory required
- **Dataset**: COCO 2017 (118K training images, 5K validation images)

## File Structure

```
Image-Captions/
├── model.py          # Core model implementation
├── train.py          # Training script  
├── inference.py      # Inference script
├── data.py           # Data loading
├── models/           # Saved checkpoints
└── data/             # COCO dataset
```