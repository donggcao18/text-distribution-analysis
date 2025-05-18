import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
# No need for 3D axes anymore
# from mpl_toolkits.mplot3d import Axes3D
import jsonlines
from transformers import AutoTokenizer, AutoModel
import torch
import os
import torch.nn as nn

# os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
# GEMINI_MODEL = ['Gemini 2.0', 'Gemini 2.0 Flash-Lite', 'Gemini 1.5 Pro', 'Gemini 1.5 Flash']
# OTHER_MODEL = ['DeepSeek V3', 'Llama-3.3-70B-Instruct-Turbo', "GPT-4o-mini"]
GEMINI_MODEL = ['Gemini 2.0','Gemini 2.0 Flash-Lite', 'Gemini 1.5 Flash']
OTHER_MODEL = ['GPT-4o-mini']
SAMPLE_SIZE = 2000
EMBEDDING_MODEL = "princeton-nlp/unsup-simcse-roberta-base"


MODEL_PATH = os.path.join("model", "model_45.pth")
if os.path.exists(MODEL_PATH):
    print(f"Loading model from {MODEL_PATH}")
    model_state = torch.load(MODEL_PATH, map_location=torch.device('cpu'))
else:
    print(f"Model checkpoint {MODEL_PATH} not found, using default model")

class TextEmbeddingModel(nn.Module):
    def __init__(self, model_name, model_state,output_hidden_states=False):
        super(TextEmbeddingModel, self).__init__()
        self.model_name = model_name
        if output_hidden_states:
            self.model = AutoModel.from_pretrained(model_name, trust_remote_code=True, output_hidden_states=True)
        else:
            if model_state != None:
                self.model = AutoModel.from_pretrained(model_name, trust_remote_code=True)
                self.model.load_state_dict(model_state, strict=False)
            else:
                self.model = AutoModel.from_pretrained(model_name, trust_remote_code=True)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

    def pooling(self, model_output, attention_mask, use_pooling='average',hidden_states=False):
        if hidden_states:
            model_output.masked_fill(~attention_mask[None,..., None].bool(), 0.0)
            if use_pooling == "average":
                emb = model_output.sum(dim=2) / attention_mask.sum(dim=1)[..., None]
            else:
                emb = model_output[:,:, 0]
            emb = emb.permute(1, 0, 2)
        else:
            model_output.masked_fill(~attention_mask[..., None].bool(), 0.0)
            if use_pooling == "average":
                emb = model_output.sum(dim=1) / attention_mask.sum(dim=1)[..., None]
            elif use_pooling == "cls":
                emb = model_output[:, 0]
        return emb
    
    def forward(self, encoded_batch, use_pooling='average',hidden_states=False):
        if "t5" in self.model_name.lower():
            input_ids = encoded_batch['input_ids']
            decoder_input_ids = torch.zeros((input_ids.shape[0], 1), dtype=torch.long, device=input_ids.device)
            model_output = self.model(**encoded_batch, 
                                  decoder_input_ids=decoder_input_ids)
        else:
            model_output = self.model(**encoded_batch)
        
        if 'bge' in self.model_name.lower() or 'mxbai' in self.model_name.lower():
            use_pooling = 'cls'
        if isinstance(model_output, tuple):
            model_output = model_output[0]
        if isinstance(model_output, dict):
            if hidden_states:
                model_output = model_output["hidden_states"]
                model_output = torch.stack(model_output, dim=0)
            else:
                model_output = model_output["last_hidden_state"]
        
        emb = self.pooling(model_output, encoded_batch['attention_mask'], use_pooling,hidden_states)
        emb = torch.nn.functional.normalize(emb, dim=-1)
        return emb

embeddings = {}

def extract_embeddings(input_file, model_name):
    texts = []
    count = 0
    num=0
    with jsonlines.open(input_file) as file:
        for line in file:
            if num<0:
                num+=1
                continue
            elif line['label_detailed'] == model_name and 'content' in line and count < SAMPLE_SIZE:
                texts.append(line['content'])
                count += 1
                num+=1
            
    
    print(f"Found {len(texts)} samples for {model_name}")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    text_embedding_model = TextEmbeddingModel(EMBEDDING_MODEL, model_state, output_hidden_states=False).to(device)
    text_embedding_model.eval()
    tokenizer = AutoTokenizer.from_pretrained(EMBEDDING_MODEL)
    
    batch_size = 8
    all_embeddings = []
    
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i+batch_size]
        try:
            inputs = tokenizer(batch_texts, return_tensors='pt', padding=True, 
                              truncation=True, max_length=512)
            
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = text_embedding_model(inputs)
                
            batch_embeddings = outputs.cpu().numpy()
            all_embeddings.append(batch_embeddings)
        except Exception as e:
            print(f"Error processing batch: {e}")
            continue
    
    if all_embeddings:
        return np.vstack(all_embeddings)
    else:
        print(f"Warning: No embeddings generated for {model_name}")
        return np.zeros((SAMPLE_SIZE, 768))  

print("Processing Gemini models...")
for model_name in GEMINI_MODEL:
    print(f"Extracting embeddings for {model_name}")
    input_file = r"text/model-specific/abstract_gemini.jsonl"
    embeddings[model_name] = extract_embeddings(input_file, model_name)

print("Processing other models...")
for model_name in OTHER_MODEL:
    print(f"Extracting embeddings for {model_name}")
    input_file = r"text/model-family/abstract_arxiv_AI_en.jsonl"  
    embeddings[model_name] = extract_embeddings(input_file, model_name)


valid_embeddings = {k: v for k, v in embeddings.items() if v.shape[0] > 1}


all_data = np.vstack(list(valid_embeddings.values()))
labels = np.concatenate([[name] * len(arr) for name, arr in valid_embeddings.items()])


print("Performing PCA...")
# Changed from 3 to 2 components
pca = PCA(n_components=2)
reduced = pca.fit_transform(all_data)
print(f"Explained variance ratio: {pca.explained_variance_ratio_}")

# Option for t-SNE instead of PCA
# print("Performing t-SNE...")
# from sklearn.manifold import TSNE
# tsne = TSNE(n_components=2, random_state=42, perplexity=30, n_iter=1000)
# reduced = tsne.fit_transform(all_data)
# print(f"t-SNE completed with perplexity: {30}")

colors = ['cyan','purple', 'deepskyblue','red','green', 'blue','orange','yellow', 'lightgrey']
markers = ['*', 'o', 'v', 's', 'x', 'D', '^', 'P']
# colors = ['cyan', 'lightgrey', 'deepskyblue', 'blue']
# markers = ['*', 'o', 'v', 's']

# Create 2D plot instead of 3D
fig = plt.figure(figsize=(12, 10))
# Remove 3D projection
ax = fig.add_subplot(111)

for i, (label_name, color, marker) in enumerate(zip(valid_embeddings.keys(), colors, markers)):
    indices = labels == label_name
    # Only use first two dimensions for 2D plot
    ax.scatter(
        reduced[indices, 0], reduced[indices, 1],
        label=label_name, c=color, marker=marker, alpha=0.7, 
        s=30  # Increased point size for better visibility in 2D
    )

ax.set_xlabel("PC 1")
ax.set_ylabel("PC 2")
# No Z-axis in 2D
ax.legend(fontsize=10)
plt.title("2D Embedding Space Visualization of AI Text Models", fontsize=14)
plt.tight_layout()
plt.grid(True, linestyle='--', alpha=0.6)  # Add grid for better readability
plt.savefig('model_embeddings_2d_visualization.png', dpi=300)
print("Visualization saved as 'model_embeddings_2d_visualization.png'")
plt.show()