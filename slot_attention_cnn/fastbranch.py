from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import torch
import cv2
import numpy as np
import time
import torch.nn as nn
import math
from matplotlib import cm
import torch.nn.functional as F
from tqdm import tqdm
import os
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataloader import default_collate
from pathlib import Path

class COMPHY(Dataset):    
    def __init__(self, root):
        super(COMPHY, self,).__init__()
        print("Comphy constructor !",root)
        self.root_dir = root
        self.files = list(Path(root).rglob("*.mp4"))
        print("Files : ",self.files)
        self.process = process_video

    def __getitem__(self, index):
        path = self.files[index]
        frame_array = self.process(path)
        print("Path : ",path)
        print("before split ")
        name = str(path).split('\\')[2]
        print("Name : ",name)
        name = name.split('.')[0]
        return np.array(frame_array), name
    
    def __len__(self):
        return len(self.files)
    
# Convert video in path to array of frames at rate of 2 FPS
def process_video(path):
    print("Processing Video : ",path)
    KPS = 10 # Target Keyframes Per Second
    frame_array = []
    vidObj = cv2.VideoCapture(str(path))
    success = 1
    count = 0
    fps = round(vidObj.get(cv2.CAP_PROP_FPS))
    hop = round(fps / KPS)
    while(success):
        success,img = vidObj.read()
        if img is not None and count % hop == 0:
            resized = cv2.resize(img, (128, 128))
            im = Image.fromarray(np.uint8(resized)).convert('RGB')
            frame_array.append(im)
        count += 1
    print("Number of frames extracted ",count)
    return frame_array

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the pre-trained model and processor
encoder_model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
encoder_model.to(device)

class PositionalEncoding(nn.Module):
    def __init__(self, embed_dim: int, drop_rate=0.1, max_len=3):
        super().__init__()

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2) * (-math.log(10000.0) / embed_dim))
        pe = torch.zeros(1, max_len, embed_dim)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, inp):
        """
        Args:
            x: Tensor, shape [batch_size, seq_len, embedding_dim]
        """
        inp = inp + self.pe[:, :inp.size(1)]
#         inp = inp + self.pe[:inp.size(0), :]
        return inp    

class AvgPool(nn.Module):
    def __init__(self):
        super().__init__()
        self.pool_layer = nn.AvgPool2d(3,stride=4)
    
    def forward(self,input):
        return self.pool_layer(input)
    
def extract_features(model,inputs,dummy_text_inputs):
    inputs = inputs.to(device)
    dummy_text_inputs = dummy_text_inputs.to(device)
    with torch.no_grad():
        outputs = model(input_ids=dummy_text_inputs, output_hidden_states=True, **inputs)
        penultimate_features = outputs.vision_model_output.last_hidden_state[:, :, :]  # [batch_size, sequence_length, hidden_size]
        return penultimate_features

# Get the input image size
H = 16
W = 16
D = 1024
DIMENSION = H*D

# Extract penultimate features
def create_dataset():
    dataset_features = {}
    for frame_np_array, name in tqdm(COMPHY('activityNet/all_videos/')):
        print("Name : ",name)
        final_vector = []
        total_number_of_frames = len(frame_np_array)
        avg_pool = AvgPool()
        for frame_idx in range(total_number_of_frames):
            img = frame_np_array[frame_idx]
            img = processor(images=img,return_tensors="pt")
            
            # Create a batch of dummy text inputs
            batch_size = img["pixel_values"].size(0)
            dummy_text_inputs = torch.zeros(batch_size, 1, dtype=torch.long, device=img["pixel_values"].device)
            features = extract_features(encoder_model,img,dummy_text_inputs)
            penultimate_features = features[:,1:,:]
            penultimate_features = penultimate_features.view(-1,H,W)

            low_resolution_features = avg_pool(penultimate_features)
            feat = low_resolution_features.detach().cpu().numpy().flatten()
            final_vector.append(feat)
        final_vector_np = np.array(final_vector)
        positional_encoding = PositionalEncoding(DIMENSION,max_len=total_number_of_frames)
        pe_final_vector = positional_encoding(torch.tensor(final_vector_np))
        pe_final_vector = pe_final_vector.reshape((pe_final_vector.shape[0],total_number_of_frames, 16,1024))
        pe_final_vector = pe_final_vector.reshape((pe_final_vector.shape[1],pe_final_vector.shape[2],pe_final_vector.shape[3]))
        print("Final Vector Shape : ",pe_final_vector.shape)
        feat = pe_final_vector.cpu().detach().numpy()
        dataset_features[name] = feat

    return dataset_features

dataset_features = create_dataset()

#dumping mega pickle file.

import pickle
with open('fast_out.pkl','wb') as f:
    pickle.dump(dataset_features, f)

TMP_DIR = Path('../temp')
TMP_DIR.mkdir(exist_ok=True)

import zipfile
import tempfile

#for each video in dataset_features, dump the features to a pickle file and zip it

try :
    if not os.path.exists('temp'):
        os.makedirs('temp')

except OSError:
    print('Error: Creating directory of data')

for video in dataset_features:
    with open(str(TMP_DIR) + '/' + video+'_fast.pkl','wb') as f:
        pickle.dump(dataset_features[video], f)
#     with tempfile.NamedTemporaryFile(dir=TMP_DIR) as f:
#         pickle.dump(dataset_features[video], f)
#         with zipfile.ZipFile('/kaggle/working/fast.zip', "a", compression=zipfile.ZIP_DEFLATED) as zipf:
#             zipf.write(f.name, video+'_fast.pkl')
with zipfile.ZipFile("fast.zip", mode="w", compression=zipfile.ZIP_DEFLATED) as archive:
    for file_path in TMP_DIR.iterdir():
        archive.write(file_path, arcname=file_path.name)