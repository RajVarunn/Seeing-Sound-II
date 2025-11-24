import os
import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from PIL import Image
import torchvision.transforms as transforms
import torchaudio
import torchaudio.transforms as T

from transformers import ClapModel, ClapProcessor  # using HF CLAP
import numpy as np
import faiss
from tqdm import tqdm

class Sound2SceneFaissDataset(Dataset):
    """
    Dataset that loads images from disk and accesses precomputed audio embeddings stored in FAISS.
    The CSV/DataFrame must contain columns: 'base_folder', 'image_file', 'audio_file'.
    """

    def __init__(self, df, faiss_index, embeddings_array, img_size=(64, 64), base_dir=None):
        super().__init__()
        self.df = df.reset_index(drop=True)
        self.faiss_index = faiss_index
        self.embeddings = embeddings_array  # numpy array of shape (num_samples, embed_dim)
        self.base_dir = base_dir
        self.transform = transforms.Compose([
            transforms.Resize(img_size),
            transforms.ToTensor(),
            transforms.Normalize([0.5] * 3, [0.5] * 3)
        ])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # Load image
        if self.base_dir is None:
            folder_name = self.df.iloc[idx]['base_folder']
        else:
            folder_name = self.df.iloc[idx]['base_folder']
            folder_name = os.path.join(self.base_dir, folder_name)

        img_name = self.df.iloc[idx]['image_file']
        img_path = os.path.join(folder_name, "image", img_name)
        if not os.path.exists(img_path):
            print(f"Warning: Image path does not exist: {img_path}")
            # You may choose to raise an error or return a placeholder
            raise FileNotFoundError(f"Image file not found: {img_path}")

        # try:
        image = Image.open(img_path).convert('RGB')
        image = self.transform(image)
        # except Exception as e:
            # print(f"Error loading image {img_path}: {e}")
            # raise e

        # Load precomputed audio embedding from numpy array (no faiss search here, direct access)
        audio_embedding = torch.from_numpy(self.embeddings[idx])

        return image, audio_embedding

import os

def build_faiss_index(df, clap_model_name="laion/clap-htsat-unfused", target_sr=48000, device='cuda',
                      save_dir=None, index_filename="faiss_index.bin", embeddings_filename="embeddings.npy", csv_filename = "metadata.csv"):
    """
    Precompute audio embeddings for all samples in the DataFrame and build FAISS index.
    Resamples audio to target_sr if needed.
    Optionally saves the FAISS index and embeddings array to disk if save_dir is provided.
    Returns: (faiss_index, embeddings_array)
    """

    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    processor = ClapProcessor.from_pretrained(clap_model_name)
    model = ClapModel.from_pretrained(clap_model_name).to(device)
    model.eval()

    embeddings = []

    for idx in tqdm(range(len(df)), desc="Processing audio embeddings"):
        folder_name = df.iloc[idx]['base_folder']
        audio_name = df.iloc[idx]['audio_file']
        audio_path = os.path.join(folder_name, "audio", audio_name)

        if not os.path.exists(audio_path):
            print(f"Warning: Audio path does not exist: {audio_path}")
            embeddings.append(np.zeros(model.config.projection_dim, dtype=np.float32))
            continue

        try:
            waveform, sr = torchaudio.load(audio_path)
            if sr != target_sr:
                resampler = T.Resample(orig_freq=sr, new_freq=target_sr)
                waveform = resampler(waveform)
                sr = target_sr

            waveform = waveform.mean(dim=0, keepdim=True)

            waveform_cpu = waveform.squeeze().cpu().numpy()
            # print(f"Processing {audio_path}, waveform shape: {waveform.shape}, sr: {sr}")

            inputs = processor(audios=waveform_cpu, sampling_rate=sr,
                                return_tensors="pt", padding=True)
            inputs = {k: v.to(device) for k, v in inputs.items()}

            with torch.no_grad():
                audio_features = model.get_audio_features(**inputs)

            emb = audio_features.squeeze(0).cpu().numpy()
            embeddings.append(emb)

        except Exception as e:
            print(f"Error processing audio {audio_path}: {e}")
            embeddings.append(np.zeros(model.config.projection_dim, dtype=np.float32))

    embeddings = np.vstack(embeddings).astype('float32')
    embed_dim = embeddings.shape[1]

    index = faiss.IndexFlatL2(embed_dim)
    index.add(embeddings)

    # Save if directory given
    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
        index_path = os.path.join(save_dir, index_filename)
        embeddings_path = os.path.join(save_dir, embeddings_filename)
        df_path = os.path.join(save_dir, csv_filename)

        faiss.write_index(index, index_path)
        np.save(embeddings_path, embeddings)
        df.to_csv(df_path, index=False)

        print(f"FAISS index saved to {index_path}")
        print(f"Embeddings array saved to {embeddings_path}")
        print(f"Dataset metadata saved to {df_path}")

    return index, embeddings

def load_faiss_index_embeddings_metadata(load_dir, index_filename="faiss_index.bin", embeddings_filename="embeddings.npy", df_filename="dataset_metadata.csv"):
    index_path = os.path.join(load_dir, index_filename)
    embeddings_path = os.path.join(load_dir, embeddings_filename)
    df_path = os.path.join(load_dir, df_filename)
    
    if not os.path.exists(index_path):
        raise FileNotFoundError(f"FAISS index file not found: {index_path}")
    if not os.path.exists(embeddings_path):
        raise FileNotFoundError(f"Embeddings file not found: {embeddings_path}")
    if not os.path.exists(df_path):
        raise FileNotFoundError(f"Dataset metadata file not found: {df_path}")
    
    index = faiss.read_index(index_path)
    embeddings = np.load(embeddings_path)
    df = pd.read_csv(df_path)
    
    print(f"Loaded FAISS index from {index_path}")
    print(f"Loaded embeddings array from {embeddings_path} with shape {embeddings.shape}")
    print(f"Loaded dataset metadata from {df_path} with {len(df)} rows")
    
    return index, embeddings, df



def get_dataloader_from_faiss(df, faiss_index, embeddings_array,
                              batch_size=32, img_size=(64, 64),
                              shuffle=True, num_workers=4, base_dir=None):
    """
    Create dataloader using precomputed FAISS audio embeddings and images loaded on the fly.
    """
    dataset = Sound2SceneFaissDataset(df, faiss_index, embeddings_array, img_size=img_size, base_dir=base_dir)
    dataloader = DataLoader(dataset, batch_size=batch_size,
                            shuffle=shuffle, num_workers=num_workers,
                            pin_memory=True)
    return dataloader
