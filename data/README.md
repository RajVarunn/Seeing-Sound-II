# VGGSound Data Preparation

This project processes the VGGSound dataset to extract representative frames and audio clips from video files. It uses OpenAI's CLIP model to select the video frame that best matches the video's caption (label).

## Overview

The core workflow involves:
1.  **Video Processing**: Iterating through video files.
2.  **Frame Selection**: Using CLIP (Contrastive Language-Image Pre-Training) to find the frame that semantically matches the video's label the best.
3.  **Audio Extraction**: Extracting the full audio track from the video.
4.  **Data Aggregation**: Compiling metadata (paths, filenames, captions) into a CSV file for downstream tasks.

## Prerequisites

The following Python libraries are required:

*   `pandas`
*   `opencv-python` (`cv2`)
*   `torch`
*   `Pillow` (`PIL`)
*   `transformers` (Hugging Face)
*   `moviepy`

## Workflow Description

### 1. Setup and Model Loading
The script initializes the CLIP model (`openai/clip-vit-base-patch32`) and processor. It checks for CUDA availability to accelerate processing.

### 2. Video Processing (`process_video_and_save_frame`)
For each video file:
*   **Metadata Extraction**: The YouTube ID and start timestamp are parsed from the filename.
*   **Caption Matching**: The script looks up the corresponding label (caption) from the `vggsound.csv` dataset.
*   **Frame Sampling**: The video is sampled at a rate of roughly 1 frame per second (1/10th of the FPS).
*   **CLIP Similarity**:
    *   The text caption is encoded into a text embedding.
    *   Each sampled frame is encoded into an image embedding.
    *   Cosine similarity is calculated between the text and image embeddings.
*   **Selection**: The frame with the highest similarity score is selected.
*   **Saving**: If the maximum similarity exceeds a threshold (0.25), the frame is saved as a `.png` file.

### 3. Audio Extraction (`save_full_audio_moviepy`)
If a valid frame is found and saved, the script extracts the audio from the original video file and saves it as a `.wav` file using `moviepy`.

### 4. Data Aggregation
The script iterates through the processed directories to verify the existence of image and audio pairs. It constructs a Pandas DataFrame containing:
*   `base_folder`: The source directory batch (e.g., `vggsound_00`).
*   `image_file`: The filename of the extracted frame.
*   `audio_file`: The filename of the extracted audio.
*   `caption`: The label associated with the clip.

Finally, duplicates are removed, and the dataset is saved to `main_dataV1.csv`.

## Files

*   `data_prep.ipynb`: The main Jupyter Notebook containing all the logic.
*   `main_dataV1.csv`: The generated dataset cataloging the processed files.
*   `main_dataV3.csv`: A versioned or filtered copy of the dataset.

## Usage

1.  Update the `base_paths` dictionary in the notebook to point to your local VGGSound data directories.
2.  Ensure `vggsound.csv` is available and the path is correctly set.
3.  Run the cells in `data_prep.ipynb` sequentially.
