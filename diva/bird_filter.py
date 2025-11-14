import pandas as pd
import os

# Read the CSV file
df = pd.read_csv("main_dataV1(in).csv")

# Define strictly bird-related keywords
bird_keywords = [
    'bird', 'duck', 'eagle', 'owl', 'pigeon', 'dove', 
    'pheasant', 'crow', 'chicken', 'woodpecker',
    'baltimore oriole', 'mynah', 'cuckoo'
]

# Create a stricter filter for bird sounds
def is_bird_sound(caption):
    caption = caption.lower()
    return any(
        keyword in caption and (
            'singing' in caption or 
            'chirping' in caption or 
            'cawing' in caption or 
            'quacking' in caption or
            'crowing' in caption or
            'cooing' in caption or
            'hooting' in caption or
            'pecking' in caption or
            'wings flapping' in caption or
            'calling' in caption or
            'squawking' in caption or
            'braying' in caption
        ) for keyword in bird_keywords
    )

# Apply the filter
mask = df['caption'].apply(is_bird_sound)
bird_df = df[mask]

# Check if files exist in vggsound_all folder
vggsound_folder = "vggsound_all"
if os.path.exists(vggsound_folder):
    # Filter to only include rows where both image and audio files exist
    def files_exist(row):
        image_path = os.path.join(vggsound_folder, row['image_file'])
        audio_path = os.path.join(vggsound_folder, row['audio_file'])
        return os.path.exists(image_path) and os.path.exists(audio_path)
    
    existing_files_mask = bird_df.apply(files_exist, axis=1)
    bird_df = bird_df[existing_files_mask]
    print(f"Filtered to {len(bird_df)} entries with existing files in {vggsound_folder}")
else:
    print(f"Warning: {vggsound_folder} folder not found. Proceeding with all filtered entries.")

# Sort and save results
bird_df = bird_df.sort_values('caption')
output_filename = "bird_sounds_filtered.csv"
bird_df.to_csv(output_filename, index=False)

print(f"Found {len(bird_df)} bird-related sounds")
print(f"Results saved to {output_filename}")

print("\nBird sounds found:")
for i, row in enumerate(bird_df[['image_file', 'audio_file', 'caption']].values, 1):
    print(f"{i}. {row[2]} (Image: {row[0]}, Audio: {row[1]})")