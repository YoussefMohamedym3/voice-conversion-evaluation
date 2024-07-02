import numpy as np
from resemblyzer import preprocess_wav, VoiceEncoder
from pathlib import Path
import json
from tqdm import tqdm
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

def calculate_score(model, metadata_path):
    metadata = json.load(Path(metadata_path).open())
    true_labels = []
    similarity_scores = []
    
    for pair in tqdm(metadata["pairs"]):
        wav = preprocess_wav(pair["converted"])
        source_emb = model.embed_utterance(wav)
        
        for target_file in pair["tgt_utts"]:
            target_wav = preprocess_wav(target_file)
            target_emb = model.embed_utterance(target_wav)
            
            cosine_similarity = (
                np.inner(source_emb, target_emb) /
                (np.linalg.norm(source_emb) * np.linalg.norm(target_emb))
            )
            
            similarity_scores.append(cosine_similarity)
            true_labels.append(1 if pair["label"] == "match" else 0)
    
    return true_labels, similarity_scores

def find_optimal_threshold(true_labels, scores):
    fpr, tpr, thresholds = roc_curve(true_labels, scores)
    optimal_idx = np.argmax(tpr - fpr)
    optimal_threshold = thresholds[optimal_idx]
    roc_auc = auc(fpr, tpr)
    
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.show()

    return optimal_threshold, roc_auc

# Load the model
model = VoiceEncoder()

# Get paths from environment variables
metadata_path = os.getenv('METADATA_PATH')
output_path = os.getenv('Threshold_Path')

# Calculate scores
true_labels, similarity_scores = calculate_score(model, metadata_path)

# Find optimal threshold
optimal_threshold, roc_auc = find_optimal_threshold(true_labels, similarity_scores)
print(f'Optimal Threshold: {optimal_threshold}, AUC: {roc_auc}')

# Write the optimal threshold to a .txt file if it doesn't exist
output_file = Path(output_path)
if not output_file.exists():
    output_file.write_text(f'Optimal Threshold: {optimal_threshold}')
    print(f'Created new file: {output_file}')
else:
    print(f'File already exists: {output_file}')
