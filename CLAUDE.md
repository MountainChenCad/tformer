Project Guide: Adapting ITFormer for High-Speed Train Bearing Fault Diagnosis
1. Project Goal

The objective is to adapt the powerful, pre-existing ITFormer framework to solve the 2025 China Graduate Mathematical Contest in Modeling Problem E. This involves reframing the bearing fault diagnosis task—a classic transfer learning problem—into a Time-Series Question Answering (TS-QA) problem.

We will leverage the existing ITFormer models to perform inference on a new, unlabeled target dataset of high-speed train bearing vibrations after being informed by a labeled source dataset from a test rig. The core challenge is to bridge the gap between the project's original data format (.h5, .jsonl) and the competition's data format (.mat).

2. Phase 1: Setup & Data Preparation

This phase focuses on ingesting the competition's .mat files and converting them into the format required by the ITFormer framework.

Step 2.1: New Directory Structure

First, organize the project directories as follows. This structure will hold the raw competition data and the processed outputs.

code
Code
download
content_copy
expand_less
ITFormer-ICML25/
├── competition_data/
│   ├── source_domain/          # Place all 161 source .mat files here
│   └── target_domain/          # Place all 16 target .mat files here
├── dataset/
│   └── bearing_dataset/        # This is where we will save the processed files
│       ├── bearing_time_series.h5
│       ├── source_qa.jsonl
│       └── target_qa.jsonl
├── LLM/
├── checkpoints/
├── inference_results/
└── ... (rest of the ITFormer project files)
Step 2.2: Create a Data Preprocessing Script

We will create a new Python script named preprocess_bearing_data.py in the root directory. This script is the cornerstone of the adaptation, responsible for converting the .mat files into ITFormer's native format.

Key tasks for this script:

Read .mat files from competition_data/.

Apply the multi-channel, time-frequency fusion strategy. For each signal segment, we will generate 6 channels:

Time-domain signals (DE, FE, BA).

Frequency-domain signals via Continuous Wavelet Transform (CWT) (DE, FE, BA).

Reframe the classification task into a QA task.

Save the processed data as bearing_time_series.h5 and corresponding QA pairs in .jsonl format.

preprocess_bearing_data.py:

code
Python
download
content_copy
expand_less
import os
import glob
import json
import re
import h5py
import numpy as np
import scipy.io as sio
import pywt
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler

# --- Configuration ---
SOURCE_DATA_DIR = 'competition_data/source_domain'
TARGET_DATA_DIR = 'competition_data/target_domain'
OUTPUT_DIR = 'dataset/bearing_dataset'
H5_FILENAME = os.path.join(OUTPUT_DIR, 'bearing_time_series.h5')
SOURCE_QA_FILENAME = os.path.join(OUTPUT_DIR, 'source_qa.jsonl')
TARGET_QA_FILENAME = os.path.join(OUTPUT_DIR, 'target_qa.jsonl')

SEGMENT_LENGTH = 2048  # Fixed length for each time series segment
STRIDE = 1024          # 50% overlap for data augmentation
FIXED_QUESTION = "请分析这段振动信号并判断轴承的健康状态。"

# --- Mappings ---
LABEL_MAP = {
    'N': "轴承运行正常。",
    'IR': "轴承存在内圈故障。",
    'OR': "轴承存在外圈故障。",
    'B': "轴承存在滚动体故障。"
}

def get_label_from_filename(filename):
    """Extracts fault label from the source domain filename."""
    base = os.path.basename(filename)
    if 'N_Normal' in base or base.startswith('N_'):
        return 'N'
    match = re.match(r"([A-Z]+)", base)
    if match:
        label_key = match.group(1)
        if label_key in LABEL_MAP:
            return label_key
    return None

def process_signal_segment(segment):
    """Applies Z-score normalization and CWT to a single segment."""
    # Time-domain: Z-score normalization
    scaler = StandardScaler()
    time_feature = scaler.fit_transform(segment.reshape(-1, 1)).flatten()

    # Frequency-domain: Continuous Wavelet Transform (CWT)
    # Using 'morl' wavelet, common for vibration analysis
    coeffs, _ = pywt.cwt(segment, scales=np.arange(1, 65), wavelet='morl')
    freq_feature = np.mean(np.abs(coeffs), axis=0)
    
    # Resize CWT output to match segment length and normalize
    if len(freq_feature) != SEGMENT_LENGTH:
        freq_feature = np.interp(
            np.linspace(0, 1, SEGMENT_LENGTH),
            np.linspace(0, 1, len(freq_feature)),
            freq_feature
        )
    freq_feature = scaler.fit_transform(freq_feature.reshape(-1, 1)).flatten()
    
    return time_feature, freq_feature

def process_mat_file(filepath):
    """Loads a .mat file and extracts DE, FE, BA channels."""
    try:
        mat_data = sio.loadmat(filepath)
        keys = mat_data.keys()
        
        # Find the correct keys for each channel
        de_key = next((k for k in keys if 'DE_time' in k), None)
        fe_key = next((k for k in keys if 'FE_time' in k), None)
        ba_key = next((k for k in keys if 'BA_time' in k), None)
        
        # If any channel is missing, fill with zeros as a fallback
        de_signal = mat_data[de_key].flatten() if de_key else np.zeros(1)
        fe_signal = mat_data[fe_key].flatten() if fe_key else np.zeros_like(de_signal)
        ba_signal = mat_data[ba_key].flatten() if ba_key else np.zeros_like(de_signal)

        # Ensure all signals have the same length by padding the shorter ones
        max_len = max(len(de_signal), len(fe_signal), len(ba_signal))
        de_signal = np.pad(de_signal, (0, max_len - len(de_signal)))
        fe_signal = np.pad(fe_signal, (0, max_len - len(fe_signal)))
        ba_signal = np.pad(ba_signal, (0, max_len - len(ba_signal)))
        
        return de_signal, fe_signal, ba_signal
    except Exception as e:
        print(f"Warning: Could not process {filepath}. Error: {e}")
        return None, None, None

def create_dataset(data_dir, is_source=True):
    """Main function to process a directory of .mat files."""
    all_features = []
    qa_pairs = []
    
    mat_files = glob.glob(os.path.join(data_dir, '*.mat'))
    
    # Use a unique global ID for each sample across all files
    sample_id_counter = 1 
    
    for filepath in tqdm(mat_files, desc=f"Processing {'Source' if is_source else 'Target'} Domain"):
        de, fe, ba = process_mat_file(filepath)
        if de is None:
            continue
            
        label = get_label_from_filename(filepath) if is_source else None
        
        # Create segments using a sliding window
        for i in range(0, len(de) - SEGMENT_LENGTH + 1, STRIDE):
            seg_de = de[i : i + SEGMENT_LENGTH]
            seg_fe = fe[i : i + SEGMENT_LENGTH]
            seg_ba = ba[i : i + SEGMENT_LENGTH]
            
            # Get time and frequency features for each channel
            time_de, freq_de = process_signal_segment(seg_de)
            time_fe, freq_fe = process_signal_segment(seg_fe)
            time_ba, freq_ba = process_signal_segment(seg_ba)
            
            # Stack features into a (6, SEGMENT_LENGTH) array
            stacked_features = np.stack([
                time_de, time_fe, time_ba,
                freq_de, freq_fe, freq_ba
            ], axis=0)
            
            all_features.append(stacked_features)
            
            # Create QA pair
            file_id = os.path.basename(filepath)
            conversation = [
                {"from": "human", "value": FIXED_QUESTION, "stage": "1", "attribute": "understanding"},
            ]
            if is_source:
                if label not in LABEL_MAP:
                    print(f"Warning: Skipping file with unknown label pattern: {filepath}")
                    continue
                conversation.append({"from": "gpt", "value": LABEL_MAP[label]})

            qa_pairs.append({
                "id": str(sample_id_counter), # Unique ID for this segment
                "file_id": file_id, # Original filename for later aggregation
                "conversations": conversation
            })
            sample_id_counter += 1

    return np.array(all_features), qa_pairs


if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # --- Process Source Domain ---
    source_features, source_qa = create_dataset(SOURCE_DATA_DIR, is_source=True)
    
    # --- Process Target Domain ---
    target_features, target_qa = create_dataset(TARGET_DATA_DIR, is_source=False)
    
    # --- Save to H5 and JSONL ---
    print("Saving processed data...")
    with h5py.File(H5_FILENAME, 'w') as hf:
        # Combine all features and save
        all_features_combined = np.concatenate([source_features, target_features], axis=0)
        hf.create_dataset('seq_data', data=all_features_combined)

    # Adjust IDs in target_qa to follow source_qa
    start_id_target = len(source_qa) + 1
    for i, item in enumerate(target_qa):
        item['id'] = str(start_id_target + i)

    with open(SOURCE_QA_FILENAME, 'w', encoding='utf-8') as f:
        for item in source_qa:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
            
    with open(TARGET_QA_FILENAME, 'w', encoding='utf-8') as f:
        for item in target_qa:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
            
    print(f"Data processing complete.")
    print(f"Total source segments: {len(source_qa)}")
    print(f"Total target segments: {len(target_qa)}")
    print(f"H5 file saved to: {H5_FILENAME}")
    print(f"JSONL files saved to: {SOURCE_QA_FILENAME} and {TARGET_QA_FILENAME}")
3. Phase 2: Adapting the Codebase
Step 3.1: Modify yaml/infer.yaml

Update the configuration to point to our newly created target dataset files.

yaml/infer.yaml:

code
Yaml
download
content_copy
expand_less
# ... (keep existing model parameters like d_model, n_heads etc.)

# --- UPDATE THESE PATHS ---
# Path to the HDF5 file containing ALL time series data (source + target)
ts_path_test: dataset/bearing_dataset/bearing_time_series.h5

# Path to the JSONL file for the TARGET domain
qa_path_test: dataset/bearing_dataset/target_qa.jsonl

# ... (keep other parameters like fp16, dataloader_num_workers etc.)
Step 3.2: Modify inference.py

The main inference script must be adapted to run in a "predict-only" mode for our unlabeled target data and to aggregate results.

Key Modifications:

Add a --mode argument to switch between evaluation and prediction.

Add a function to aggregate segment-level predictions into file-level predictions using majority voting.

Save the final aggregated predictions to a CSV file.

Apply these changes to inference.py:

code
Python
download
content_copy
expand_less
# Add these imports at the top of inference.py
import pandas as pd
from collections import Counter

# ... (keep existing functions like set_seed, save_results, etc.)

def aggregate_and_save_predictions(results, output_dir, config_name):
    """Aggregates segment predictions to file-level and saves to CSV."""
    if not results:
        print("Warning: No results to aggregate.")
        return

    # Use file_id which we added during preprocessing
    df = pd.DataFrame(results)
    
    # Majority vote aggregation
    def majority_vote(series):
        return Counter(series).most_common(1)[0][0]

    agg_results = df.groupby('file_id')['prediction'].apply(majority_vote).reset_index()
    agg_results.rename(columns={'prediction': 'predicted_fault_description'}, inplace=True)
    
    # Save to CSV
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"aggregated_predictions_{config_name}_{timestamp}.csv"
    filepath = os.path.join(output_dir, filename)
    agg_results.to_csv(filepath, index=False, encoding='utf-8')
    print(f"\n✅ Aggregated predictions saved to: {filepath}")

def main_inference(args):
    """Main inference pipeline."""
    # ... (existing setup code)

    # --- MAIN LOOP ---
    results = []
    with torch.no_grad():
        # ... (existing tqdm loop)
        for batch_idx, batch in enumerate(batch_iterator):
            # ... (existing model.generate call)
            
            # --- MODIFICATION: Store file_id along with results ---
            # Retrieve file_id from the dataset/dataloader
            # NOTE: This requires a small change in dataset.py or using the index
            # to map back. We will use the 'index' to get the original file_id.
            batch_indices = batch['index'].tolist()
            original_samples = [test_dataset.datas[i] for i in batch_indices]

            for i in range(len(batch_predictions)):
                prediction = batch_predictions[i].split('assistant\n')[-1]
                results.append({
                    "index": batch['index'][i].item(),
                    "file_id": original_samples[i]['file_id'], # Store original filename
                    "prediction": prediction.strip(),
                    # For predict mode, label is not available/relevant
                    "label": "N/A" if args.mode == 'predict' else batch_labels[i],
                })

    if accelerator.is_main_process:
        if args.mode == 'evaluate':
            print("\n📊 Calculating evaluation metrics...")
            metrics = compute_metrics_from_results(results, args)
            print_metrics(metrics)
            config_base = os.path.basename(args.config).split('.yaml')[0]
            save_results(results, args.output_dir, config_base) # Save segment-level results
            save_metrics(metrics, args.output_dir, config_base)
        elif args.mode == 'predict':
            print("\n🗳️ Aggregating predictions for target domain...")
            config_base = os.path.basename(args.config).split('.yaml')[0]
            aggregate_and_save_predictions(results, args.output_dir, config_base)


# --- Add file_id to the TsQaDataset __getitem__ return dict in dataset/dataset.py ---
# This is a minor but necessary change. In dataset.py, in the __getitem__ method:
# Ensure the returned dictionary includes 'file_id': sample['file_id']
# And add it to the collator. This is cleaner than mapping by index.
# For simplicity in this guide, we'll map by index as shown above.

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='ITFormer Inference for Bearing Fault Diagnosis')
    # ... (existing arguments)
    
    # --- ADD NEW ARGUMENT ---
    parser.add_argument(
        '--mode', 
        type=str, 
        default='predict', 
        choices=['evaluate', 'predict'], 
        help='Set to "predict" for unlabeled target data, "evaluate" for labeled data.'
    )
    
    args = parser.parse_args()
    main_inference(args)
4. Phase 3: Execution Plan

Follow these steps to run the complete project.

Step 4.1: Environment Setup

Ensure the environment is correctly set up using the provided instructions.

code
Bash
download
content_copy
expand_less
# Install core dependencies
pip install -r requirements.txt

# Install additional dependencies required for our new script
pip install scikit-learn scipy pywavelets
Step 4.2: Place Data

Place the downloaded competition .mat files into the competition_data/source_domain/ and competition_data/target_domain/ directories respectively.

Step 4.3: Run Preprocessing

Execute the script we created to convert the data. This will create the necessary .h5 and .jsonl files in dataset/bearing_dataset/.

code
Bash
download
content_copy
expand_less
python preprocess_bearing_data.py
Step 4.4: Run Inference on Target Data

Now, run the modified inference script. It will load the ITFormer model, process the target domain data, and save the final aggregated predictions in a CSV file inside the inference_results directory.

code
Bash
download
content_copy
expand_less
# Ensure you have downloaded the ITFormer-0.5B model and the Qwen2.5-0.5B-Instruct LLM
# as per the original project's README.

# Run inference in "predict" mode on the target data
python inference.py --config yaml/infer.yaml --mode predict --model_checkpoint checkpoints/ITFormer-0.5B

After execution, a file named aggregated_predictions_infer_YYYYMMDD_HHMMSS.csv will be generated in inference_results/, containing the predicted fault type for each of the 16 target domain files (A.mat to P.mat). This file is the final output for the competition task.