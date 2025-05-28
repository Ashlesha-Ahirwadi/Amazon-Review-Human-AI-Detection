# Amazon Review Human/AI Detection

This project focuses on detecting whether Amazon reviews are written by humans or AI using machine learning techniques.

## Setup

1. Create the conda environment:
   ```bash
   conda env create -f environment.yml
   ```

2. Activate the environment:
   ```bash
   conda activate review_detect
   ```

3. Download the dataset:
   ```bash
   python download_data.py
   ```
   This script will download the required dataset for the project. Make sure you have an active internet connection.

## Run Models

### Baseline Model
To run the baseline model:
```bash
python baseline/baseline_model.py
```

### BERT Fine-tuning
To run the BERT fine-tuning model:
```bash
python bert_finetune/bert_finetune.py
```

## Note
- Make sure to run `download_data.py` first to obtain the dataset before running any models
- The dataset will be stored in the `data` directory
