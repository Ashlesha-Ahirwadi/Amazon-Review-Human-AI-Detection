# Amazon Review Human/AI Detection

## Setup

1. Create the conda environment:
   ```bash
   conda env create -f environment.yml
   ```
2. Activate the environment:
   ```bash
   conda activate review_detect
   ```
3. Download the data (if not already done):
   ```bash
   python download_data.py
   ```

## Run Baseline Model
```bash
python baseline/baseline_model.py
```

## Run BERT Fine-tuning
```bash
python bert_finetune/bert_finetune.py 