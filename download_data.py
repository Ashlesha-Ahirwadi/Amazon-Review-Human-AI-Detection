import os

# Make sure the data directory exists
os.makedirs('data', exist_ok=True)

# Download the dataset using the Kaggle CLI
# Make sure you have your kaggle.json API key set up in ~/.kaggle/
os.system('kaggle datasets download -d mexwell/fake-reviews-dataset -p data --unzip')

print("Dataset downloaded to data/") 