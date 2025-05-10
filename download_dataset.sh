#!/bin/bash

# Check if kaggle is installed
if ! command -v kaggle &> /dev/null; then
    echo "Kaggle CLI is not installed. Installing..."
    pip install kaggle
fi

# Check if kaggle.json exists
if [ ! -f ~/.kaggle/kaggle.json ]; then
    echo "Error: kaggle.json not found in ~/.kaggle/"
    echo "Please download your kaggle.json from https://www.kaggle.com/settings/account"
    echo "and place it in ~/.kaggle/kaggle.json"
    exit 1
fi

# Set permissions for kaggle.json
chmod 600 ~/.kaggle/kaggle.json

# Create data directory if it doesn't exist
mkdir -p data

# Download dataset
echo "Downloading dataset..."
kaggle datasets download -d phnvnh/yellow-sticky-traps-dataset-vip -p data --unzip

# Check if download was successful
if [ $? -eq 0 ]; then
    # Check if the directory structure is correct
    if [ ! -d "data/yellow-sticky-traps-dataset-main" ]; then
        echo "Error: Dataset structure is incorrect"
        echo "Expected: data/yellow-sticky-traps-dataset-main"
        echo "Please check the downloaded dataset"
        exit 1
    fi
    
    echo "Dataset downloaded and extracted successfully!"
    echo "Dataset location: $(pwd)/data/yellow-sticky-traps-dataset-main"
else
    echo "Error: Failed to download dataset"
    exit 1
fi 