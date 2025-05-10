#!/bin/bash

# Create data directory if it doesn't exist
mkdir -p data

# Download dataset from Google Drive using file ID
echo "Downloading dataset from Google Drive..."
gdown 1c-b0r294GiZ3akk5XLgOYZYkcyTccoUs -O data/dataset.zip

# Check if download was successful
if [ $? -eq 0 ]; then
    echo "Dataset downloaded successfully!"
    
    # Unzip the dataset
    echo "Extracting dataset..."
    unzip -q data/dataset.zip -d data/
    
    # Check if the directory structure is correct
    if [ ! -d "data/yellow-sticky-traps-dataset-main" ]; then
        echo "Error: Dataset structure is incorrect"
        echo "Expected: data/yellow-sticky-traps-dataset-main"
        echo "Please check the downloaded dataset"
        rm data/dataset.zip  # Remove zip file even if extraction failed
        exit 1
    fi
    
    # Remove zip file after successful extraction
    rm data/dataset.zip
    echo "Removed zip file"
    
    echo "Dataset extracted successfully!"
    echo "Dataset location: $(pwd)/data/yellow-sticky-traps-dataset-main"
else
    echo "Error: Failed to download dataset"
    exit 1
fi 