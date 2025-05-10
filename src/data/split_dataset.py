import os
import json
import random
import shutil
from pathlib import Path
from collections import defaultdict

def count_objects_per_file(labels_dir):
    """
    Count number of objects of each class in each file
    """
    file_stats = defaultdict(lambda: defaultdict(int))
    class_mapping = {0: 'WF', 1: 'MR', 2: 'NC'}
    
    for label_file in Path(labels_dir).glob('*.txt'):
        with open(label_file, 'r') as f:
            for line in f:
                class_id = int(line.strip().split()[0])
                class_name = class_mapping[class_id]
                file_stats[label_file.name][class_name] += 1
    
    return file_stats

def split_dataset(file_stats, train_ratio=0.9, seed=42):
    """
    Split dataset into train and test sets while maintaining class balance
    """
    random.seed(seed)
    
    # Group files by their class distribution
    files_by_distribution = defaultdict(list)
    for file_name, class_counts in file_stats.items():
        # Create a tuple of class counts as key
        dist_key = tuple(sorted(class_counts.items()))
        files_by_distribution[dist_key].append(file_name)
    
    # Split each group
    train_files = []
    test_files = []
    
    # Calculate total number of files
    total_files = len(file_stats)
    train_size = int(total_files * train_ratio)
    
    # First, add all files to train
    all_files = list(file_stats.keys())
    random.shuffle(all_files)
    train_files = all_files[:train_size]
    test_files = all_files[train_size:]
    
    return train_files, test_files

def create_split_directories(output_dir):
    """
    Create train and test directories
    """
    train_dir = os.path.join(output_dir, 'train')
    test_dir = os.path.join(output_dir, 'test')
    
    # Create main directories
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)
    
    # Create subdirectories for images and labels
    for dir_path in [train_dir, test_dir]:
        os.makedirs(os.path.join(dir_path, 'images'), exist_ok=True)
        os.makedirs(os.path.join(dir_path, 'labels'), exist_ok=True)
    
    return train_dir, test_dir

def copy_files(file_list, src_dir, dst_dir):
    """
    Copy label files and corresponding images
    """
    for label_file in file_list:
        # Copy label file
        src_label = os.path.join(src_dir, 'labels', label_file)
        dst_label = os.path.join(dst_dir, 'labels', label_file)
        shutil.copy2(src_label, dst_label)
        
        # Copy corresponding image
        image_file = label_file.replace('.txt', '.jpg')
        src_image = os.path.join(src_dir, 'images', image_file)
        dst_image = os.path.join(dst_dir, 'images', image_file)
        shutil.copy2(src_image, dst_image)

def print_class_distribution(stats, title):
    """
    Print class distribution in a fixed order
    """
    print(f"\n{title}:")
    for class_name in ['WF', 'MR', 'NC']:
        count = stats.get(class_name, 0)
        print(f"  {class_name}: {count}")

def main():
    # Paths
    base_dir = 'data/yellow-sticky-traps-dataset-main'
    output_dir = 'data/data_for_train_test'
    
    # Create output directories
    train_dir, test_dir = create_split_directories(output_dir)
    
    # Count objects per file
    labels_dir = os.path.join(base_dir, 'labels')
    file_stats = count_objects_per_file(labels_dir)
    
    # Split dataset
    train_files, test_files = split_dataset(file_stats)
    
    # Copy files
    copy_files(train_files, base_dir, train_dir)
    copy_files(test_files, base_dir, test_dir)
    
    # Calculate statistics
    train_stats = defaultdict(int)
    test_stats = defaultdict(int)
    
    for file_name in train_files:
        for class_name, count in file_stats[file_name].items():
            train_stats[class_name] += count
            
    for file_name in test_files:
        for class_name, count in file_stats[file_name].items():
            test_stats[class_name] += count
    
    print("\nDataset Split Statistics:")
    print(f"Total files: {len(file_stats)}")
    print(f"Train files: {len(train_files)}")
    print(f"Test files: {len(test_files)}")
    
    # Print class distributions in fixed order
    print_class_distribution(train_stats, "Class Distribution in Train Set")
    print_class_distribution(test_stats, "Class Distribution in Test Set")
    
    # Save split information
    split_info = {
        'train_files': train_files,
        'test_files': test_files,
        'train_stats': dict(train_stats),
        'test_stats': dict(test_stats)
    }
    
    with open(os.path.join(output_dir, 'split_info.json'), 'w') as f:
        json.dump(split_info, f, indent=2)
    
    print(f"\nDataset has been split and saved to {output_dir}")
    print(f"Split information saved to {os.path.join(output_dir, 'split_info.json')}")
    
    # Remove annotations directory since we have labels in correct format
    try:
        annotations_dir = os.path.join(base_dir, 'annotations')
        shutil.rmtree(annotations_dir)
        print(f"\nRemoved annotations directory: {annotations_dir}")
    except Exception as e:
        print(f"\nWarning: Failed to remove annotations directory: {e}")

if __name__ == '__main__':
    main() 