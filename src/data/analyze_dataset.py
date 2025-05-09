import os
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from collections import defaultdict

def analyze_labels(labels_dir):
    """
    Analyze YOLO format labels and return statistics
    """
    stats = {
        'total_images': 0,
        'total_objects': 0,
        'class_distribution': defaultdict(int),
        'objects_per_image': [],
        'object_sizes': {
            'WF': [],  # Whiteflies
            'MR': [],  # Macrolophus
            'NC': []   # Nesidiocoris
        }
    }
    
    class_mapping = {
        0: 'WF',
        1: 'MR',
        2: 'NC'
    }
    
    # Process each label file
    for label_file in Path(labels_dir).glob('*.txt'):
        stats['total_images'] += 1
        objects_in_image = 0
        
        with open(label_file, 'r') as f:
            for line in f:
                objects_in_image += 1
                stats['total_objects'] += 1
                
                # Parse YOLO format: class_id x_center y_center width height
                class_id, x_center, y_center, width, height = map(float, line.strip().split())
                class_name = class_mapping[int(class_id)]
                
                # Update class distribution
                stats['class_distribution'][class_name] += 1
                
                # Store object size (width * height)
                stats['object_sizes'][class_name].append(width * height)
        
        stats['objects_per_image'].append(objects_in_image)
    
    # Convert defaultdict to regular dict for JSON serialization
    stats['class_distribution'] = dict(stats['class_distribution'])
    
    # Calculate additional statistics
    stats['objects_per_image_stats'] = {
        'mean': float(np.mean(stats['objects_per_image'])),
        'median': float(np.median(stats['objects_per_image'])),
        'min': float(np.min(stats['objects_per_image'])),
        'max': float(np.max(stats['objects_per_image']))
    }
    
    # Calculate object size statistics per class
    for class_name in stats['object_sizes']:
        sizes = stats['object_sizes'][class_name]
        if sizes:  # Only calculate if there are objects of this class
            stats['object_sizes'][class_name] = {
                'mean': float(np.mean(sizes)),
                'median': float(np.median(sizes)),
                'min': float(np.min(sizes)),
                'max': float(np.max(sizes))
            }
    
    return stats

def plot_statistics(stats, output_dir, prefix=""):
    """
    Create visualizations of the dataset statistics
    """
    os.makedirs(output_dir, exist_ok=True)
    # 1. Class Distribution Bar Plot
    plt.figure(figsize=(10, 6))
    ordered_classes = ['WF', 'MR', 'NC']
    counts = [stats['class_distribution'].get(cls, 0) for cls in ordered_classes]
    plt.bar(ordered_classes, counts)
    plt.title('Class Distribution')
    plt.xlabel('Class')
    plt.ylabel('Number of Objects')
    plt.savefig(os.path.join(output_dir, f'{prefix}class_distribution.png'))
    plt.close()
    # 2. Objects per Image Histogram
    plt.figure(figsize=(10, 6))
    plt.hist(stats['objects_per_image'], bins=30)
    plt.title('Objects per Image Distribution')
    plt.xlabel('Number of Objects')
    plt.ylabel('Number of Images')
    plt.savefig(os.path.join(output_dir, f'{prefix}objects_per_image.png'))
    plt.close()
    # 3. Object Size Distribution Box Plot
    plt.figure(figsize=(12, 6))
    size_data = []
    labels = []
    for class_name in ordered_classes:
        if class_name in stats['object_sizes']:
            sizes = stats['object_sizes'][class_name]
            if isinstance(sizes, dict):
                sizes = [sizes['mean'], sizes['median'], sizes['min'], sizes['max']]
            if sizes:
                size_data.append(sizes)
                labels.append(class_name)
    plt.boxplot(size_data, labels=labels)
    plt.title('Object Size Distribution by Class')
    plt.xlabel('Class')
    plt.ylabel('Normalized Area (width * height)')
    plt.yscale('log')
    plt.savefig(os.path.join(output_dir, f'{prefix}object_sizes.png'))
    plt.close()

def main():
    parser = argparse.ArgumentParser(description='Analyze YOLO format dataset statistics')
    parser.add_argument('--data_dir', type=str, required=True, help='Path to dataset directory')
    parser.add_argument('--output_dir', type=str, required=True, help='Output directory to save results')
    parser.add_argument('--labels_dir', type=str, default=None, help='Path to YOLO labels directory (optional, overrides data_dir)')
    args = parser.parse_args()
    
    # Determine labels directory
    if args.labels_dir:
        labels_dir = args.labels_dir
        prefix = os.path.basename(os.path.dirname(labels_dir.rstrip('/'))) + "_"
    else:
        labels_dir = os.path.join(args.data_dir, 'yellow-sticky-traps-dataset-main', 'labels')
        prefix = ""
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Analyze dataset
    stats = analyze_labels(labels_dir)
    
    # Save statistics to JSON
    json_path = os.path.join(args.output_dir, f'{prefix}stats.json')
    with open(json_path, 'w') as f:
        json.dump(stats, f, indent=2)
    
    # Create visualizations
    plot_statistics(stats, args.output_dir, prefix=prefix)
    
    print(f"\nDataset Statistics:")
    print(f"Total Images: {stats['total_images']}")
    print(f"Total Objects: {stats['total_objects']}")
    print("\nClass Distribution:")
    for class_name, count in stats['class_distribution'].items():
        print(f"  {class_name}: {count}")
    print("\nObjects per Image:")
    print(f"  Mean: {stats['objects_per_image_stats']['mean']:.2f}")
    print(f"  Median: {stats['objects_per_image_stats']['median']:.2f}")
    print(f"  Min: {stats['objects_per_image_stats']['min']}")
    print(f"  Max: {stats['objects_per_image_stats']['max']}")
    
    print(f"\nResults saved to {args.output_dir}")
    print(f"Visualizations and statistics saved to {args.output_dir}")

if __name__ == '__main__':
    main() 