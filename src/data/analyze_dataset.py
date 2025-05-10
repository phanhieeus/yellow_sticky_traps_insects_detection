import os
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from collections import defaultdict

# Define consistent colors and ordered classes globally
class_colors = {
    'WF': 'green',    # Whiteflies
    'MR': 'red',      # Macrolophus
    'NC': 'blue'      # Nesidiocoris
}

ordered_classes = ['WF', 'MR', 'NC']

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
        },
        'class_co_occurrence': defaultdict(lambda: defaultdict(int))  # For co-occurrence analysis
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
        classes_in_image = set()  # Track classes in current image
        
        with open(label_file, 'r') as f:
            for line in f:
                objects_in_image += 1
                stats['total_objects'] += 1
                
                # Parse YOLO format: class_id x_center y_center width height
                class_id, x_center, y_center, width, height = map(float, line.strip().split())
                class_name = class_mapping[int(class_id)]
                
                # Update class distribution
                stats['class_distribution'][class_name] += 1
                classes_in_image.add(class_name)
                
                # Store object size in pixels (width * height * image_size)
                # Assuming image size is 864x864 (from tile_images.py)
                image_size = 864
                pixel_width = width * image_size
                pixel_height = height * image_size
                pixel_area = pixel_width * pixel_height
                stats['object_sizes'][class_name].append(pixel_area)
        
        # Update co-occurrence matrix
        for class1 in classes_in_image:
            for class2 in classes_in_image:
                stats['class_co_occurrence'][class1][class2] += 1
        
        stats['objects_per_image'].append(objects_in_image)
    
    # Convert defaultdict to regular dict for JSON serialization
    stats['class_distribution'] = dict(stats['class_distribution'])
    stats['class_co_occurrence'] = {k: dict(v) for k, v in stats['class_co_occurrence'].items()}
    
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
    counts = [stats['class_distribution'].get(cls, 0) for cls in ordered_classes]
    colors = [class_colors[cls] for cls in ordered_classes]
    bars = plt.bar(ordered_classes, counts, color=colors)
    
    # Add value labels on top of each bar
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height):,}',
                ha='center', va='bottom')
    
    plt.title('Class Distribution')
    plt.xlabel('Class')
    plt.ylabel('Number of Objects')
    plt.savefig(os.path.join(output_dir, f'{prefix}class_distribution.png'))
    plt.close()
    
    # 2. Class Distribution Pie Chart
    plt.figure(figsize=(10, 8))
    total_objects = sum(counts)
    percentages = [count/total_objects*100 for count in counts]
    plt.pie(percentages, labels=[f'{cls}\n({pct:.1f}%)' for cls, pct in zip(ordered_classes, percentages)],
            colors=colors, autopct='%1.1f%%', startangle=90)
    plt.title('Class Distribution (Percentage)')
    plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle
    plt.savefig(os.path.join(output_dir, f'{prefix}class_distribution_pie.png'))
    plt.close()
    
    # 3. Objects per Image Histogram
    plt.figure(figsize=(10, 6))
    plt.hist(stats['objects_per_image'], bins=30)
    plt.title('Objects per Image Distribution')
    plt.xlabel('Number of Objects')
    plt.ylabel('Number of Images')
    plt.savefig(os.path.join(output_dir, f'{prefix}objects_per_image.png'))
    plt.close()
    
    # 4. Object Size Distribution Box Plot
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
    
    box = plt.boxplot(size_data, labels=labels, patch_artist=True)
    
    # Set colors for boxes
    for patch, class_name in zip(box['boxes'], labels):
        patch.set_facecolor(class_colors[class_name])
    
    plt.title('Object Size Distribution by Class')
    plt.xlabel('Class')
    plt.ylabel('Area (pixels²)')
    plt.yscale('log')
    plt.savefig(os.path.join(output_dir, f'{prefix}object_sizes.png'))
    plt.close()
    
    # 5. Class Co-occurrence Heatmap
    plt.figure(figsize=(10, 8))
    co_occurrence = np.zeros((len(ordered_classes), len(ordered_classes)))
    for i, class1 in enumerate(ordered_classes):
        for j, class2 in enumerate(ordered_classes):
            co_occurrence[i, j] = stats['class_co_occurrence'].get(class1, {}).get(class2, 0)
    
    # Create heatmap
    sns.heatmap(co_occurrence, 
                annot=True,  # Show numbers in cells
                fmt='g',     # Format as integers
                xticklabels=ordered_classes,
                yticklabels=ordered_classes,
                cmap='YlOrRd')  # Yellow to Orange to Red colormap
    
    plt.title('Class Co-occurrence Matrix')
    plt.xlabel('Class')
    plt.ylabel('Class')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{prefix}class_co_occurrence.png'))
    plt.close()

def get_output_dir(labels_dir):
    """
    Generate output directory name based on the labels directory structure
    Example:
    - Input: data/tiled_data/train/labels
    - Output: dataset_statistics/tiled_data/train
    """
    # Convert to Path object for easier manipulation
    labels_path = Path(labels_dir)
    
    # Find the index of 'data' in the path
    path_parts = labels_path.parts
    try:
        data_index = path_parts.index('data')
    except ValueError:
        # If 'data' is not found, use the entire path
        return 'dataset_statistics'
    
    # Get the path after 'data'
    relative_path = Path(*path_parts[data_index + 1:])
    
    # Remove 'labels' from the end if present
    if relative_path.name == 'labels':
        relative_path = relative_path.parent
    
    # Create the output directory path
    output_dir = Path('dataset_statistics') / relative_path
    
    return str(output_dir)

def main():
    parser = argparse.ArgumentParser(description='Analyze YOLO format dataset statistics')
    parser.add_argument('--labels_dir', type=str, required=True, help='Path to YOLO labels directory')
    args = parser.parse_args()
    
    # Generate output directory based on input path
    output_dir = get_output_dir(args.labels_dir)
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Analyze dataset
    stats = analyze_labels(args.labels_dir)
    
    # Save statistics to JSON
    json_path = os.path.join(output_dir, 'stats.json')
    with open(json_path, 'w') as f:
        json.dump(stats, f, indent=2)
    
    # Create visualizations
    plot_statistics(stats, output_dir)
    
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
    
    print("\nClass Co-occurrence Matrix:")
    for class1 in ordered_classes:
        print(f"\n{class1}:")
        for class2 in ordered_classes:
            count = stats['class_co_occurrence'].get(class1, {}).get(class2, 0)
            print(f"  with {class2}: {count} images")
    
    print(f"\nResults saved to {output_dir}")
    print(f"Visualizations and statistics saved to {output_dir}")

if __name__ == '__main__':
    main() 