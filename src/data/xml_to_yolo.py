import os
import xml.etree.ElementTree as ET
from pathlib import Path

def convert_annotation(xml_file, output_dir, class_mapping):
    """
    Convert a single XML annotation file to YOLO format
    """
    tree = ET.parse(xml_file)
    root = tree.getroot()
    
    # Get image dimensions
    size = root.find('size')
    img_width = int(size.find('width').text)
    img_height = int(size.find('height').text)
    
    # Create output filename
    xml_filename = os.path.basename(xml_file)
    txt_filename = os.path.splitext(xml_filename)[0] + '.txt'
    output_path = os.path.join(output_dir, txt_filename)
    
    # Process each object
    with open(output_path, 'w') as f:
        for obj in root.findall('object'):
            class_name = obj.find('name').text
            if class_name not in class_mapping:
                continue
                
            class_id = class_mapping[class_name]
            
            # Get bounding box coordinates
            bbox = obj.find('bndbox')
            xmin = float(bbox.find('xmin').text)
            ymin = float(bbox.find('ymin').text)
            xmax = float(bbox.find('xmax').text)
            ymax = float(bbox.find('ymax').text)
            
            # Convert to YOLO format (normalized)
            x_center = (xmin + xmax) / (2 * img_width)
            y_center = (ymin + ymax) / (2 * img_height)
            width = (xmax - xmin) / img_width
            height = (ymax - ymin) / img_height
            
            # Write to file
            f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")

def main():
    # Define class mapping
    class_mapping = {
        'WF': 0,  # Whiteflies
        'MR': 1,  # Macrolophus
        'NC': 2   # Nesidiocoris
    }
    
    # Define paths
    current_dir = Path(__file__).parent
    project_root = current_dir.parent.parent
    dataset_dir = project_root / 'data' / 'yellow-sticky-traps-dataset-main'
    xml_dir = dataset_dir / 'annotations'
    output_dir = dataset_dir / 'labels'  # Changed to be a subdirectory of the dataset
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Process all XML files
    xml_files = list(xml_dir.glob('*.xml'))
    for xml_file in xml_files:
        print(f"Converting {xml_file.name}...")
        convert_annotation(xml_file, output_dir, class_mapping)
    
    print(f"\nConversion complete! YOLO annotations saved to: {output_dir}")
    print(f"Class mapping:")
    for class_name, class_id in class_mapping.items():
        print(f"  {class_name}: {class_id}")

if __name__ == '__main__':
    main() 