from ultralytics import YOLO
from pathlib import Path
import yaml
import shutil

def create_dataset_yaml(data_dir: str, output_path: str):
    """
    Create YAML configuration file for YOLOv8 training.
    
    Args:
        data_dir: Directory containing the dataset
        output_path: Path to save the YAML file
    """
    data_dir = Path(data_dir)
    
    # Create YAML content
    yaml_content = {
        'path': str(data_dir.absolute()),
        'train': 'train/images',
        'val': 'val/images',
        'test': 'test/images',
        'names': {
            0: 'Macrolophus',
            1: 'Nesidiocoris',
            2: 'Whiteflies'
        }
    }
    
    # Save YAML file
    with open(output_path, 'w') as f:
        yaml.dump(yaml_content, f, default_flow_style=False)

def train_model(data_yaml: str,
                output_dir: str,
                model_size: str = 'm',
                epochs: int = 100,
                batch_size: int = 16,
                img_size: int = 864,
                device: str = '0'):
    """
    Train YOLOv8 model on the dataset.
    
    Args:
        data_yaml: Path to dataset YAML file
        output_dir: Directory to save model and results
        model_size: Model size ('n', 's', 'm', 'l', 'x')
        epochs: Number of training epochs
        batch_size: Batch size
        img_size: Input image size
        device: Device to use for training ('0' for GPU, 'cpu' for CPU)
    """
    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize model
    model = YOLO(f'yolov8{model_size}.pt')
    
    # Train model
    results = model.train(
        data=data_yaml,
        epochs=epochs,
        batch=batch_size,
        imgsz=img_size,
        device=device,
        project=str(output_dir),
        name='train',
        exist_ok=True,
        pretrained=True,
        optimizer='Adam',
        verbose=True,
        seed=42,
        deterministic=True
    )
    
    # Save best model
    best_model_path = output_dir / 'train' / 'weights' / 'best.pt'
    if best_model_path.exists():
        shutil.copy2(best_model_path, output_dir / 'best.pt')
    
    return results

def validate_model(model_path: str,
                  data_yaml: str,
                  output_dir: str,
                  batch_size: int = 16,
                  img_size: int = 864,
                  device: str = '0'):
    """
    Validate trained model on test set.
    
    Args:
        model_path: Path to trained model weights
        data_yaml: Path to dataset YAML file
        output_dir: Directory to save validation results
        batch_size: Batch size
        img_size: Input image size
        device: Device to use for validation
    """
    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load model
    model = YOLO(model_path)
    
    # Validate model
    results = model.val(
        data=data_yaml,
        batch=batch_size,
        imgsz=img_size,
        device=device,
        project=str(output_dir),
        name='val',
        exist_ok=True,
        split='test'
    )
    
    return results 