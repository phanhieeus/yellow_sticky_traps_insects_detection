from ultralytics import YOLO
from pathlib import Path
import yaml
import argparse
import shutil

def create_dataset_yaml(base_dir, output_path):
    """
    Tạo file dataset.yaml cho YOLOv8 từ cấu trúc tiled_data đã chia train/test.
    """
    base_dir = Path(base_dir)
    yaml_content = {
        'path': str(base_dir.absolute()),
        'train': 'train/images',
        'val': 'test/images',
        'names': ['WF', 'MR', 'NC']
    }
    with open(output_path, 'w') as f:
        yaml.dump(yaml_content, f, default_flow_style=False)
    print(f"Created dataset.yaml at {output_path}")

def main():
    parser = argparse.ArgumentParser(description='Train YOLOv8 on tiled insect dataset')
    parser.add_argument('--data_dir', type=str, default='data/tiled_data', help='Path to tiled dataset directory')
    parser.add_argument('--output_dir', type=str, default='runs/yolov8', help='Directory to save model and results')
    parser.add_argument('--model_size', type=str, default='m', choices=['n','s','m','l','x'], help='YOLOv8 model size')
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    parser.add_argument('--img_size', type=int, default=864, help='Input image size')
    parser.add_argument('--device', type=str, default='0', help='Device for training (0 for GPU, cpu for CPU)')
    args = parser.parse_args()

    # Tạo file dataset.yaml
    dataset_yaml = Path(args.data_dir) / 'dataset.yaml'
    create_dataset_yaml(args.data_dir, dataset_yaml)
    
    # Khởi tạo model
    model = YOLO(f'yolov8{args.model_size}.pt')
    
    # Train
    results = model.train(
        data=str(dataset_yaml),
        epochs=args.epochs,
        batch=args.batch_size,
        imgsz=args.img_size,
        device=args.device,
        project=args.output_dir,
        name='train',
        exist_ok=True,
        pretrained=True,
        optimizer='Adam',
        verbose=True,
        seed=42,
        deterministic=True
    )
    
    # Lưu best.pt về output_dir
    best_model_path = Path(args.output_dir) / 'train' / 'weights' / 'best.pt'
    if best_model_path.exists():
        shutil.copy2(best_model_path, Path(args.output_dir) / 'best.pt')
        print(f"Best model saved to {Path(args.output_dir) / 'best.pt'}")
    else:
        print("Warning: best.pt not found!")

    # Validate trên test set
    print("\nValidating on test set...")
    results_test = model.val(
        data=str(dataset_yaml),
        batch=args.batch_size,
        imgsz=args.img_size,
        device=args.device,
        project=args.output_dir,
        name='test',
        exist_ok=True
    )
    print("Test validation complete.")
    
if __name__ == '__main__':
    main() 