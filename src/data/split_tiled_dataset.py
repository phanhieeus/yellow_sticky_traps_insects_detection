import os
import shutil
import random
from pathlib import Path
from collections import defaultdict

def count_objects_per_tile(label_dir):
    tile_stats = {}
    for label_file in Path(label_dir).glob('*.txt'):
        class_counts = defaultdict(int)
        with open(label_file, 'r') as f:
            for line in f:
                class_id = int(line.strip().split()[0])
                class_counts[class_id] += 1
        tile_stats[label_file.stem] = dict(class_counts)
    return tile_stats

def split_tiles(tile_stats, train_ratio=0.9, seed=42):
    random.seed(seed)
    all_tiles = list(tile_stats.keys())
    random.shuffle(all_tiles)
    # Sort by max class count per tile (prioritize tiles with rare classes)
    all_tiles.sort(key=lambda t: min(tile_stats[t].values()) if tile_stats[t] else 0, reverse=True)
    # Greedy balance
    train, test = [], []
    train_class, test_class = defaultdict(int), defaultdict(int)
    for tile in all_tiles:
        # Assign to set with fewer objects for rarest class in this tile
        tile_classes = tile_stats[tile]
        if not tile_classes:
            continue
        rare_class = min(tile_classes, key=tile_classes.get)
        if train_class[rare_class] <= test_class[rare_class] * (train_ratio/(1-train_ratio)):
            train.append(tile)
            for k, v in tile_classes.items():
                train_class[k] += v
        else:
            test.append(tile)
            for k, v in tile_classes.items():
                test_class[k] += v
    # Adjust to exact ratio if needed
    n_train = int(len(all_tiles) * train_ratio)
    if len(train) > n_train:
        move = train[n_train:]
        test.extend(move)
        train = train[:n_train]
    elif len(train) < n_train:
        move = test[:n_train-len(train)]
        train.extend(move)
        test = test[n_train-len(train):]
    return train, test

def copy_tiles(tiles, src_img_dir, src_label_dir, dst_img_dir, dst_label_dir):
    os.makedirs(dst_img_dir, exist_ok=True)
    os.makedirs(dst_label_dir, exist_ok=True)
    for tile in tiles:
        img_file = f"{tile}.jpg"
        label_file = f"{tile}.txt"
        shutil.copy2(os.path.join(src_img_dir, img_file), os.path.join(dst_img_dir, img_file))
        shutil.copy2(os.path.join(src_label_dir, label_file), os.path.join(dst_label_dir, label_file))

def print_stats(tiles, tile_stats, name):
    class_total = defaultdict(int)
    class_names = {0: 'WF', 1: 'MR', 2: 'NC'}
    for tile in tiles:
        for k, v in tile_stats[tile].items():
            class_total[k] += v
    print(f"\n{name} set: {len(tiles)} tiles")
    for k in sorted(class_total):
        print(f"  {class_names[k]}: {class_total[k]} objects")

def main():
    base_dir = 'data/tiled_data'
    src_img_dir = os.path.join(base_dir, 'train/images')
    src_label_dir = os.path.join(base_dir, 'train/labels')
    out_train_img = os.path.join(base_dir, 'train_split/images')
    out_train_label = os.path.join(base_dir, 'train_split/labels')
    out_test_img = os.path.join(base_dir, 'test/images')
    out_test_label = os.path.join(base_dir, 'test/labels')
    # Clean output dirs
    for d in [out_train_img, out_train_label, out_test_img, out_test_label]:
        if os.path.exists(d):
            shutil.rmtree(d)
    # Count objects per tile
    tile_stats = count_objects_per_tile(src_label_dir)
    # Split
    train_tiles, test_tiles = split_tiles(tile_stats, train_ratio=0.9)
    # Copy
    copy_tiles(train_tiles, src_img_dir, src_label_dir, out_train_img, out_train_label)
    copy_tiles(test_tiles, src_img_dir, src_label_dir, out_test_img, out_test_label)
    # Stats
    print_stats(train_tiles, tile_stats, 'Train')
    print_stats(test_tiles, tile_stats, 'Test')
    print(f"\nOutput structure:")
    print(f"{base_dir}/train_split/images/  # {len(train_tiles)} train tiles")
    print(f"{base_dir}/train_split/labels/  # {len(train_tiles)} train labels")
    print(f"{base_dir}/test/images/         # {len(test_tiles)} test tiles")
    print(f"{base_dir}/test/labels/         # {len(test_tiles)} test labels")

if __name__ == '__main__':
    main() 