import os
import shutil
import random

def copy_percentage(src_dir, dest_dir, percent=0.6, extensions=('jpg', 'jpeg', 'png', 'gif')):
    """
    Copy a percentage of images from source directory to destination directory
    :param src_dir: Source directory containing images
    :param dest_dir: Destination directory to copy selected images
    :param percent: Percentage of images to copy (0.0-1.0)
    :param extensions: Tuple of valid image extensions
    """
    # Get all image files
    images = [f for f in os.listdir(src_dir) 
             if f.lower().endswith(extensions) and os.path.isfile(os.path.join(src_dir, f))]
    
    if not images:
        print("No images found in source directory")
        return

    # Calculate number of images to select
    num_select = int(len(images) * percent)
    
    # Randomly select without replacement
    selected = random.sample(images, num_select)
    
    # Create destination directory if needed
    os.makedirs(dest_dir, exist_ok=True)
    
    # Copy files
    for file in selected:
        src_path = os.path.join(src_dir, file)
        dest_path = os.path.join(dest_dir, file)
        shutil.copy2(src_path, dest_path)
    
    print(f"Copied {len(selected)}/{len(images)} images to {dest_dir}")

def split_dataset(input_dir, output_dir, test_size=0.2, val_size=0.1, 
                 extensions=('jpg', 'jpeg', 'png', 'gif')):
    """
    Split dataset into train, test, and validation sets
    :param input_dir: Input directory containing class folders with images
    :param output_dir: Output root directory for split datasets
    :param test_size: Proportion for test set (0.0-1.0)
    :param val_size: Proportion of training set to use for validation (0.0-1.0)
    """
    # Create output directories
    splits = ['train', 'test', 'valid']
    for split in splits:
        os.makedirs(os.path.join(output_dir, split), exist_ok=True)

    # Process each class directory
    for class_name in os.listdir(input_dir):
        class_dir = os.path.join(input_dir, class_name)
        
        if not os.path.isdir(class_dir):
            continue
            
        # Get all images for this class
        images = [f for f in os.listdir(class_dir) 
                 if f.lower().endswith(extensions) and os.path.isfile(os.path.join(class_dir, f))]
        
        if not images:
            continue
            
        # Shuffle images
        random.shuffle(images)
        
        # Split into test and train+val
        split_idx = int(len(images) * (1 - test_size))
        train_val = images[:split_idx]
        test = images[split_idx:]
        
        # Split train into train and validation
        val_split_idx = int(len(train_val) * (1 - val_size))
        train = train_val[:val_split_idx]
        val = train_val[val_split_idx:]
        
        # Copy files to respective directories
        for split, files in [('train', train), ('test', test), ('valid', val)]:
            dest_dir = os.path.join(output_dir, split, class_name)
            os.makedirs(dest_dir, exist_ok=True)
            
            for file in files:
                src_path = os.path.join(class_dir, file)
                dest_path = os.path.join(dest_dir, file)
                shutil.copy2(src_path, dest_path)
                
        print(f"Class {class_name}:")
        print(f"  Train: {len(train)} images")
        print(f"  Valid: {len(val)} images")
        print(f"  Test: {len(test)} images")

if __name__ == "__main__":
    # Example usage

    # copy_percentage(
    #     src_dir="/home/sorter3/Arefeh/Kernel-Sorter/dataset/3-class-garbage-goldar-kalakpink/t/colored-goo",
    #     dest_dir="/home/sorter3/Arefeh/Kernel-Sorter/dataset/3-class-garbage-goldar-kalakpink/t/colored-goo_percent",
    #     percent=0.50
    # )
    
    split_dataset(
        input_dir="/home/sorter3/Arefeh/Kernel-Sorter/dataset/3-class-garbage-goldar-kalakpink/t",
        output_dir="/home/sorter3/Arefeh/Kernel-Sorter/dataset/3-class-garbage-goldar-kalakpink/t",
        test_size=0.2,
        val_size=0.1
    )