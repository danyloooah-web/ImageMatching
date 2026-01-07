"""
Dataset downloader for image matching training
Downloads and prepares popular datasets
"""

import os
import urllib.request
import zipfile
import tarfile
from tqdm import tqdm


class DownloadProgressBar(tqdm):
    """Progress bar for downloads"""
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


def download_url(url, output_path):
    """Download file with progress bar"""
    with DownloadProgressBar(unit='B', unit_scale=True, miniters=1, desc=url.split('/')[-1]) as t:
        urllib.request.urlretrieve(url, filename=output_path, reporthook=t.update_to)


def extract_archive(archive_path, extract_to):
    """Extract zip or tar archive"""
    print(f"Extracting {archive_path}...")
    
    if archive_path.endswith('.zip'):
        with zipfile.ZipFile(archive_path, 'r') as zip_ref:
            zip_ref.extractall(extract_to)
    elif archive_path.endswith('.tar.gz') or archive_path.endswith('.tgz'):
        with tarfile.open(archive_path, 'r:gz') as tar_ref:
            tar_ref.extractall(extract_to)
    elif archive_path.endswith('.tar'):
        with tarfile.open(archive_path, 'r:') as tar_ref:
            tar_ref.extractall(extract_to)
    
    print(f"Extracted to {extract_to}")


def download_coco(data_dir='datasets'):
    """
    Download COCO dataset (train2017)
    Size: ~18GB
    """
    print("=" * 60)
    print("Downloading COCO Dataset (train2017)")
    print("Size: ~18GB - This will take a while!")
    print("=" * 60)
    
    coco_dir = os.path.join(data_dir, 'coco')
    os.makedirs(coco_dir, exist_ok=True)
    
    url = 'http://images.cocodataset.org/zips/train2017.zip'
    zip_path = os.path.join(coco_dir, 'train2017.zip')
    
    if not os.path.exists(zip_path):
        print("Downloading COCO train2017...")
        download_url(url, zip_path)
    else:
        print("Archive already downloaded")
    
    if not os.path.exists(os.path.join(coco_dir, 'train2017')):
        extract_archive(zip_path, coco_dir)
    else:
        print("Already extracted")
    
    image_dir = os.path.join(coco_dir, 'train2017')
    print(f"\n✓ COCO dataset ready at: {image_dir}")
    return image_dir


def download_hpatches(data_dir='datasets'):
    """
    Download HPatches dataset (specifically for image matching!)
    Size: ~1.4GB
    """
    print("=" * 60)
    print("Downloading HPatches Dataset")
    print("Purpose-built for image matching!")
    print("Size: ~1.4GB")
    print("=" * 60)
    
    hpatches_dir = os.path.join(data_dir, 'hpatches')
    os.makedirs(hpatches_dir, exist_ok=True)
    
    url = 'http://icvl.ee.ic.ac.uk/vbalnt/hpatches/hpatches-sequences-release.tar.gz'
    tar_path = os.path.join(hpatches_dir, 'hpatches-sequences-release.tar.gz')
    
    if not os.path.exists(tar_path):
        print("Downloading HPatches...")
        download_url(url, tar_path)
    else:
        print("Archive already downloaded")
    
    if not os.path.exists(os.path.join(hpatches_dir, 'hpatches-sequences-release')):
        extract_archive(tar_path, hpatches_dir)
    else:
        print("Already extracted")
    
    image_dir = os.path.join(hpatches_dir, 'hpatches-sequences-release')
    print(f"\n✓ HPatches dataset ready at: {image_dir}")
    return image_dir


def download_sun397(data_dir='datasets'):
    """
    Download SUN397 scene dataset
    Size: ~37GB
    """
    print("=" * 60)
    print("Downloading SUN397 Dataset")
    print("Scene understanding dataset")
    print("Size: ~37GB - Large download!")
    print("=" * 60)
    
    sun_dir = os.path.join(data_dir, 'sun397')
    os.makedirs(sun_dir, exist_ok=True)
    
    url = 'http://vision.princeton.edu/projects/2010/SUN/SUN397.tar.gz'
    tar_path = os.path.join(sun_dir, 'SUN397.tar.gz')
    
    if not os.path.exists(tar_path):
        print("Downloading SUN397...")
        download_url(url, tar_path)
    else:
        print("Archive already downloaded")
    
    if not os.path.exists(os.path.join(sun_dir, 'SUN397')):
        extract_archive(tar_path, sun_dir)
    else:
        print("Already extracted")
    
    image_dir = os.path.join(sun_dir, 'SUN397')
    print(f"\n✓ SUN397 dataset ready at: {image_dir}")
    return image_dir


def download_tiny_imagenet(data_dir='datasets'):
    """
    Download Tiny ImageNet (lighter version)
    Size: ~237MB (much smaller than full ImageNet)
    """
    print("=" * 60)
    print("Downloading Tiny ImageNet")
    print("Lightweight ImageNet subset")
    print("Size: ~237MB")
    print("=" * 60)
    
    tiny_dir = os.path.join(data_dir, 'tiny-imagenet')
    os.makedirs(tiny_dir, exist_ok=True)
    
    url = 'http://cs231n.stanford.edu/tiny-imagenet-200.zip'
    zip_path = os.path.join(tiny_dir, 'tiny-imagenet-200.zip')
    
    if not os.path.exists(zip_path):
        print("Downloading Tiny ImageNet...")
        download_url(url, zip_path)
    else:
        print("Archive already downloaded")
    
    if not os.path.exists(os.path.join(tiny_dir, 'tiny-imagenet-200')):
        extract_archive(zip_path, tiny_dir)
    else:
        print("Already extracted")
    
    image_dir = os.path.join(tiny_dir, 'tiny-imagenet-200', 'train')
    print(f"\n✓ Tiny ImageNet dataset ready at: {image_dir}")
    return image_dir


def download_sample_dataset(data_dir='datasets'):
    """
    Download a small sample dataset for quick testing
    Using Unsplash Lite dataset
    """
    print("=" * 60)
    print("Downloading Sample Dataset")
    print("Small dataset for quick testing")
    print("=" * 60)
    
    print("For a quick start, you can use:")
    print("1. Your own images from the cameras")
    print("2. Free images from Unsplash: https://unsplash.com/")
    print("3. Pexels: https://www.pexels.com/")
    print("\nOr run one of the larger dataset downloads above.")
    
    return None


def main():
    """Main download interface"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Download datasets for image matching training')
    parser.add_argument('--dataset', type=str, required=True,
                       choices=['hpatches', 'coco', 'sun397', 'tiny-imagenet', 'all'],
                       help='Dataset to download')
    parser.add_argument('--data_dir', type=str, default='datasets',
                       help='Directory to store datasets (default: datasets)')
    
    args = parser.parse_args()
    
    os.makedirs(args.data_dir, exist_ok=True)
    
    print("\n")
    print("=" * 60)
    print("Image Matching Dataset Downloader")
    print("=" * 60)
    print("\n")
    
    if args.dataset == 'hpatches' or args.dataset == 'all':
        download_hpatches(args.data_dir)
        print("\n")
    
    if args.dataset == 'tiny-imagenet' or args.dataset == 'all':
        download_tiny_imagenet(args.data_dir)
        print("\n")
    
    if args.dataset == 'coco' or args.dataset == 'all':
        download_coco(args.data_dir)
        print("\n")
    
    if args.dataset == 'sun397' or args.dataset == 'all':
        download_sun397(args.data_dir)
        print("\n")
    
    print("\n" + "=" * 60)
    print("Download complete!")
    print("=" * 60)
    print("\nTo train with the downloaded dataset:")
    print(f"  python train_regressor.py --image_dir {args.data_dir}/<dataset_name>")
    print("\nExample:")
    print(f"  python train_regressor.py --image_dir {args.data_dir}/hpatches/hpatches-sequences-release")
    print("\n")


if __name__ == "__main__":
    main()

