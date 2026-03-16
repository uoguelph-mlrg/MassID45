import argparse
import os
import requests
from urllib.parse import urlparse
import zipfile

URLS = [
    "https://zenodo.org/records/17831807/files/bulk_images_edited.zip?download=1",
    #<PLACEHOLDER FOR UPDATED ZENODO LINK TO ANNOTATED_TILES>
]

def download_data(save_path):
    """Downloads a zip file to disk and then extracts it."""
    os.makedirs(save_path, exist_ok=True)
    for url in URLS:
        filename = os.path.basename(urlparse(url).path)
        zip_file_path = os.path.join(save_path, filename)
        
        print(f"Downloading to {zip_file_path}...")
        try:
            response = requests.get(url, stream=True)
            response.raise_for_status() 

            with open(zip_file_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192): 
                    f.write(chunk)
            print("Download complete. Extracting...")

            with zipfile.ZipFile(zip_file_path, 'r') as z:
                z.extractall(save_path)
            print(f"Extraction complete to directory: {save_path}")

        except requests.exceptions.RequestException as e:
            print(f"An error occurred during download: {e}")

    # Clean up zip files 
    dir_files = os.listdir(save_path)
    for item in dir_files:
        if item.endswith(".zip"):
            os.remove(os.path.join(save_path, item))
            

def main(args):
    download_data(args.output_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Download MassID45 image data into the specified dataset path')
    parser.add_argument(
        '--output_dir',
        type=str,
        required=True,
        help='Output directory for MassID45 data'
    )
    args = parser.parse_args()
    main(args)
