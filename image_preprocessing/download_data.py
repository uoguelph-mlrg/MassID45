import gdown
import zipfile
import os
import zipfile
import argparse


def download_data(output_dir):
    # Ensure the output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    else:
        return

    # Google Drive file links and their respective output zip names
    drive_files = {
        "https://drive.google.com/file/d/1bhgTt0OTy7MktVIgbFml6N0fYzHHgNMI/view?usp=drive_link": "batch-1.zip",
        "https://drive.google.com/file/d/1iD0uRy0xcnqT9-E4ctjs9vSZnfrpDIYS/view?usp=drive_link": "batch-2.zip",
        "https://drive.google.com/file/d/1WPITJZiR37BNdBb2llpRvG4sgtwL5sKq/view?usp=drive_link": "bulk_batch_1_and_2.zip",
    }

    # Download and extract each file
    for drive_link, zip_name in drive_files.items():
        # Download the file from Google Drive
        zip_path = os.path.join(output_dir, zip_name)
        gdown.download(drive_link, zip_path, quiet=False, fuzzy=True)

        # Extract the zip file to the output directory
        with zipfile.ZipFile(zip_path, "r") as f:
            f.extractall(output_dir)
    
    # Clean up zip fles 
    dir_files = os.listdir(output_dir)
    for item in dir_files:
        if item.endswith(".zip"):
            os.remove(os.path.join(output_dir, item))

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
