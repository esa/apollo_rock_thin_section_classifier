import argparse
import os
import shutil
from utils import load_file, get_filename
from collections import defaultdict

# TODO: Unfortunate truth is that all pds images are filtered out as they have no image_type field
#  -> check processing_combined.py


def get_photo_map_for_tag(tag, data):
    """
    Collect photos containing tag in their information. Group photos based on different tag information
    Args:
        tag: what tag to look for
        data: data to collect and group photos from

    Returns:
        Grouped photos based on tag
    """
    map_of_samples = defaultdict(list)
    for samples in data.values():
        for sample in samples["photos"]:
            if tag in sample:
                if sample[tag]:
                    map_of_samples[sample[tag]].append(sample)
    return map_of_samples


def collect_pictures(data, src, data_folder):
    """
    Copy pictures from src folder to data_folder according to keys in data
    Args:
        data: dictionary of data
        src: place where pictures are stored
        data_folder: where to save the processed pictures

    Returns:
        Does not return but creates directories according to the structure defined in data
    """
    filelist = os.listdir(src)
    filelist_lower = [name.lower() for name in filelist]

    def callback(filename, target):
        # If picture is in src folder, copy it to the target directory
        if filename.lower() in filelist_lower:
            found_index = filelist_lower.index(filename.lower())
            full_src_path = os.path.join(src, filelist[found_index])
            shutil.copy(full_src_path, target)
            return 0
        return 1

    if os.path.exists(data_folder):
        shutil.rmtree(data_folder)
    os.makedirs(data_folder)

    # Keep track of statistics
    not_found = 0
    photos_processed = 0

    for k, photos in data.items():
        # initialize folder paths and create directory if necessary
        folder_name = k.replace(" ", "_")
        folder_name = os.path.join(data_folder, folder_name)
        if not os.path.exists(folder_name):
            os.makedirs(folder_name)

        for photo in photos:
            photos_processed += 1

            # Add SAMPLE_ID as prefix for easier bookkeeping later
            path = photo["filename"]
            photo_filename = get_filename(path)
            prefix = ""
            if not photo_filename.startswith(photo["sample"]+","):
                prefix = photo["sample"]+","

            destination = os.path.join(folder_name, prefix + photo_filename)
            # Try copying file from src directory to destination, returning 1 if unsuccessful
            file_not_found = callback(photo_filename, destination)
            not_found += file_not_found

    print(f"{not_found} photos not used out of {photos_processed} photos processed")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Download images based on msm file of sample metadata')
    parser.add_argument('input', type=str, action="store",
                        help='input msm file name')
    parser.add_argument('stored', type=str, action="store", nargs="?", default=None,
                        help='path to already stored images. names should match NASA database')

    args = parser.parse_args()

    data_msm = load_file(args.input)
    print(str(len(data_msm)) + " samples")

    samples_map = get_photo_map_for_tag('image_type', data_msm)

    data_dir = "../datasets/"
    collect_pictures(samples_map, args.stored, data_dir + 'image_type')
