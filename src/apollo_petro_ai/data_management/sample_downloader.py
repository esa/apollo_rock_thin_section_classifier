# This file is responsible for getting all links to the images and label files in the nasa database,
# processing data from msm data file, and downloading images from the nasa database

import os
import random
import re
import shutil
import urllib.error
from urllib.request import Request, urlopen

from bs4 import BeautifulSoup

from .utils import get_filename, load_file, write_to_file


class ImageFinder:
    """Authors: Piotr Knapczyk (piotr.knapczyk@me.com) and Romeo Haak
    Parses NASA's lunar Rock Sample Images dataset and saves a file 'full_database_tree' with a list of stored links
    pointing to images and label files that can be downloaded for further use
    """

    def __init__(self):
        self.valid_modes = ["combine", "all", "missed only"]

    def _get_url_of_images(
        self, url, sub_dir=".*", avoid_dir=None, missed_url_file=None
    ):
        """Recursively find all subdirectories from a given root url, but only save
        the links that include the subdirectory specified in sub_dir
        Args:
            url: current root url
            sub_dir: subdirectory to save
            avoid_dir: directories to avoid opening for efficiency
            missed_url_file: filestream to store any urls that could not be opened

        Returns:
            A list containing all links that include a specific subdirectory
        """
        save_urls = True
        if avoid_dir is None:
            avoid_dir = []
        if missed_url_file is None:
            save_urls = False
        url = url.replace(" ", "%20")
        req = Request(url)
        name_list = []
        # Try except block as it may happen that the site does not respond to a request
        try:
            a = urlopen(req).read()
            soup = BeautifulSoup(a, "html.parser")
            x = soup.find_all("a")
            for i in x:
                file_name = i.extract().get_text()
                url_new = url + file_name
                url_new = url_new.replace(" ", "%20")
                # If the found file is a directory, not the ../ directory to avoid getting an infinite loop,
                # call this function again with the found directory as the new root.
                # For efficiency, avoid opening subdirectories that we do not wish to save
                if (
                    file_name[-1] == "/"
                    and file_name[0] != "."
                    and file_name[:-1] not in avoid_dir
                ):
                    sub_urls = self._get_url_of_images(url_new, sub_dir, avoid_dir)
                    name_list.extend(sub_urls)
                # Images of laboratory equipment are not useful for the geological classifier, hence you probably only
                # want to save images that are in the THIN_SECTIONS directories, (avoiding the ../)
                if re.search(rf"{sub_dir}/[^=.]", url_new):
                    name_list.append(url_new)
        except urllib.error.URLError:
            # In case there was an url that could not be opened, save them if filestream is specified
            if save_urls:
                missed_url_file.write(url + "\n")
            else:
                print(f"following url was never loaded: {url}")

        return name_list

    def director(self, mode):
        """Handles main logic of this class, calling the correct functions based on the specified mode
        Args:
            mode: which mode to run

        """
        if mode in self.valid_modes:
            if mode == "combine":
                data = load_file("database_tree")
                data_extra = load_file("database_tree_rest")

                data.extend(data_extra)
                destination_file_name = "full_database_tree"

            elif mode == "missed only":
                data = []
                with open("leftover_urls", "r") as leftover_urls:
                    missed_urls = leftover_urls.read().splitlines()
                    for missed in missed_urls:
                        retrieved = self._get_url_of_images(
                            missed,
                            "THIN_SECTIONS",
                            ["REDUCED_RES_JPEG", "THUMBNAIL_JPEG"],
                        )
                        data.extend(retrieved)
                destination_file_name = "database_tree_rest"

            else:
                with open("leftover_urls", "w") as leftovers:
                    data = self._get_url_of_images(
                        "https://pdsimage2.wr.usgs.gov/Missions/Apollo/"
                        "Lunar_Sample_Photographs/",
                        "THIN_SECTIONS",
                        ["REDUCED_RES_JPEG", "THUMBNAIL_JPEG"],
                        leftovers,
                    )
                destination_file_name = "database_tree"

            write_to_file(data, destination_file_name)

        else:
            print(
                f"Invalid mode: {mode}, enter one of the following: {self.valid_modes}"
            )


# All functions below are used for actually downloading files
# Only works for PDS
def download_sample(data, data_folder="filtered_data"):
    """Call _processing with an url download callback
    Args:
        data: data to process
        data_folder: where to store the retrieved data

    Returns:

    """
    import urllib.request as ur

    callback = ur.urlretrieve
    _processing(data, callback, data_folder, "FULL_RES_JPEG")


# Only works for PDS
def download_sample_from_local(
    data, src, data_folder="filtered_data", download_if_not_found=False
):
    """Copies files from src folder to destination data_folder. In case download_if_not_found = True,
    this function downloads the file if they were not found in the src folder.

    Args:
        data: dictionary containing data
        src: folder to look for existing images
        data_folder: folder where data is going to be stored
        download_if_not_found: whether to download if a file is not found in src folder

    Returns:
        Does not return but creates directories of images according to data
    """
    import shutil

    def callback(path, target):
        for file in os.listdir(src):
            filename = get_filename(target)
            if file == filename:
                print("found")
                full_src_path = os.path.join(src, filename)
                if os.path.isfile(full_src_path):
                    shutil.copy(full_src_path, target)
                    return
        if download_if_not_found:
            import urllib.request as ur

            ur.urlretrieve(path, target)

    _processing(data, callback, data_folder, "FULL_RES_JPEG")


# Once again, only works for PDS
def _processing(data, download_callback, data_folder, photo_path):
    """Constructs and checks necessary paths for downloading or copying to data_folder
    Args:
        data: dictionary containing data to process
        download_callback: supplied function that should be used to download or copy a file
        data_folder: where files are going to be stored
        photo_path: particular path to get urls from

    Returns:
        Does not return but creates data folders according to data
    """
    if not os.path.exists(data_folder):
        os.makedirs(data_folder)
    for k, samples in data.items():
        folder_name = k.replace(" ", "_")
        folder_name = os.path.join(data_folder, folder_name)
        if not os.path.exists(folder_name):
            os.makedirs(folder_name)

        for sample in samples:
            for photo in sample["PHOTOS"]:
                for path in photo["PATHS"]:
                    if photo_path in path:
                        filename = get_filename(path)
                        filename = os.path.join(folder_name, filename)
                        print("Processing: " + path)
                        try:
                            download_callback(path, filename)
                        except urllib.error.ContentTooShortError:
                            continue
                        break


def process_local_samples(data, src, data_folder, random_photo_from_sample=False):
    """Copy pictures from src folder to data_folder according to keys in data
    Args:
        data: dictionary of data
        src: place where pictures are stored
        data_folder: where to save the processed pictures
        random_photo_from_sample: whether to randomly choose a photo from a sample

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
    # Keep track of some statistics
    not_found = 0
    samples_not_used = 0
    samples_processed = 0
    for k, samples in data.items():
        # initialize folder paths and create directory if necessary
        folder_name = k.replace(" ", "_")
        folder_name = os.path.join(data_folder, folder_name)
        if not os.path.exists(folder_name):
            os.makedirs(folder_name)

        for sample in samples:
            samples_processed += 1
            sample_photos_found = 0

            def handle_photo(photo):
                # Add SAMPLE_ID as prefix for easier bookkeeping later
                path = photo["filename"]
                filename = get_filename(path)
                prefix = ""
                if not filename.startswith(sample["SAMPLE_ID"] + ","):
                    prefix = sample["SAMPLE_ID"] + ","

                destination = os.path.join(folder_name, prefix + filename)
                return callback(filename, destination)

            # Take random photo from sample
            if random_photo_from_sample:
                handle_photo(random.choice(sample["photos"]))
            else:
                # Handle photos for this sample and keep track whether no photos were found of this sample
                for photo_sample in sample["photos"]:
                    file_not_found = handle_photo(photo_sample)
                    not_found += file_not_found
                    sample_photos_found += 1 - file_not_found
            if sample_photos_found == 0:
                samples_not_used += 1

    print(
        f"{samples_not_used} samples not used out of {samples_processed} samples processed"
    )


def download_labels(urlfile, label_folder):
    """Download label files from urls specified in url file and place them in label_folder
    Args:
        urlfile: file containing urls
        label_folder: where to place the downloaded label files

    Returns:
        Does not return, but creates label_folder and places downloaded data in it
    """
    if not os.path.exists(label_folder):
        os.makedirs(label_folder)
    urls = load_file(urlfile)
    file_list = os.listdir(label_folder)
    for url in urls:
        # Get only those from the DATA folder as they contain most information and avoid duplicates that way
        if re.search("DATA[A-Za-z0-9/_-]+.LBL", url):
            file_name = get_filename(url)
            # Do not need to do an url fetch if file is already downloaded
            if file_name not in file_list:
                destination = os.path.join(label_folder, file_name)
                # Open url and save it to destination
                with urlopen(url) as response, open(destination, "wb") as out_file:
                    shutil.copyfileobj(response, out_file)
