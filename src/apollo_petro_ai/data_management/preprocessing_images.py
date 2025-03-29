import math
import os
import shutil
from typing import Any, Dict

import cv2
import numpy as np
from PIL import Image
from skimage.transform import resize


# OBJ: create a folder
# INPUT: folder to be created
def folder_create(folder):
    if os.path.exists(folder):
        shutil.rmtree(folder)
    os.makedirs(folder)


# OBJ: convert images from RGB to HSI (hue, saturation, intensity)
# INPUT: img in RGB
# OUTPUT: img in HSI
def RGB_TO_HSI(img):
    with np.errstate(divide="ignore", invalid="ignore"):
        # Load image with 32-bit floats as variable type
        rgb = np.float32(img) / 255

        # Separate color channels
        red = rgb[:, :, 0]
        green = rgb[:, :, 1]
        blue = rgb[:, :, 2]

        # Calculate Intensity
        def calc_intensity(r, b, g):
            return np.divide(b + g + r, 3) * 255

        # Calculate Saturation
        def calc_saturation(r, b, g):
            minimum = np.minimum(np.minimum(r, g), b)
            saturation = 1 - (3 / (r + g + b + 0.001) * minimum) * 255

            return saturation

        # Calculate Hue
        def calc_hue(r, b, g):
            hue = np.copy(r)

            for i in range(0, b.shape[0]):
                for j in range(0, b.shape[1]):
                    numi = 0.5 * ((r[i][j] - g[i][j]) + (r[i][j] - b[i][j]))
                    denom = math.sqrt(
                        (r[i][j] - g[i][j]) ** 2
                        + ((r[i][j] - b[i][j]) * (g[i][j] - b[i][j]))
                    )
                    hue[i][j] = math.acos(numi / (denom + 0.000001))

                    if b[i][j] <= g[i][j]:
                        hue[i][j] = hue[i][j]
                    else:
                        hue[i][j] = ((360 * math.pi) / 180.0) - hue[i][j]
            return hue * 360 / (2 * math.pi)

        # Merge channels into picture and return image
        hsi = cv2.merge(
            (
                calc_hue(red, blue, green),
                calc_saturation(red, blue, green),
                calc_intensity(red, blue, green),
            )
        )
        return hsi


# OBJ: clean directory in input, remove resizing folders and HSI folders
# INPUT: directory to clean
def clean_folder(directory):
    for folder in os.listdir(directory):
        if folder.rfind("resize") != -1:
            shutil.rmtree(directory + "/" + folder)
        else:
            folder_in = directory + "/" + folder
            folder_out = directory + "/" + folder[0 : folder.rfind("_") + 1] + "HSI"
            if folder_in == folder_out:
                shutil.rmtree(folder_out)


# OBJ: create list of dictionaries containing names of all files from input folder
# INPUT: input folder with pictures
# OUTPUT: return list of photos
def create_dict(folder_in):
    list_files = os.listdir(folder_in)
    list_photos = []
    for i in range(len(list_files)):
        photo = {"filename_rgb": list_files[i], "filename_hsi": "HSI_" + list_files[i]}
        list_photos.append(photo)
    return list_photos


# OBJ: convert all files into a list of dictionaries
# INPUT: dictionary and input folder
# OUTPUT: dictionary
def convert_files_to_dict(folder_in, dict_photos):
    dict_photos[(folder_in[folder_in.rfind("/") + 1 :])] = []
    dict_photos[(folder_in[folder_in.rfind("/") + 1 :])] = create_dict(folder_in)
    return dict_photos


# OBJ: resize images and save them in a dictionary
# INPUT: input folder, resizing features
# OUTPUT: dictionary
def resize_and_save_into_dict(
    data_dir, dir_output, img_rows, img_cols, median_filter, histogram_equalization
):
    dict_photos: Dict[str, Any] = {}
    for folder in os.listdir(data_dir):
        print("Processing with " + folder)
        folder_in = data_dir + "/" + folder
        dict_photos = convert_files_to_dict(folder_in, dict_photos)
        folder_resize_rgb = dir_output + "/" + folder[0 : folder.find("_")] + "_resize/"
        folder_resize_hsi = (
            dir_output + "/" + folder[0 : folder.find("_")] + "_HSI_resize/"
        )
        folder_create(folder_resize_rgb)
        folder_create(folder_resize_hsi)
        list_photos = dict_photos[folder]
        for photo in list_photos:
            photo_path_rgb = data_dir + "/" + folder + "/" + photo["filename_rgb"]
            photo_path_rgb_resize = folder_resize_rgb + photo["filename_rgb"]
            photo_path_hsi_resize = folder_resize_hsi + photo["filename_hsi"]

            img = cv2.imread(photo_path_rgb)

            # histogram equalization
            if histogram_equalization:
                img_yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
                # equalize the histogram of the Y channel
                img_yuv[:, :, 0] = cv2.equalizeHist(img_yuv[:, :, 0])
                # convert the YUV image back to RGB format
                img = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)

            # median filter
            if median_filter:
                img = cv2.medianBlur(img, 3)

            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            image_resize_rgb = (
                resize(img, (img_cols, img_rows), anti_aliasing=True) * 255
            )
            image_resize_hsi = RGB_TO_HSI(image_resize_rgb)
            Image.fromarray(image_resize_hsi.astype(np.uint8)).save(
                photo_path_hsi_resize
            )
            Image.fromarray(image_resize_rgb.astype(np.uint8)).save(
                photo_path_rgb_resize
            )

            # keep track of the info for the dictionary
            photo["array_rgb_resize"] = image_resize_rgb / 255
            h_normalized = image_resize_hsi[:, :, 0] * ((2 * math.pi) / 360)
            s_normalized = image_resize_hsi[:, :, 1] / 100
            i_normalized = image_resize_hsi[:, :, 2] / 255
            image_resize_hsi_normalized = np.dstack(
                (h_normalized, s_normalized, i_normalized)
            )
            photo["array_hsi_resize"] = image_resize_hsi_normalized
    return dict_photos


# OBJ: create list of features for each image (vector of 12 features: RGB PPL, HSI PPL, RGB XPL, HSI XPL)
# INPUT: dictionary of photos
# OUTPUT: list of dictionaries with photos and associated features
def create_features_list(dict_photos, feature_all, dir_output):
    key_list = dict_photos.keys()

    list_photos_features = []

    # TODO: can happen that folders folder_non_lin_* are not initialized therefore code in lines 186 and 187 will fail
    if not feature_all:
        # save images non lin comb
        folder_non_lin_rgb = dir_output + "/non_linear_combi_rgb"
        folder_create(folder_non_lin_rgb)

        folder_non_lin_hsi = dir_output + "/non_linear_combi_hsi"
        folder_create(folder_non_lin_hsi)

    index = 0
    for photo_plane in dict_photos[list(key_list)[0]]:
        photo_cross = dict_photos[list(key_list)[1]][index]
        if (
            photo_plane["filename_rgb"][0 : photo_plane["filename_rgb"].rfind("_")]
            != photo_cross["filename_rgb"][0 : photo_cross["filename_rgb"].rfind("_")]
        ):
            print("Houston we have a problem")
        else:
            array_rgb_plane = photo_plane["array_rgb_resize"]
            array_hsi_plane = photo_plane["array_hsi_resize"]
            array_rgb_cross = photo_cross["array_rgb_resize"]
            array_hsi_cross = photo_cross["array_hsi_resize"]
            if feature_all:
                array_features = np.concatenate(
                    (
                        array_rgb_plane,
                        array_hsi_plane,
                        array_rgb_cross,
                        array_hsi_cross,
                    ),
                    axis=2,
                )
            else:
                non_linear_combi_rgb = np.sqrt(
                    np.square(array_rgb_plane) + np.square(array_rgb_cross)
                )
                non_linear_combi_hsi = np.sqrt(
                    np.square(array_hsi_plane) + np.square(array_hsi_cross)
                )
                array_features = np.concatenate(
                    (non_linear_combi_rgb, non_linear_combi_hsi), axis=2
                )
                photo_path_rgb = folder_non_lin_rgb + "/" + photo_plane["filename_rgb"]
                photo_path_hsi = folder_non_lin_hsi + "/" + photo_plane["filename_hsi"]
                Image.fromarray((non_linear_combi_rgb * 255).astype(np.uint8)).save(
                    photo_path_rgb
                )
                Image.fromarray((non_linear_combi_hsi * 255).astype(np.uint8)).save(
                    photo_path_hsi
                )
            photo = {
                "filename_rgb": photo_plane["filename_rgb"],
                "features": array_features,
            }
            list_photos_features.append(photo)
        index += 1
    return list_photos_features


# OBJ: all preprocessing steps (=MAIN)
# INPUT: input folder with images and input folder
# OUTPUT: list of dictionaries with photos and associated features
def pre_processing(
    data_dir, dir_output, feature_all, median_filter, histogram_equalization
):
    img_rows, img_cols = 120, 89
    clean_folder(data_dir)
    dict_photos = resize_and_save_into_dict(
        data_dir, dir_output, img_rows, img_cols, median_filter, histogram_equalization
    )
    list_photos_features = create_features_list(dict_photos, feature_all, dir_output)
    return list_photos_features
