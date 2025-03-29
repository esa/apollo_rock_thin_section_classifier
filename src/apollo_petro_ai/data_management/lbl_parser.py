#!/usr/bin/python3
#   authors: piotr.knapczyk@me.com and Romeo Haak
#   processing of LBL files into one database in the form a dictionary
#   requires namelist file - list of all links to all images from nasa database

import argparse
import os
from .utils import load_file, write_to_file


def process_line(line):
    """ Helper function, parses one line by splitting on ':' """
    key_value_list = line.split(':')
    key_value_list[0] = key_value_list[0].strip()
    key_value_list[1] = key_value_list[1].strip()
    return key_value_list


def parse_multiline_description(line, file_reader):
    """
    Helps filereader process information that spans multiple lines by concatenating them into one string
    and processing it as such.
    Args:
        line: current line to be parsed
        file_reader: the filestream of the file to be parsed

    Returns:
        key and value of multiline description, e.g. PHOTO_DESCRIPTION, Black and white
    """
    description = line.strip()
    line = next(file_reader)
    while line.startswith('\t'):
        description += ' ' + line.strip()
        line = next(file_reader)
    key, value = process_line(description)
    return key, value


def generate_sample_database(label_dir, img_list):
    """
    Function responsible for creating a dictionary of all the labels in label_dir which should be in the NASA format.
    The content of the file adheres to the following structure:
    ======================
    MISSION                        : X
    STATION                        : X
    LANDMARK                       : X
    SAMPLE_ID                      : X
    SPECIFICS                      : X
    ORIGINAL_WEIGHT                : X
    SUPER_CLASS                    : X
    SUBCLASS                       : X
    SAMPLE_DESCRIPTION             : X
    PHOTO_TYPE                     : X
    PHOTO_DESCRIPTION              : X

    SAMPLE_REFERENCES
    ------------------
    Since the order is always the same, we can use that knowledge to our advantage.

    Args:
        label_dir: Directory where to read the label files from
        img_list: List containing paths to images from NASA database

    Returns:
        A dictionary where the label files are parsed and saved into a dictionary with the sample number as a key,
        and the information about it as well as the path(s) to the picture(s) of that sample specified in img_list.
    """
    assert os.path.exists(label_dir)

    samples = {}
    for label_file in os.listdir(label_dir):
        label_file = label_file.lower()
        if label_file.endswith('.lbl'):
            path = os.path.join(label_dir, label_file)
            with open(path, 'r') as file_stream:
                cur_line = file_stream.readline()
                while not cur_line.strip().startswith('SAMPLE_ID'):
                    cur_line = next(file_stream)
                descriptor, value = process_line(cur_line)
                sample = samples.get(value)
                # If the sample does not exist yet, get information about the sample.
                # Otherwise, information is duplicate, no use to parsing it again so skip to PHOTO_DESCRIPTION
                if sample is None:
                    sample = {descriptor: value, 'PHOTOS': []}
                    while not cur_line.strip().startswith('SUPER_CLASS'):
                        cur_line = next(file_stream)
                    # Parse information of SUPER_CLASS and SUBCLASS
                    for i in range(2):
                        descriptor, value = process_line(cur_line)
                        sample[descriptor] = value
                        cur_line = next(file_stream)
                    # Get SAMPLE_DESCRIPTION which could span multiple lines
                    descriptor, value = parse_multiline_description(cur_line, file_stream)
                    sample[descriptor] = value
                while not cur_line.strip().startswith('PHOTO_DESCRIPTION'):
                    cur_line = next(file_stream)
                photo = {}
                # Get PHOTO_DESCRIPTION which could span multiple lines
                descriptor, value = parse_multiline_description(cur_line, file_stream)
                photo[descriptor] = value
                photo['FILENAME'] = label_file.rstrip('.lbl')
                # Save paths of pictures that this label file describes
                photo['PATHS'] = list(filter(lambda x: photo['FILENAME'] in x.lower(), img_list))
                sample['PHOTOS'].append(photo)

            samples[sample['SAMPLE_ID']] = sample
    return samples


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Processing sample metadata from LBL files into one data store')
    parser.add_argument('directory', action="store",
                        help='directory of LBL files to be parsed')
    parser.add_argument('output', type=str, action="store",
                        help='output file name')
    parser.add_argument('-v', '--verbose', action='store_true', help="outputs processed data")

    args = parser.parse_args()
    all_urls = load_file('full_database_tree')

    images = list(filter(lambda x: not x.endswith('.LBL'), all_urls))

    data = generate_sample_database(args.directory, images)

    write_to_file(data, args.output + ".msm")
