import collections
import itertools
import re
import urllib.parse
import urllib.request
from functools import reduce

from bs4 import BeautifulSoup

from .utils import join_dicts, load_file, write_to_file


def combine_pds_and_lunar():
    """Loads and combines pds and lunar dictionaries
    Returns:
        Does not return but creates a file with the combined data
    """
    pds = load_file("pds_data.msm")
    lunar = load_file("lunar_institute_data.msm")
    combined = join_dicts(lunar, pds)
    write_to_file(combined, "combined_data.msm")


def frequency_of_values(dkey, data, count_images=False):
    """For a specific key in the dictionary, check how often a certain assigned value occurs
    Args:
        dkey: key in the dictionary
        data: dictionary containing samples
        count_images: count extra if there are multiple pictures of this sample

    Returns:
        Count of assigned values
    """
    if count_images:
        result: list[float] = reduce(
            (lambda x, y: x + [y[dkey]] * len(y["photos"])), data.values(), []
        )
    else:
        result = reduce((lambda x, y: x + [y[dkey]]), data.values(), [])
    return collections.Counter(result).most_common()


def get_information_from_curator(key, sample_id):
    """Get information from curator website. Look up the sample id and extract the information specified in key.
    For example, when key is Rock Type, it will extract what is saved in the Rock Type field. Have a look at
    https://curator.jsc.nasa.gov/lunar/samplecatalog/sampleinfo.cfm?sample=70017 to see an example of
    the information present.

    Args:
        key: what information to extract
        sample_id: the id of this particular rock sample

    Returns:
        The requested information if it could be downloaded
    """
    link = (
        "https://curator.jsc.nasa.gov/lunar/samplecatalog/sampleinfo.cfm?sample="
        + sample_id
    )
    print("Downloading data: " + sample_id)
    try:
        with urllib.request.urlopen(link) as response:
            html = response.read()
            parsed_html = BeautifulSoup(html, "lxml")
            elem = parsed_html.find("td", text=key)
            elem = elem.nextSibling.nextSibling
            data = str(elem.text)
            print(key, data)
            return data
    except Exception as e:
        print("Cant download data, ignoring ", e)
        return None


def extract_grain(sample):
    """Processes sample and extract grain information if present. Also standardize a bit what is returned
    Args:
        sample:
            the sample to process
    Returns:
        Grain description
    """
    filter_tag = "grain"
    grain_key = "grain_size"
    sample[grain_key] = None

    # Helper function to find 'grain' in the description if present
    def find_desc(description):
        if filter_tag in description:
            for tag in description.split(", "):
                if filter_tag in tag:
                    return tag
        return None

    if "SAMPLE_DESCRIPTION" in sample:
        descriptor = find_desc(sample["SAMPLE_DESCRIPTION"])
        if descriptor:
            return descriptor

    if "description" in sample:
        descriptor = find_desc(sample["description"])
        if descriptor:
            return descriptor

    if "curator_description" in sample:
        desc = sample["curator_description"]
        grains = {"F.G.": "fine-grain", "C.G.": "coarse-grain", "M.G.": "medium-grain"}
        if desc:
            for grain_type in grains.keys():
                if grain_type in desc:
                    return grains.get(grain_type)
            return find_desc(desc)

    return None


def get_rock_category(sample):
    """Divides samples into 4 main categories - basalts, pristine highland rocks, breccias and regolith (soil).
    Classification is saved into "rock_group"
    Args:
        sample: sample to get lithology information from

    Returns:
        the main rock group category this sample belongs to
    """
    lithology = sample["lithology_normalised"]

    if lithology is None:
        return None
    else:
        lowercase_lithology = lithology.lower()
        rock_category = {
            "breccia": "breccia",
            "basalt": "basalt",
            "soil": "regolith",
            "rake": "rake",
            "crustal": "crustal",
            "core": "core",
        }
        for rock_group in rock_category.keys():
            if rock_group in lowercase_lithology:
                return rock_category.get(rock_group)

    print(lithology)

    return None


def get_grain_category(grain_description):
    """Normalize grain category to either medium or fine-grained, otherwise return None
    Args:
        grain_description:
            description to extract grain category from
    Returns:
        medium or fine-grained or None depending on grain description
    """
    if grain_description:
        if re.search(
            "medium-grain|coarse-grain|corse-grain|coarsegrain", grain_description
        ):
            return "medium-grain"
        elif re.search("fine-grain|finegrained|fine grained", grain_description):
            return "fine-grain"
    return None


def get_breccia_or_basalt(data):
    """Get the rock type and return it if it is breccia or basalt
    Args:
        data: current sample

    Returns:
        Rock type if it is breccia or basalt
    """
    rock_type = data["rock_type_category"]
    return rock_type if rock_type == "breccia" or rock_type == "basalt" else None


def change_paths(data):
    """Tries to standardize the way the msm files are saved
    Args:
        data:
            combined msm files
    Returns:
        new dictionary with changed keys and information
    """
    if "photos" not in data:
        data["photos"] = []

    # It may happen that there are duplicate photos. To keep track, create a dictionary with potential duplicates
    duplicates_map = {}
    for i, image in enumerate(data["photos"]):
        duplicates_map[image["photo_number"].lower()] = i
        image["photo_type"] = "Thin Section"

    # Images from the pds database are present, try to remove duplicates and standardize the way they are saved to
    # mimic the lunar samples
    if "PHOTOS" in data:
        for image in data["PHOTOS"]:
            if "PATHS" in image:
                image["other_urls"] = list(set(image["PATHS"]))
                image.pop("PATHS", None)
                # Save one of the images as original url, preferably the full resolution jpeg
                for path in image["other_urls"]:
                    if "FULL_RES_JPEG" in path:
                        image["original_url"] = path
                        # Avoid also having original url in other urls
                        image["other_urls"].remove(path)
                # Convert keys to mimic those in the lunar samples
                image["filename"] = image["FILENAME"] + ".jpg"
                image["photo_number"] = image["FILENAME"]
                image.pop("FILENAME", None)
                image["photo_type"] = "Thin Section"
                image.pop("PHOTO_TYPE", None)
                phot_nr = image["photo_number"]
                # Check if this photo already existed in the lunar samples, if so copy info from that sample
                if phot_nr in duplicates_map:
                    image.pop("filename", None)
                    # Make sure to save the full res jpeg in other urls again to avoid it being dropped
                    image["other_urls"].append(image["original_url"])
                    image.pop("original_url", None)
                    image.pop("photo_number", None)
                    image.pop("photo_type", None)
                    # Since this picture was a duplicate, update it with the info saved in the lunar samples
                    image.update(data["photos"][duplicates_map[phot_nr]])
                    # Duplicate was found, so we can remove it as a potential duplicate candidate
                    duplicates_map.pop(phot_nr)
        # Keep those photos in the lunar samples of which no duplicate was found
        data["photos"] = [
            x
            for x in data["photos"]
            if x["photo_number"].lower() in duplicates_map.keys()
        ]
        # Add pds photos to existing dictionary
        data["photos"] += data["PHOTOS"]
        data.pop("PHOTOS", None)

    # Useful to save SAMPLE_ID like done in pds samples for easy access
    if "SAMPLE_ID" not in data:
        data["SAMPLE_ID"] = data["photos"][0]["sample"]

    lithology_normalised: str = ""
    # Extract rock type information and save as lithology. Ask curator if needed
    if "SUPER_CLASS" in data and data["SUPER_CLASS"] != "":
        lithology_normalised += data["SUPER_CLASS"].lower() + " "
    elif "SAMPLE_ID" in data:
        data["SUPER_CLASS"] = get_information_from_curator(
            "Rock Type", data["SAMPLE_ID"]
        )
        if data["SUPER_CLASS"]:
            lithology_normalised += data["SUPER_CLASS"].lower() + " "
    data.pop("SUPER_CLASS", None)

    if "lithology" in data:
        # Can happen that lithology is the same as the super class
        # in which case adding it twice does not give extra info
        if (
            data["lithology"] != ""
            and data["lithology"] != lithology_normalised.strip()
        ):
            lithology_normalised += data["lithology"].lower()
        data.pop("lithology", None)

    lithology_normalised = lithology_normalised.strip()  # Remove trailing spaces

    # Clean up old info that is now saved into lithology_normalised
    data["lithology_normalised"] = (
        None if lithology_normalised == "" else lithology_normalised
    )
    if "lithology" in data:
        data.pop("lithology", None)
    if "SUPER_CLASS" in data:
        data.pop("SUPER_CLASS", None)

    # All functionality below is to extract extra information and save it in the dictionary for easy access
    data["rock_type_category"] = get_rock_category(data)
    data["breccia_or_basalt"] = get_breccia_or_basalt(data)

    if "curator_description" not in data:
        data["curator_description"] = get_information_from_curator(
            "Description", data["SAMPLE_ID"]
        )

    if "grain_size" not in data or data["grain_size"] is None:
        data["grain_size"] = extract_grain(data)

    data["classification_category"] = get_grain_category(data["grain_size"])
    return data


def print_descriptions(data):
    """Get some info on the descriptions present in the data
    Args:
        data: dictionary potentially containing descriptions

    Returns:
        Prints information
    """
    for k, v in data.items():
        print("Sample: " + k)
        if "SAMPLE_DESCRIPTION" in v:
            print(v["SAMPLE_DESCRIPTION"])
        else:
            print("-- no NASA description --")

        if "description" in v:
            print(v["description"])
        else:
            print("-- no Lunar Institute description --")

        if "curator_description" in v:
            print(v["curator_description"])
        else:
            print("-- no curator description --")
        print("Grain data: " + str(v["grain_size"]))


def clean_lunar(sample):
    """Clean lunar samples as images were accidentally scraped multiple times. If multiple pictures
    were present on one page, only the first one was scraped times the amount of pictures on that html page

    Args:
        sample:
            specific sample for which this occurred
    Returns:
        Corrected list of photos with paths changed to the intended way of saving it
    """
    # For efficiency, group based on link as otherwise we may need to do 244 html requests instead of 58.
    grouped_photos = collections.defaultdict(list)
    for key, value in itertools.groupby(sample["photos"], lambda x: x["original_url"]):
        grouped_photos[key].extend(list(value))

    new_photos = []
    for photos in grouped_photos.values():
        # In case this particular link does have duplicates, correct the paths
        if len(photos) > 1:
            file_name = photos[0]["photo_number"]
            # Construct link to fetch
            main_link = (
                "https://www.lpi.usra.edu/lunar/samples/atlas/thin_section/?mission=Apollo%2014&sample=14321&"
                f"source_id={file_name}"
            )
            with urllib.request.urlopen(main_link) as response:
                html = response.read()
                parsed_html = BeautifulSoup(html, "lxml")
                # Parse and find all Split values in the tables as this is the only piece of information that changes
                all_elem = parsed_html.find_all("th", text="Split")
                for i, photo in enumerate(photos):
                    # Get value associated to this row for the split header
                    elem = all_elem[i]
                    elem = elem.find_next("td")
                    split = str(elem.text)
                    # Save information by replacing split with the corrected value
                    photo["split"] = split
                    photo["filename"] = re.sub(r"(?<=,)\d{4}", split, photo["filename"])
                    url = photo["original_url"]
                    photo["original_url"] = re.sub(r"(?<=,)\d{4}", split, url)
                    new_photos.append(photo)
        # Otherwise, just add it immediately
        else:
            new_photos.append(photos[0])
    return new_photos


if __name__ == "__main__":
    # testing:
    # get file
    data_msm = load_file("combined_data.msm")
    data_msm = {k: change_paths(v) for k, v in data_msm.items()}

    filtered_non_grain = {k: v for k, v in data_msm.items() if v["grain_size"] is None}
    print("described: " + str(len(data_msm) - len(filtered_non_grain)))
    print("rest: " + str(len(filtered_non_grain)))

    print(frequency_of_values("classification_category", data_msm, True))
    print(frequency_of_values("rock_type_category", data_msm))
    print(frequency_of_values("breccia_or_basalt", data_msm))

    write_to_file(data_msm, "combined_data2x.msm")
