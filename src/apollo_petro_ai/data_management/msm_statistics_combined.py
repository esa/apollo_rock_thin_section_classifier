import argparse
import re
from collections import defaultdict
from functools import reduce

from .sample_downloader import (
    download_sample,
    download_sample_from_local,
    process_local_samples,
)
from .utils import load_file

# Responsible for parsing msm file and processing files
# modify to changed processed rules


class MsmStatisticsPdsimage:
    """Specific to msm files created from NASA's pdsimage database in lbl_parser.py"""

    def __init__(self, data):
        self.data = data

    def filter_samples_with_description_only(self, description):
        """Filter for samples to contain only photos which have a description similar to the parameter description.
        If the sample does not contain such photos, drop it from the list
        Args:
            description: which description to filter for

        Returns:
            The samples containing photos which have a description similar to the parameter description
        """

        def filter_sample(sample):
            sample["PHOTOS"] = [
                x
                for x in sample["PHOTOS"]
                if description in x["PHOTO_DESCRIPTION"].lower()
            ]
            if len(sample["PHOTOS"]) == 0:
                return None
            return sample

        return {k: filter_sample(v) for k, v in self.data.items() if filter_sample(v)}

    def generate_tags(self, count_photos=False):
        """Collects all sample descriptions. If count_photos = True, the sample description will be added as many times
        as there are photos of this sample
        Args:
            count_photos: specifying whether to take the amount of photos of the specific sample in mind

        Returns:
            All sample descriptions
        """
        if count_photos:
            return reduce(
                (lambda x, y: x + [y["SAMPLE_DESCRIPTION"]] * len(y["PHOTOS"])),
                self.data.values(),
                [],
            )
        return reduce(
            (lambda x, y: x + y["SAMPLE_DESCRIPTION"]), self.data.values(), []
        )

    def build_and_store_sample_map(self):
        """Based on data from msm file, gain insight by printing some statistics and build a map based on a certain tag.
        In this case, we are interested in figuring out the different grains. A map is built based on whether the sample
        describes something like 'fine-grained', 'medium-grained' et cetera. The tag, for example medium-grained',
        will be used as a key and all samples containing that description will be saved as a value.
        Note: a sample can contain multiple grain descriptions, thus the same sample can be saved under multiple tags
        Returns:
            Does not return, but creates a folder 'grain' with subdirectories based on the different grain tags,
            saving the pictures belonging to a sample that has a tag corresponding to the grain tag of that directory.
        """
        # self.data = filter_samples_with_description_only(self.data, 'magnification')
        # self.data = filter_samples_with_description_only(self.data, 'color')
        print(str(len(self.data)) + " samples with thin sections")
        # Count number of thin section photos
        print(
            str(reduce((lambda x, y: x + len(y["PHOTOS"])), self.data.values(), 0))
            + " thin sections photos"
        )

        filter_tag = "grain"
        # tags_msm = self.generate_tags(count_photos=True)
        # Count tags
        # print(collections.Counter(tags_msm).most_common())

        map_of_samples = defaultdict(list)

        for sample_msm in self.data.values():
            # Use regular expression to find the filter tag
            if re.search(
                rf"(?<=,)*[^,]*{filter_tag}[^,]*(?=,)*",
                sample_msm["SAMPLE_DESCRIPTION"],
            ):
                matches = re.findall(
                    rf"(?<=,)*[^,]*{filter_tag}[^,]*(?=,)*",
                    sample_msm["SAMPLE_DESCRIPTION"],
                )
                # Possible to have multiple matches
                for match in matches:
                    tag = match.strip().lower()  # May have trailing spaces, remove them
                    map_of_samples[tag].append(sample_msm)
            else:
                map_of_samples["rest"].append(sample_msm)

        # downloads all selected samples into folders group from map_of_samples
        if args.stored:
            download_sample_from_local(map_of_samples, args.stored, "grain")
        # TODO: Does not take into account that remote host can close connection
        else:
            download_sample(map_of_samples, "grain")


class MsmCombinedStatistics:
    """Specific to msm files that have been processed using the processing_combined.py file"""

    def __init__(self, data):
        self.data = data

    def filter_samples_with_photos_of_type(self, type_of_photo):
        """Filter for samples to contain only photos which have the photo type type_of_photo
        If as a result of this operation no photos are left for the sample, drop it from the list
        Args:
            type_of_photo: which type of photo to filter for

        Returns:
            The samples containing photos which have the photo type given by the type_of_photo parameter
        """

        def filter_sample(sample):
            sample["photos"] = [
                x for x in sample["photos"] if x["photo_type"] == type_of_photo
            ]
            if len(sample["photos"]) == 0:
                return None
            return sample

        return {k: filter_sample(v) for k, v in self.data.items() if filter_sample(v)}

    def generate_keys(self, key, count_photos=False):
        """Collects all values in the samples from a certain key. E.g. take the key rock_type_category.
        If count_photos = True, the description of the rock type of the sample
        will be added as many times as there are photos of this sample
        Args:
            key: which key in the dictionary to get value from
            count_photos: specifying whether to take the amount of photos of the specific sample in mind

        Returns:
            All descriptions of the samples for a particular key
        """
        if count_photos:
            return reduce(
                (lambda x, y: x + [y[key]] * len(y["photos"])), self.data.values(), []
            )
        return reduce((lambda x, y: x + [y[key]]), self.data.values(), [])

    def get_all_description_tags(self):
        """Collect all descriptions
        Returns:
            All descriptions
        """

        def get_description(value):
            if "description" in value:
                return value["description"].split(", ")
            return []

        return reduce(lambda x, y: x + get_description(y), self.data.values(), [])

    def get_sample_map_for_tag(self, tag, include_other_group=False):
        """Create a map of samples grouped on tag. If include_other_group = True, samples that do not contain
        this particular tag will be saved as 'other'.

        Args:
            tag: Which tag to look for and group on
            include_other_group: whether samples that do not contain tag

        Returns:
            A map of samples grouped on tag
        """
        map_of_samples = defaultdict(list)

        for sample in self.data.values():
            if sample[tag]:
                map_of_samples[sample[tag]].append(sample)
            elif include_other_group:
                map_of_samples["other"].append(sample)
        return map_of_samples

    def build_and_store_sample_map(self):
        """Based on data from msm file, gain insight by printing some statistics and build two maps based on certain tags.
        In this case, we are interested in figuring out different rock types and classification categories.
        One map for rock types will describe what rock type this sample is such as
        'basalt', 'breccia', 'regolith', 'rake' and so on.
        The other map focuses on grain sizes which is either medium or fine-grain.

        Returns:
            Does not return, but creates a folder 'grain' with subdirectories based on different grain sizes,
            saving the pictures belonging to a sample that has a tag corresponding to the grain tag of that directory.
            The same is done for the different rock types, but in the directory 'rock_type'
        """
        self.data = self.filter_samples_with_photos_of_type("Thin Section")
        print(str(len(self.data)) + " samples with thin sections")
        print(
            str(reduce((lambda x, y: x + len(y["photos"])), self.data.values(), 0))
            + " thin sections photos"
        )

        # filter_tag = "grain"
        # tags = self.generate_keys("rock_type_category")
        #
        # print(collections.Counter(tags).most_common())
        #
        # description = self.get_all_description_tags()

        map_of_samples = self.get_sample_map_for_tag("classification_category", True)
        map_of_samples_type = self.get_sample_map_for_tag("breccia_or_basalt", True)

        # downloads all selected samples into folders group from map_of_samples
        if args.stored:
            data_dir = "datasets/"
            process_local_samples(map_of_samples, args.stored, data_dir + "grain")
            process_local_samples(
                map_of_samples_type, args.stored, data_dir + "rock_type"
            )
        # TODO: Calls function that assumes PDS saving method so will not work as intended
        else:
            download_sample(map_of_samples, "grain")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generates statistics based on msm file of sample metadata"
    )
    parser.add_argument("input", type=str, action="store", help="input msm file name")
    parser.add_argument(
        "stored",
        type=str,
        action="store",
        nargs="?",
        default=None,
        help="path to already stored images. names should match NASA database",
    )

    args = parser.parse_args()

    data_msm = load_file(args.input)
    print(str(len(data_msm)) + " samples")
    statistic_type = MsmCombinedStatistics(data_msm)
    statistic_type.build_and_store_sample_map()
