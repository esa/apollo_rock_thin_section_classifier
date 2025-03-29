import pickle

import pandas as pd
from pandas import ExcelWriter
import bs4
from bs4 import BeautifulSoup
import requests

data_directory = '/home/spaceship/users/romeo_haak/thin-slice-classifier-1/datasets/rock_type'/ # FIXME

def extract_info(website_info):
    info_string = ""
    for result in website_info:
        if not isinstance(result, bs4.NavigableString):  # there are weird NavigableStrings inbetween...
            info_string += f"{result.text} "  # append single mineral to string
    return info_string[:-1]  # remove trailing whitespace


def get_mineral_info_from_links(all_links):
    microscope_info = []

    for spec_link in all_links:

        sample_path = spec_link.split('/')[4]  # this yields the specimen path as a result e.g. "15085-19-pigeonite-basalt"
        sample_id = int(sample_path.split('-')[0])  # e.g. 15085
        specimen_page = requests.get(spec_link)
        soup = BeautifulSoup(specimen_page.content, 'html.parser')

        mineral_results = soup.findAll("div", class_="field--items")  # Find divider that holds mineral information
        mineral_string = 'NaN'
        accessory_string = 'NaN'
        # May happen that no rock forming mineral information is present, nor accessory, but accessory is never present
        # if rock forming mineral is not present
        if len(mineral_results) > 0:
            mineral_string = extract_info(mineral_results[0])
            # May not be accessory mineral is present
            if len(mineral_results) > 1:
                accessory_string = extract_info(mineral_results[1])

        sample_desc = sample_path[sample_path.find('-')+1:]
        split = 0
        if sample_desc[0].isdigit():
            split = int(sample_desc.split('-')[0])
            sample_desc = sample_desc[sample_desc.find('-') + 1:]

        microscope_views = soup.findAll("div", class_="launchMicroscope3")
        microscope_viewer = 'NaN'
        for view_link in microscope_views:
            view_info = view_link.find("a")["href"]
            if view_info.find(str(sample_id)) >= 0:
                microscope_viewer = f"http://www.virtualmicroscope.org{view_info}"
        microscope_info.append([sample_id, sample_desc, mineral_string, split, accessory_string, microscope_viewer])

    microscope_info = pd.DataFrame(microscope_info)
    microscope_info.columns = ['SampleID', 'Sample_description', 'Rock-forming_mineral',
                               'Split', 'Accessory_minerals', 'Microscope_URL']

    return microscope_info


def scrape_info_to_excel(): # FIXME: used to not be in a function
    with open(data_directory + 'virtual_microscope/virtual_microscope_links', 'rb') as f:
        links = pickle.load(f)

    scrape_info = get_mineral_info_from_links(links)
    writer = ExcelWriter('VirtualMicroscopeMetadata.xlsx', mode='a')
    scrape_info.to_excel(writer, sheet_name='All Sample Info', index=False)
    writer.save()
