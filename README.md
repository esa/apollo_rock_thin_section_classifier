

# Apollo Rock Thin Section Classifier
This document outlines the functionality of various approaches within the Thin Slice Classifier project.

Authors: Freja Thoresen, Aidan Cowley, Romeo Haak, Jonas Lewe, Clara Moriceau, Piotr Knapczyk, Victoria Engelschion

## Data Sources
- NASA PDS database (https://pdsimage2.wr.usgs.gov/Missions/Apollo/Lunar_Sample_Photographs/)
- Lunar Institute Data (https://www.lpi.usra.edu/lunar/samples/atlas/thin_sections/)
- Virtual Microscope (http://www.virtualmicroscope.org/explore)

## Models
Trained models can be downloaded from wandb.

https://wandb.ai/freja-thoresen/SimCLR
https://wandb.ai/freja-thoresen/Geological%20Binary%20Classifier


## Binary Classifier:
To execute the binary classifier, run the `msm_statistics.py` script using the following command: 

```bash
combined_data2x.msm <image_directory_name>
```

This command will generate a "datasets" folder containing two sub-folders: "grain" and "rock_type". For example, in the "rock_type" sub-folder, images will be classified into either a "breccia" or a "basalt" folder based on the generated classification dictionary. Additionally, sample IDs will be added as prefixes to the existing file names for easier management. If using the binary classifier, ensure to remove the "other" folder.

---
### Machine Learning / AI Components
This section assumes you have completed the Quick Guide in the `Set everything up` section.

#### stratified_group_kfold.py
This script creates data folds, ensuring that images from the same sample are grouped together in either the training or testing sets. It is utilized by `preprocessing_helper.py`.

#### preprocessing_helper.py
This file is responsible for cleaning and organizing folders for training, testing, and validation. It ensures that images from the same sample are stored together. 

#### networks.py
This is the core component of the binary classifier, which is responsible for training and fine-tuning the networks. 

You can adjust the network type and parameters by the following:

```python
network = InceptionResNet(training_directory, validation_directory, test_directory, 
                          epochs=20, finetune_epochs=30, batch_size=32)
```

The available networks include VGG16, VGG19, and InceptionResNet. Additional functionalities include:
- `-c`: Enables cross-validation training
- `-f`: Enables fine-tuning after initial training
- `-x`: Executes experiment 1 to check for repeated false positives
- `-t`: Runs for only 2 epochs for testing purposes
- `-g`: Draws precision-recall curves (not available for cross-validation)
- `-T`: Evaluates model performance on the test set

To train the network for rock type prediction, use:

```bash
python networks.py -f -T ../datasets/rock_type
```

---
## Preprocessing
While not necessary for running the classifier, understanding the following information can be helpful regarding saved and processed files.

#### full_database_tree
Contains links to all high-resolution JPEG and TIF images in the PDS database. This is used in the `download_labels` function within `sample_downloader.py`.

#### pds_data.msm
This file contains the moon sample metadata (msm) for the data from the PDS database.
Essentially, this file contains information about the sample, specifying its superclass, subclass, sample ID, etc.

See https://pdsimage2.wr.usgs.gov/Missions/Apollo/Lunar_Sample_Photographs/A14VIS_0001/DATA/BASALT/FELDSPATHIC/14053/THIN_SECTIONS/S71-23315.LBL
for an example

This file is a product of running lbl_parser.py. This file is used in processing_combined.py and statistic_combined.py
if using the MsmStatisticsPdsimage class.

#### lunar_institute_data.msm
This file serves a similar purpose to `pds_data.msm` but employs a slightly different data structure.

See https://www.lpi.usra.edu/lunar/samples/atlas/thin_section/?mission=Apollo%2011&sample=10058&source_id=JSC04230 for an example of the data saved.

This file is also utilized in `processing_combined.py`.


#### combined_data2x.msm
**IMPORTANT FILE.** This file consolidates data from the lunar sample atlas and NASA's PDS. It also indicates grain size and rock type for the samples.

To create this file:
1. Combine `lunar_institute.msm` and `pds_data.msm` using `combine_pds_and_lunar()` in `processing_combined.py`.
2. Run the following lines of code with the newly created `combined_data.msm` in `processing_combined.py` to generate `combined_data2x.msm`:

```python
data_msm = load_file("combined_data.msm")
data_msm = {k: change_paths(v) for k, v in data_msm.items()}
write_to_file(data_msm, 'combined_data2x.msm')
```

This file is referenced by `msm_statistics_combined.py`.

#### sample_downloader.py
This file is usually the first point of reference for acquiring data. It contains the `ImageFinder` class responsible for compiling links from NASA's PDS image database. The `full_database_tree` is also available on GitHub for immediate use.
### Class: ImageFinder

You can use this class as follows:

```python
image_finder = ImageFinder()
image_finder.director(<mode>)
```

The director method has three modes: 'combine', 'all', and 'missed only'. Depending on your needs, replace `<mode>` with one of these valid options. Running in "all" mode assumes you don't have any data yet and will attempt to scrape all links from NASA's PDS database located at: 
https://pdsimage2.wr.usgs.gov/Missions/Apollo/Lunar_Sample_Photographs/. 

Please note that the site may occasionally close connections to prevent bot behavior. If this occurs while in "full" mode, any links that could not be accessed will be saved to a file named 'leftover_urls'.

If you then run the director in "missed only" mode, it will retry accessing the missed URLs and attempt to scrape those links and their subdirectories. After calling "full" and "missed only" once, you should have all the links you need. Finally, you can run the director in "combine" mode to merge the results from the "database_tree" file produced by the "full" mode and "database_tree_rest" from the "missed only" mode into a single file called "full_database_tree".

Once you've created the "full_database_tree" file, you can use it to download label files and images. It is recommended to download the label files first, as they are essential for preprocessing. To download the labels, use:

```python
all_urls = load_file('full_database_tree')
download_labels(all_urls, os.path.join('Data', 'labels'))
```

Be aware that the same issue of the remote host closing the connection might occur here as well, so you may need to run this function multiple times. Don't worry; it will inform you when everything has succeeded. 

The other functions in this module are mainly used by different files but serve to fetch or download the actual images of a sample. The most critical function among them is `process_local_samples`.

### lbl_parser.py

This file is responsible for parsing the label files from the PDS database. You can run the script using:

```shell
python lbl_parser.py <Directory where label files are stored> <Desired filename>
```

If everything is successful, you should receive an output file named `pds_data.msm`.

### processing_combined.py

This is an important file that assists in cleaning up and combining the two MSM files obtained from the PDS and LPI databases. It standardizes the information saved from both files into one. Call the necessary functions as needed.

### msm_statistics_combined.py

This file is tasked with extracting specific statistics and information. Depending on which MSM file you are using, you'll need to select the appropriate class.

For example, if you're using `combined_data2x.msm`, the relevant class is `MSMCombinedStatistics`. In that case, you can initialize it like so on line 213:

```python
statistic_type = MsmCombinedStatistics(data_msm)
```
