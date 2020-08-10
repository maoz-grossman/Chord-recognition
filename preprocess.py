import json
import os
import math
from pitch_class_profiling import PitchClassProfiler

DATASET_PATH = "jim2012Chords\\Guitar_Only"
JSON_PATH = "data2.json"

def save_pitch(dataset_path, json_path):
    """Extracts pitch class from music dataset and saves them into a json file along witgh genre labels.
        :param dataset_path (str): Path to dataset
        :param json_path (str): Path to json file used to save pitchs
        :return:
        """

    # dictionary to store mapping, labels, and pitch classes
    data = {
        "mapping": [],
        "labels": [],
        "pitch": []
    }

    # loop through all chord sub-folder
    for i, (dirpath, dirnames, filenames) in enumerate(os.walk(dataset_path)):

        # ensure we're processing a sub-folder level
        if dirpath is not dataset_path:

            # save chord label (i.e., sub-folder name) in the mapping
            semantic_label = dirpath.split("\\")[-1]
            data["mapping"].append(semantic_label)
            print("\nProcessing: {}".format(semantic_label))

            # process all audio files in chord sub-dir
            for f in filenames:

                # load audio file
                file_path = os.path.join(dirpath, f)

                # process all segments of audio file

                ptc=PitchClassProfiler(file_path)
                data["pitch"].append(ptc.get_profile())
                data["labels"].append(i - 1)
                print("{}, segment:{}".format(file_path, 1))

    # save pitch classes to json file
    with open(json_path, "w") as fp:
        json.dump(data, fp, indent=4)


if __name__ == "__main__":
    save_pitch(DATASET_PATH, JSON_PATH)