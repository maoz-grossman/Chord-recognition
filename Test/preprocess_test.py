import json
import os
import math
from pitch_class_profiling import PitchClassProfiler

DATASET_PATH = "TheData/where"
JSON_PATH = "test.json"

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
        "pitch": [],
        "order":[]
    }

    # loop through all chord sub-folder
    for i, (dirpath, dirnames, filenames) in enumerate(os.walk(dataset_path)):

        # ensure we're processing a sub-folder level
        if dirpath is not dataset_path:

            # save chord label (i.e., sub-folder name) in the mapping
            semantic_label = dirpath.split("/")[-1]
            data["mapping"].append(semantic_label)
            print("\nProcessing: {}".format(semantic_label))

            # process all audio files in chord sub-dir
            for f in filenames:

                # load audio file
                file_path = os.path.join(dirpath, f)
                file_name= file_path.split("/")[-1]
                file_name2=file_name.split(".")[0]
                data["order"].append(file_name2)
                # process all segments of audio file

                ptc=PitchClassProfiler(file_path)
                data["pitch"].append(ptc.get_profile())
                data["labels"].append(i - 1)
                print("{}, segment:{}".format(file_path, 1))
        else:
            print("oups")
    
    #Sorting the dictionary
    n = len(data["order"]) 
    for i in range(n-1): 
    # range(n) also work but outer loop will repeat one time more than needed. 
  
        # Last i elements are already in place 
        for j in range(0, n-i-1): 
  
            # traverse the array from 0 to n-i-1 
            # Swap if the element found is greater 
            # than the next element 
            if int(data["order"][j]) > int(data["order"][j+1]) : 
                data["order"][j],data["order"][j+1]=data["order"][j+1],data["order"][j]
                data["pitch"][j],data["pitch"][j+1]=data["pitch"][j+1],data["pitch"][j]
                data["labels"][j],data["labels"][j+1]=data["labels"][j+1],data["labels"][j]


    # save pitch classes to json file
    with open(json_path, "w") as fp:
        json.dump(data, fp, indent=4)



if __name__ == "__main__":
    save_pitch(DATASET_PATH, JSON_PATH)