import os
import librosa
import math
import json
#json path is the path of the json file where
# we will store the MFCCs and labels
# for DL we need lots of data.. since only 100 tracks per genre - so we chop up each track into segments => many input data
DATASET_PATH = "D:\Codes\DL fo Music Practice\genre_dataset\Data\genres_original"
JSON_PATH = "dataset.json"
SAMPLE_RATE = 22050 #customary value for music processing
DURATION = 30 #measured in seconds
SAMPLES_PER_TRACK = SAMPLE_RATE * DURATION
def save_mfcc(dataset_path,json_path,n_mfcc=13,n_fft=2048,hop_length=512,num_segments=5):
    #build a dictionary to store data
    data = {
        "mapping":[],
        "mfcc":[],
        "labels":[]
    } #for mfcc[0] label is 0 that is classical, mfcc[2] is 1 so blue
    num_samples_per_segment = SAMPLES_PER_TRACK // num_segments
    expected_num_mfcc_vectors_per_segment = math.ceil(num_samples_per_segment / hop_length) # 1.2=>2

# loop through all the genres
#dirpath is current folder, dirnames all the names of the subfolders, and filenames all files so that we can recursively got through dataset
    for i, (dirpath,dirnames,filenames) in enumerate(os.walk(dataset_path)):
    
        #ensure that we are not at the root level
        if dirpath != dataset_path:
            #save the semantic label i.e. mappings
            semantic_label = os.path.basename(dirpath)
            data["mapping"].append(semantic_label)
            print("\nProcessing Genre:{}".format(semantic_label))

            #process files for a specific genre
            for f in filenames:
                #load the audio file
                filepath = os.path.join(dirpath,f)
                signal, sr = librosa.load(filepath,sr=SAMPLE_RATE)

                #we need to divide the signal to a bunch of segments
                #process segments extracting mfcc and storing data
                for s in range(num_segments):
                    start_sample = num_samples_per_segment * s # s=0 =>0
                    finish_sample = start_sample + num_samples_per_segment #s = 0 => num_samples_per_segment
                    mfcc = librosa.feature.mfcc(y=signal[start_sample:finish_sample],
                                                sr = sr,
                                                n_mfcc= n_mfcc,
                                                n_fft = n_fft,
                                                hop_length = hop_length)
                    mfcc = mfcc.T
                    #sometimes the audio file may not have expected number of samples
                    #maybe less or more - we don't need to include this in our training data - we ensure they have the same shape
                    #store mfcc for segment if it has the expected length
                    if len(mfcc) == expected_num_mfcc_vectors_per_segment:
                        data["mfcc"].append(mfcc.tolist())
                        data["labels"].append(i-1) #since i=0 is the root directory which we don't want to include
                        print("{},segment:{}".format(filepath,s))
#save data to json file
    with open(json_path,"w") as fp:
        json.dump(data,fp,indent=4) #indent for pretty print
if __name__ == "__main__":
    save_mfcc(DATASET_PATH,JSON_PATH,num_segments=10)
                    

