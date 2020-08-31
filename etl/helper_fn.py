from google.cloud import storage
import os
import json


def upload_TFrecord_gcs(filepath, client, bucket):
    '''
    function to upload TFrecord filepath to gcs bucket
    
    intended to be used for TFrecord files, but can be used for any filetype
    
    args:
        filepath: str, path of file to be uploaded
        client: gcs google.storage.Client object
        bucket: str, existing gcs bucket to upload file to
    
    returns:
        None 
    '''
    gcs_bucket = client.bucket(bucket)
    blob = gcs_bucket.blob(filepath)
    blob.upload_from_filename(filepath)

def get_imgBytestring_from_filePath(path):
    '''
    converts an image file into a bytestring
    
    args:
        path: str, file path of image file that will be
        converted into bystestring
        
    returns: bytestring of image file
    '''
    return open(path, 'rb').read()

def load_jsonl_file(file_path):
    '''
    loads jsonl file and creates a list of dicts
    
    args:
        file_path: str, path of jsonl file to load
        
    returns: list of dicts in the file located at fle_path
    '''
    with open(file_path) as file:
        json_list = list(file)
    list_of_jsons = []
    for json_line in json_list:
        line = json.loads(json_line)
        list_of_jsons.append(line)
    
    return list_of_jsons

def calc_idxs(meme_list, num_splits=10):
    '''
    calculate start and end index's of a list in order to split up a list
    evenly
    
    args:
        meme_list: list, list that needs to be split up
        num_splits, int, default 10, number of splits the list needs to be 
        split up into
    
    returns:
        idxs: zip of start and end indexes of meme_list that will evenly split up
        meme_list by num_splits
    
    raises:
        ValueError: if length of meme_list is not evenly divisible by num_splits
    '''
    len_ = len(meme_list)
    if len_ % num_splits > 0:
        raise ValueError('meme_list must be evenly divisible by num_splits')
    
    start_idxs = []
    end_idxs = []
    start_idx = 0
    end_idx = len_ / num_splits - 1
    for _ in range(num_splits):
        start_idxs.append(int(start_idx))
        end_idxs.append(int(end_idx))
        start_idx += len_ / num_splits
        end_idx += len_ / num_splits
    
    idxs = zip(start_idxs, end_idxs)
    return idxs

