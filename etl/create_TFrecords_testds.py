#!/usr/bin/env python
# coding: utf-8


import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.python.keras.preprocessing import sequence
from tensorflow.python.keras.preprocessing import text
import json
import matplotlib.pyplot as plt
import spacy
from google.cloud import storage

#helper functions
from helper_fn import load_jsonl_file, upload_TFrecord_gcs, get_imgBytestring_from_filePath, calc_idxs
from features import int_feature, text_feature, imageString_feature
from nlp_transform import transform_to_lemma, remove_stopwords, tokenize, create_tokenizer


from google.oauth2 import service_account

credentials = service_account.Credentials.from_service_account_file( #file location of GCS private key
    '/Users/jeremiahherberg/Downloads/hateful-memes-af65c70c1b79.json')

client = storage.Client(project='hateful-memes', credentials=credentials)



dev_ds = load_jsonl_file('dev.jsonl')
train_ds = load_jsonl_file('train.jsonl')



def create_TFexample(dict_, tokenizer, padding):
    '''
    creates a TFexample with the following features:
        image
        label
        id
        text
        text_lemma
        text_lemma_no_stopwords
        text_no_stopwords
    
    args:
        dict_: dictionary with the following keys:
            id: int, id of image
            img: str, file path of image
            label: int, indicator if meme is hateful or not
            text: str, text on meme
        tokenizer: keras.preprocessing.text.Tokenizer object that will be used to preprocess text
        padding: int, length of each text vector. If text length is less, zeros will be added to 
        beginning, and if the text length is greater than padding, it will be truncated
        
    
    returns: TFexample with above features
    '''
    
    features = {
        'image': imageString_feature(get_imgBytestring_from_filePath(dict_['img'])),
#         'label': int_feature(dict_['label']),
        'id': int_feature(dict_['id']),
        'text': text_feature(tokenize(dict_['text'], tokenizer, padding)),
        #add stopwords and lemons
        'text_lemma' : text_feature(tokenize(transform_to_lemma(dict_['text']), tokenizer, padding)),
        'text_lemma_no_stopwords' : text_feature(tokenize(transform_to_lemma(dict_['text'], remove_stop=True),
                                                          tokenizer, padding)),
        'text_no_stopwords' : text_feature( tokenize(remove_stopwords(dict_['text']), tokenizer,
                                                     padding))
        
    }
    example = tf.train.Example(features=tf.train.Features(feature=features))
    return example



def create_TFrecord(meme_list, 
                    start_idx, end_idx,
                    tokenizer, padding,
                    tfr_file_num, ttl_tfr_files=10):
    '''
    creates a TFrecord file
    
    args:
        meme_list: list
    
    
    returns:
        TFrecord_filepath, str, file path of newly created tfrecord file
    '''
    TFrecord_filepath = 'hatefulmemes_{}_of_{}.tfrecord'.format(tfr_file_num,
                                                               ttl_tfr_files)
    with tf.io.TFRecordWriter(TFrecord_filepath) as writer:
        for idx in range(start_idx, end_idx + 1):
            TFexample = create_TFexample(meme_list[idx], tokenizer, padding)
            writer.write(TFexample.SerializeToString())
    
    return TFrecord_filepath
    ###continue working on documentation
    #move to helper_fn once documentation is complete (import tensorflow as tf)



def main(ds_path, client, bucket, num_splits=10, top_words=20000, padding=41, preprocess=transform_to_lemma):
    '''
    creates all TFrecord files
    '''
    ds = load_jsonl_file(ds_path)
    json_file_name = 'tokenizer.json'
    tokenizer_json = load_jsonl_file(json_file_name)
    tokenizer = text.tokenizer_from_json(tokenizer_json[0])
    startEnd_idxs = calc_idxs(ds, num_splits)
    file_num = 1
    for startIdx, endIdx in startEnd_idxs:
        TFrecord_path = create_TFrecord(ds, startIdx, endIdx,
                                        tokenizer, padding,
                                        file_num, num_splits)
        upload_TFrecord_gcs(TFrecord_path, client, bucket)
        file_num +=1
    


main('test_seen.jsonl', client, 'jh_hateful_memes_test', padding=58, num_splits=2)
main('test_unseen.jsonl', client 'jh_hateful_memes_test_unseen', padding=58, num_splits=2)






