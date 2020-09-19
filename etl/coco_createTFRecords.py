#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#download 2014 train images and annotations from https://cocodataset.org/#download unzip, 
#and place files in same folder as this notebook/script


# In[ ]:


import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.python.keras.preprocessing import sequence
from tensorflow.python.keras.preprocessing import text
import json
import matplotlib.pyplot as plt
import spacy
from google.cloud import storage
import os

#helper functions
from helper_fn import load_jsonl_file, upload_TFrecord_gcs, calc_idxs, get_imgBytestring_from_filePath
from features import int_feature, text_feature, imageString_feature
from nlp_transform import tokenize, create_tokenizer


# In[ ]:


from google.oauth2 import service_account

credentials = service_account.Credentials.from_service_account_file( #file location of GCS private key
    '/Users/jeremiahherberg/Downloads/hateful-memes-af65c70c1b79.json')

client = storage.Client(project='hateful-memes', credentials=credentials)


# In[ ]:


ds = load_jsonl_file('captions_train2014.json')[0]


# In[ ]:


ds['licenses'] #4, 7


# In[ ]:


def create_model():
    '''
    creates inceptionV3 pretrained model without the last layer
    output will be reshaped to: (batch_size, 64, 2048)
    
    args: None
    
    returns: model as described above
    
    '''
    model = tf.keras.applications.InceptionV3(include_top=False, input_shape=(299, 299, 3))
    inp = tf.keras.layers.Input((299, 299, 3))
    x = model(inp)
    x = tf.keras.layers.Reshape((64, 2048))(x)
    out = x
    mdl = tf.keras.Model(inp, out)
    return mdl


# In[ ]:


def get_image_features(file_name, model):
    '''
    extracts image features from pretrained model
    
    args:
        file_name: str, file name of image
        model: tf.keras pretrained model
    
    returns:
        output of model after passing image through 
    
    '''
    img = open(file_name, 'rb').read()
    img = tf.io.decode_jpeg(img, channels=3)
    img = tf.image.resize_with_pad(img, 299, 299)
    img = tf.expand_dims(img, 0)
    img = img/127.5 #preprocess to change pixel values to -1 -> 1
    img = img - 1
    out = model(img)
    return out


# In[ ]:


def create_TFrecord(image_df, caption_ds, 
                    tokenizer,
                    padding, model, acceptable_license_nums,
                    start_pad, end_pad,
                    start_idx, end_idx,
                    tfr_num, ttl_tfr=20):
    '''
    creates a TFrecord file
    '''
    TFrecord_filepath = 'coco2014_{}_of_{}.tfrecord'.format(tfr_num, ttl_tfr)
    with tf.io.TFRecordWriter(TFrecord_filepath) as writer:
        for caption in range(start_idx, end_idx + 1):
            img_id = caption_ds[caption]['image_id']
            if image_df[image_df['id'] == img_id]['license'].item() in acceptable_license_nums:
                image_file = image_df[image_df['id'] == img_id]['file_name'].item()
                image_file = os.path.join('train2014', image_file)
                image_feature = get_image_features(image_file, model)
                image_raw = get_imgBytestring_from_filePath(image_file)
                TFexample = create_TFexample(caption_ds[caption], tokenizer, padding,
                                             image_feature, image_raw,
                                             start_pad, end_pad)
                writer.write(TFexample.SerializeToString())
    
    return TFrecord_filepath
                
            
    
    


# In[ ]:


def create_TFexample(dict_, tokenizer, padding, image_features, image_raw, start_pad, end_pad):
    '''
    creates a TFexample with the following features:
        image
        text
    
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
        'image': text_feature(image_features),
        'raw_image': imageString_feature(image_raw),
        'text': text_feature(tokenize(dict_['caption'], tokenizer, padding, start_pad, end_pad, 'post')),
        
    }
    example = tf.train.Example(features=tf.train.Features(feature=features))
    return example


# In[ ]:


def main(ds_path, client, bucket, annotation_key='annotations', caption_key='caption', max_words=10000, max_len=49,
        acceptable_licenses=[4, 7], num_splits=7, oov_token='<unknown>'):
    '''
    creates TFrecord files, and uploads to GCS bucket
    '''
    ds = load_jsonl_file(ds_path)[0]
    caption_ds = ds[annotation_key]
    tokenizer = create_tokenizer(caption_ds, max_words, key=caption_key, oov_token=oov_token)
    tokenizer.fit_on_texts('enddd starttt') #words to signal end and beg of sequence
    tokenizer.word_counts['enddd'] = 10000
    tokenizer.word_counts['starttt'] = 10000
    tokenizer.fit_on_texts('enddd starttt')
    json_file_name = 'coco_tokenizer.json'
    tokenizer_json = tokenizer.to_json()
    with open(json_file_name, 'w') as json_file:
        json.dump(tokenizer_json, json_file)
    upload_TFrecord_gcs(json_file_name, client, bucket)
        
    word_idx = json.loads(tokenizer.get_config()['word_index'])
    start_idx_token = word_idx['starttt']
    end_idx_token = word_idx['enddd']
    startEnd_idxs = calc_idxs(caption_ds, num_splits)
    image_df = pd.DataFrame.from_records(ds['images'])
    model = create_model()
    file_num = 1
    for startIdx, endIdx in startEnd_idxs:
        TFrecord_path = create_TFrecord(image_df, caption_ds, tokenizer,
                                        max_len, model,acceptable_licenses,
                                        start_idx_token, end_idx_token,
                                        startIdx, endIdx, 
                                        file_num, num_splits)
        upload_TFrecord_gcs(TFrecord_path, client, bucket)
        file_num +=1
    
    
    


# In[ ]:


main('captions_train2014.json', client, 'jh_coco_2014', )

