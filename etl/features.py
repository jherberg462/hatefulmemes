import tensorflow as tf
def int_feature(int_):
    '''
    creates a feature that is an int to be used in a TFexample
    
    args:
        int_: int, value to be used as the feature
        
    returns: feature that can be used in a TFexample
    '''
    feature = tf.train.Feature(int64_list=tf.train.Int64List(value=[int_]))
    return feature

def text_feature(text):
    '''
    creates a feature from a string of text to be used in a TFexample
    
    args:
        text: str, string to be used as the feature
    
    returns: feature that can be used in a TFexample
    '''
    text_serialized = tf.io.serialize_tensor(text[0]).numpy() #.tolist()
    feature = tf.train.Feature(bytes_list=tf.train.BytesList(value=[text_serialized]))

#     feature = tf.train.Feature(int64_list=tf.train.Int64List(value=text.flatten() ))
    return feature

def imageString_feature(img_string):
    '''
    creates a feature from a bytestring to be used in a TFexample
    
    args:
        img_string: bytestring, image to be used as the feature
        
    returns: feature that can be used in a TFexample
    
    intended to be used to put images into TFrecords, however 
    this can be used for any bytestring
    '''
    feature = tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_string]))
    return feature


    