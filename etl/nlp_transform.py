import spacy
from tensorflow.python.keras.preprocessing import sequence
from tensorflow.python.keras.preprocessing import text


def transform_to_lemma(doc, remove_stop=False, language='en_core_web_sm'):
    '''
    transforms each word in a text to lemma words
    
    args:
        doc: str, text to be transformed
        
        remove_stop, Bool, default: False, if set to True, stopwords
        will be removed
        
        language: str, default: 'en_core_web_sm', 
    
    returns:
        lemma_text: str, original text converted to lemma words
    '''
    lemma_text = ''
    nlp = spacy.load(language)
    doc = nlp(doc)
    for word in doc:
        if remove_stop:
            if word.is_stop == False:
                lemma_text = '{} {}'.format(lemma_text, word.lemma_)
        else:
            lemma_text = '{} {}'.format(lemma_text, word.lemma_)
    return lemma_text

def remove_stopwords(doc, language='en_core_web_sm'):
    '''
    removes stopwords from a text
    
    args:
        doc: str, text to be transformed
        
        language: str, default: 'en_core_web_sm', 
    
    returns:
        no_stops: str, original text with stopwords removed
    '''
    no_stops = ''
    nlp = spacy.load(language)
    doc = nlp(doc)
    for word in doc:
        if word.is_stop == False:
            no_stops = '{} {}'.format(no_stops, word)
    return no_stops

def tokenize(string, tokenizer, padding, start_pad=None, end_pad=None):
    '''
    calls .texts_to_sequences on a tokenizer using string as input
    
    args:
        string: str, text to transform into a sequence
        tokenizer: keras.preprocessing.text.Tokenizer object
        padding: int, length of output vector. If len of output vector is 
        less than padding, zeros will be added to beginning, if len is greater
        than len of output vector, it will be truncated 
        start_pad: int, default None, if not None, will pad beginning of sequence
            with this value 
        end_pad: int, default None, if not None, will pad end of sequence with 
            this value
            start_pad and end_pad will allow you to add the same word at the 
            beginning and end of each sequence
    
    returns: output of tokenizer.texts_to_sequences with string as input
    with a len of padding
    '''
    vector = tokenizer.texts_to_sequences([string])
    if start_pad:
        vector[0].insert(0, start_pad)
    if end_pad:
        vector[0].append(end_pad)
    return sequence.pad_sequences(vector, maxlen=padding)

def create_tokenizer(input_ds, top_words, preprocess_fn=None, key='text'):
    '''
    creates keras.preprocessing.text.Tokenizer object based on
    input dataset, top number of words, and nlp preprocessing
    functions
    
    args:
        input_ds: list of dicts, each dict has the following key:
            key: str, text that needs to be tokenized, default 'text'
        top_words: int, top number of words to be tokenized
        preprocess_fn: function, default, None, the text in the input_ds will be
        passed into this function to train the tokenizer
        (in input_ds text will also not be passed into the preprocess_fn)
    
    returns:keras.preprocessing.text.Tokenizer object
    '''
    word_list = [] #list of texts
    for item in input_ds:
        words = item[key]
        word_list.append(words)
        if preprocess_fn:
            preprocessed_words = preprocess_fn(words)
            word_list.append(preprocessed_words)
    
    tokenizer = text.Tokenizer(num_words=top_words)
    tokenizer.fit_on_texts(word_list)
    return tokenizer


    