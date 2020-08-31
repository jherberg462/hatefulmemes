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

def tokenize(string, tokenizer, padding):
    '''
    calls .texts_to_sequences on a tokenizer using string as input
    
    args:
        string: str, text to transform into a sequence
        tokenizer: keras.preprocessing.text.Tokenizer object
        padding: int, length of output vector. If len of output vector is 
        less than padding, zeros will be added to beginning, if len is greater
        than len of output vector, it will be truncated 
    
    returns: output of tokenizer.texts_to_sequences with string as input
    with a len of padding
    '''
    vector = tokenizer.texts_to_sequences([string])
    return sequence.pad_sequences(vector, maxlen=padding)

def create_tokenizer(input_ds, top_words, preprocess_fn):
    '''
    creates keras.preprocessing.text.Tokenizer object based on
    input dataset, top number of words, and nlp preprocessing
    functions
    
    args:
        input_ds: list of dicts, each dict has the following key:
            'text': str, text that needs to be tokenized
        top_words: int, top number of words to be tokenized
        preprocess_fn: function, the text in the input_ds will be
        passed into this function to train the tokenizer
        (in input_ds text will also not be passed into the preprocess_fn)
    
    returns:keras.preprocessing.text.Tokenizer object
    '''
    word_list = [] #list of texts
    for item in input_ds:
        words = item['text']
        word_list.append(words)
        preprocessed_words = preprocess_fn(words)
        word_list.append(preprocessed_words)
    
    tokenizer = text.Tokenizer(num_words=top_words)
    tokenizer.fit_on_texts(word_list)
    return tokenizer


    