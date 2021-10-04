# Copyright FMR LLC <opensource@fidelity.com>
# SPDX-License-Identifer: Apache-2.0


import os
import re
import nltk
import spacy
import string
import enchant
from nltk.util import ngrams
from nltk.corpus import stopwords
from gensim.parsing.preprocessing import STOPWORDS
from nltk.tokenize import sent_tokenize, word_tokenize

nltk.download('punkt')
nltk.download('stopwords')
nlp = spacy.load('en_core_web_lg')


def get_smart_stopwords(stop_word_file):
    """
    Utility function to load stop words from a file and return as a list of words
    @param stop_word_file Path and file name of a file containing stop words.
    @return list A list of stop words.
    """
    stop_words = []
    try:
        for line in open(stop_word_file):
            if line.strip()[0:1] != "#":
                for word in line.split():  # in case more than one per line
                    stop_words.append(word)
    except IOError:
        print ('File not found!')
    return stop_words


newDict = enchant.DictWithPWL("en_US",'../phraseextraction/morewords.txt')
nltk_stopwords = stopwords.words('english')
spacy_stopwords = list(nlp.Defaults.stop_words)
gensim_stopwords = list(STOPWORDS)
smart_stopwords = get_smart_stopwords('../phraseextraction/SmartStoplist.txt')
all_stopwords = list(set(nltk_stopwords + gensim_stopwords + spacy_stopwords + smart_stopwords))


def filter_nonascii(text):
    '''
    function removes non ascii character
    @text string or list of string
    @return a string filtering our all non ascii characters 
    '''
    if type(text) ==  str:
        return text.encode("ascii", "ignore").decode()
    elif iter(text):
        return list(map(filter_ascii, text))
    else:
        raise TypeError(f"String type object expected, found {type(text)}")

def remove_punct_num(text, remove_num= True, include_char='_-'):
    """
    function returns back a cleaned string after removing numbers, and special characters   
    @text string object
    @remove_num flag, removes number if turnd on
    @include_char retains special characters given here
    @return cleaned string
    """
    try:       
        include_char = include_char.replace('-','\-')
        rpl_srt1 = fr'[^a-zA-Z0-9{include_char}\s]+'
        text =" "+re.sub(rpl_srt1, ' ', text)+" "#remove special characters:
        if remove_num:
            text = " ".join([ x for x in text.split() if not all(list(map(str.isnumeric,re.split(fr'[{include_char}]',x))))])
        return re.sub(r'[\s]+',' ',text )
    except:
        raise TypeError(f"Expected a string like object, found {type(text)}")
        
def remove_non_english(doc, userdict_filename='morewords.txt'):
    """
    function removes non english words
    @doc accepts str like object from which non englsh needs to be removed
    @userdict_filename File name which contains all the non english word that needs to be treated as english word
    @lang language. Default to US English
    @return processed doc    
    """

    #Initiaize variables
    processeddoc = doc
    
    #validate the string is empty
    if not doc.strip():
        return("")
        
    
    #Remove one English words
    processeddoc = " ".join([word for word in doc.split() if newDict.check(word) and (word not in string.punctuation)])
    return processeddoc


def get_named_entities(doc, ent_list=[]):
    """
    function to tag NER labels
    @param doc - text to be assigned the labels
    @param ent_list - optional list of labels
    @return list A list of tuples containing the word and the NER label.
    """
    if not doc.strip():
        raise Exception("doc is empty")    
    if not iter(ent_list):
        raise TypeError(f"Iterable object expected, found {type(ent_list)}")
        
    entities = nlp(doc)    
    #print(entities.ents)
    
    if len(ent_list)==0:
        return [(entity.text,entity.label_) for entity in entities.ents]
    else:
        return [(entity.text,entity.label_) for entity in entities.ents if entity.label_ in ent_list]
    
    

def remove_named_entities(doc, ent_list):
    """
    function to remove named entities
    @param doc - input text
    @param ent_list - A list of labels
    @return String A string that has the named entities removed
    """
    if not doc.strip():
        raise Exception("doc is empty")    
    if not iter(ent_list):
        raise TypeError(f"Iterable object expected, found {type(ent_list)}")
    if len(ent_list) == 0:
        raise IndexError("Empty list found")
        
    out_doc=doc   
    #print(entities.ents)
    
    entities= get_named_entities(doc, ent_list)
    
    for text,label in entities:
        out_doc=out_doc.replace(text,'')
        
    return out_doc



def get_nltk_stopwords():
    """
    Function to return stopwords from nltk
    @return list A list of stop words.
    """
    return stopwords.words('english')


    
def get_gensim_stopwords():
    """
    Function to return stopwords from gensim
    @return list A list of stop words.
    """
    return list(STOPWORDS)



def get_spacy_stopwords():
    """
    Function to return stopwords from spacy
    @return list A list of stop words.
    """
    return list(nlp.Defaults.stop_words)



def get_all_stopwords():
    """
    function gets all stopword lists from NLTK, Gensim, Spacy and SMART stopword list
    """
    
    #NLTK stopwords
    nltk_stopwords = get_nltk_stopwords()
    
    #Gensim stopwords
    gensim_stopwords = get_gensim_stopwords()

    #Spacy stopwords
    spacy_stopwords = get_spacy_stopwords()
    
    #SMART stopword list (Salton,1971)
    smart_stopwords = get_smart_stopwords('SmartStoplist.txt')
    
    #Creating a stopword list with all the above and any custom_stop_word_list
    all_stopwords = nltk_stopwords + gensim_stopwords + spacy_stopwords + smart_stopwords 
    all_stopwords = list(set(all_stopwords))
    return all_stopwords


def remove_stopwords(text, custom_stopword_list=[]):
    """
    function takes text as input and a custom stopword list and removes stopwords from the text using stopword list
    from NLTK, Gensim and Spacy
    @text input text
    @custom_stopword_list list of custom stop words 
    @return text, after removing all the stopwords
    """
    
    stops = all_stopwords + custom_stopword_list
    
    #Returning text without stopwords
    tokens = word_tokenize(text)
    tokens_without_sw = [word for word in tokens if not word in stops]
    return ' '.join(tokens_without_sw)


def is_number(s):
    """
    returns true if token is a number
    """
    try:
        float(s) if '.' in s else int(s)
        return True
    except ValueError:
        return False
    
    
