# Copyright FMR LLC <opensource@fidelity.com>
# SPDX-License-Identifer: Apache-2.0

import re
import nltk
import spacy
import numpy as np
from rule import grammar
from nltk.util import ngrams
from nltk.probability import MLEProbDist
from utility import all_stopwords

# Settings
nlp = spacy.load('en_core_web_lg')

def get_ngrams(phraselist, ngram_=(3,3)):
    
    if type(ngram_) == tuple:        
        ngrams = list(range(min(ngram_), max(ngram_)+1))
    else:
        raise Exception("Supported types for 'ngram_' is: tuple")
    
    phrase_list=[phrase for phrase in phraselist if len(phrase.split()) in ngrams]
    return phrase_list
    

class Ngram_Keyphrase:
    """
    ngram_ :
           integer - if a single ngram length required
           tuple - if a range of ngram length are requried
           list - if specifi length ngrams are requried

    """
    
    def __init__(self,ngram_=(3,3)):
                
        if type(ngram_) == int:
            self.ngram_ = [ngram_]
        elif type(ngram_) == tuple:        
            self.ngram_ = list(range(min(ngram_), max(ngram_)+1))
        elif type(ngram_) == list:
            self.ngram_ = ngram_
        else:
            raise Exception("Supported types for 'ngram_' are: int, tuple and list")
        
    def get_keyphrases(self, tokens): 
        """
        generate_phrases accepts an interable objects of string.
        the funtions returns back an iterable object of ngram phrases    
        """
        if tokens is None or len(tokens)==0:
            raise TypeError(f"Found an empty or None object")
    
        if type(tokens) == str:
            tokens = nltk.word_tokenize(tokens)
        
        phrase_list = []
        if type(tokens) == list:
            self.tokens=tokens
        else:
            raise Exception("Supported type is a list of iterable string objects: stream of tokens in sequence.")
        
        try:
            for each in self.ngram_:
                phrase_list.extend([e for e in ngrams(tokens,each)])
        except:
             raise InputError(f"Could not create ngrams for the input")
        return phrase_list

    
class Grammar_Keyphrase: 
    
    def __init__(self, grammar):
        self.grammar=grammar 
        
        
    def __chunk_sentences(self, document):
        """
        returns chunkized sentences of a text document
        """
        sentences = [s.text for s in nlp(document).sents]
        return sentences

    
    def __get_pos_tag(self):
        """
        returns POS tagged sentences of a text document
        """

        sentences = self.__chunk_sentences((self.doc))
        tokenized_sentences = [nltk.word_tokenize(sent) for sent in sentences]
        pos_sentences = [nltk.pos_tag(sent) for sent in tokenized_sentences if sent]
        return pos_sentences


    def get_keyphrases(self, doc):
        """
        Extracts key-phrases of depending on the grammar provided
        """
        
        if not doc.strip():
            raise Exception("document cannot be empty") 
        self.doc=doc
        
        phrase_list=[]
        sentences = self.__get_pos_tag()

        for sent in sentences:

            cp = nltk.RegexpParser(self.grammar)
            result = cp.parse(sent)
            for subtree in result.subtrees(filter=lambda t: t.label() == 'KEYPHRASES'):
                keyword = ' '.join([i[0] for i in subtree])
                phrase_list.append(keyword)    
        return phrase_list


class Rake_Keyphrase:
    
    def __init__(self, ngram_ = (2,3), custom_stop_words=[]):
        
        if type(ngram_) == int:
            self.ngram_ = [ngram_]
        elif type(ngram_) == tuple:        
            self.ngram_ = list(range(min(ngram_), max(ngram_)+1))
        elif type(ngram_) == list:
            self.ngram_ = ngram_
        else:
            raise Exception("Supported types for 'ngram_' are: int, tuple and list")
            
        #       stop_word_list = get_all_stopwords()
        
        if type(custom_stop_words) != list:
            raise Exception('stop_word_list should be of type list')
            
        self.stopword_list = all_stopwords + custom_stop_words
        
    
    def __split_sentences(self,text):
        """
        Utility function to return a list of sentences.
        @param text The text that must be split in to sentences.
        """
        sentence_delimiters = re.compile(u'[.!?,;:\t\\\\"\\(\\)\\\'\u2019\u2013]|\\s\\-\\s')
        sentences = sentence_delimiters.split(text)
        return sentences
    
                
    def __build_stop_word_regex(self):
        """
        Function to return stopword regex
        @param stop_word_list A list of stop words.
        @return string A string containing stopword regex pattern.
        """
        stop_word_regex_list = []
        for word in self.stopword_list:
            word_regex = r'\b' + word + r'(?![\w-])'  # added look ahead for hyphen
            stop_word_regex_list.append(word_regex)
        stop_word_pattern = re.compile('|'.join(stop_word_regex_list), re.IGNORECASE)
        return stop_word_pattern
    
    
    def __split_phrases(self, phrases, ngram=3):
        """
        Function to split long phrases
        @param phrases A list of input phrases.
        @param ngram An integer specifying the maximum number of grams in a phrase.
        @return list A list of phrases with each phrase having as many or fewer grams than the input ngram.
        """
        final_phrases = []
        for ph in phrases:
            list_of_words = ph.split()
            if len(list_of_words) > ngram:
                i=0
                for p in list_of_words:
                    if i < len(list_of_words)-ngram:
                        final_phrases.append(" ".join(list_of_words[i:i+ngram]))
                    else:
                        final_phrases.append(" ".join(list_of_words[i:]))
                    i = i + ngram
                    if i >= len(list_of_words):
                        break
            else:
                final_phrases.append(ph)
        return final_phrases
    

    def get_keyphrases(self, doc):
        """
        Function to split the sentences based on stop words
        @param sentence_list A list of sentences.
        @param stopword_pattern A string containing stopword regex pattern.
        @return list A list of keyphrases
        """
        
        if not doc.strip():
            raise Exception("document cannot empty") 
        self.doc=doc
        
        phraseList = []
        phrase_list = []
        self.stopword_pattern=self.__build_stop_word_regex()
        sentenceList = self.__split_sentences(self.doc)
        sentence_list=[i.replace('\n','') for i in sentenceList]
        for s in sentence_list:
            tmp = re.sub(self.stopword_pattern, '|', s.strip())
            phrases = tmp.split("|")
            for phrase in phrases:
                phrase = phrase.strip().lower()
                if phrase != "":
                    phraseList.append(phrase)
                    
        phrase_list=self.__split_phrases(phraseList,3)        
        phrase_list=[phrase for phrase in phrase_list if len(phrase.split()) in self.ngram_]
        return phrase_list