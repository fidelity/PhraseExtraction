# Copyright FMR LLC <opensource@fidelity.com>
# SPDX-License-Identifer: Apache-2.0

import re
import nltk
import numpy as np
import pandas as pd
from utility import is_number
from textrank import TextRank_WindowSize
from nltk.probability import MLEProbDist
from textrank import TextRank_WordEmbeddings



def sort_phrases(phrases_dict):
    """
    Takes dictionary as input and return sorted dataframe
    """
    df = pd.DataFrame({'Phrases':[i for i,j in phrases_dict.items()], 'Score':[j for i,j in phrases_dict.items()]})
    sorted_df = df.sort_values(['Score'], ascending=False)
    return sorted_df.reset_index().drop(['index'], 1)



class TextRank:
    
    """
    TextRank is a graph based ranking model.
    it uses PageRank algorithm to obtain the score for the candidate phrases.
    """
    def __init__(self,original_text="", method="WordEmbeddings", pos_kept=['NOUN', 'PROPN','VERB','ADJ'], edge_weight = 1.0, token_lookback=3):
        
        self.method = method
        if self.method == 'WordEmbeddings':
            self.Ranker = TextRank_WordEmbeddings()
            
        if self.method == 'WindowSize':
            self.doc = original_text
            self.pos_kept = pos_kept
            self.edge_weight = edge_weight
            self.token_lookback = token_lookback
            self.Ranker = TextRank_WindowSize(self.doc, self.pos_kept, self.edge_weight, self.token_lookback)
            
        if self.method  not in ['WindowSize','WordEmbeddings']:
            raise InputError(f"{method} is not supported, please select between 'WindowSize' or 'WordEmbeddings'")
        
 
    def rank_phrases(self,phrases):
        return sort_phrases(self.Ranker.rank_phrases(phrases))




class FrequencyDistRank: 
                                 
    """
    FrequencyDistRank is ranking method.
    It learns frequency distribution from the original document and evaluates their ranking in rank_phrases module
    """
                                 
    def __init__(self, original_text):
        if type(original_text) == str:
            original_text = nltk.word_tokenize(original_text)
        self.original_text = list(map(lambda x:x.lower(),original_text))
        self.prob_dist = MLEProbDist(nltk.FreqDist(original_text))

    def rank_phrases(self, phrases):
        if phrases== None or len(phrases) ==0:
            raise InputError(f"An empty or None input object found'")
        if type(phrases) == str:
            phrases = nltk.word_tokenize(phrases)
        if type(phrases[0]) ==str:
            phrases = list(map(lambda x: tuple(x.split()),phrases))
        phrases = set(map(lambda x:tuple(" ".join(x).lower().split()),phrases))
        ranking = {}
        for e in phrases:
            score = [np.log2(1+self.prob_dist.prob(f)) for f in e]
            cand = len(score)
            ranking[" ".join(e)] =np.mean(score)
        return sort_phrases(ranking)

    

class RakeRank:
    
    def __init__(self, method='rake'):
        self.method = method
        if method  not in ['degree','rake']:
            raise InputError(f"{method} is not supported, please select between 'degree' or 'rake'")
                      
    
    def rank_phrases(self, phrase_list):
        """
        Function to calculate the rake rank or degree score for each phrase
        @param phrase_list A list of phrases.
        @return Dict A dictionary with keys as phrases and degree scores as values
        """
        
        keyword_candidates = {}
        
        if self.method == 'rake':
            word_score = self.calculate_word_scores(phrase_list,'rake')   
        elif self.method =='degree':
            word_score = self.calculate_word_scores(phrase_list,'degree')
        
        for phrase in phrase_list:
            keyword_candidates.setdefault(phrase, 0)
            word_list = self.separate_words(phrase, 0)
            candidate_score = 0
            for word in word_list:
                candidate_score += word_score[word]
            keyword_candidates[phrase] = candidate_score
        return sort_phrases(keyword_candidates)
           
    
    def separate_words(self, text, min_word_return_size):
        """
        function to return a list of all words that are have a length greater than a specified number of characters.
        @param text The text that must be split in to words.
        @param min_word_return_size The minimum no of characters a word must have to be included.
        """
        splitter = re.compile('[^a-zA-Z0-9_\\+\\-/]')
        words = []
        for single_word in splitter.split(text):
            current_word = single_word.strip().lower()
            #leave numbers in phrase, but don't count as words, since they tend to invalidate scores of their phrases
            if len(current_word) > min_word_return_size and current_word != '' and not is_number(current_word):
                words.append(current_word)
        return words
                                 
    def calculate_word_scores(self, phraseList,method):
        """
        Function to calculate the score for each word in the phrase
        @param phraseList A list of phrases.
        @param method To indicate whether the ranking should be rake based or degree based.
        @return Dict A dictionary with words as keys and scores as values
        """
        word_frequency = {}
        word_degree = {}
        for phrase in phraseList:
            word_list = self.separate_words(phrase, 0)
            word_list_length = len(word_list)
            word_list_degree = word_list_length - 1
            for word in word_list:
                word_frequency.setdefault(word, 0)
                word_frequency[word] += 1
                word_degree.setdefault(word, 0)
                word_degree[word] += word_list_degree  
        for item in word_frequency:
            word_degree[item] = word_degree[item] + word_frequency[item]

        # Calculate Word scores = deg(w)/frew(w)
        word_score = {}
        for item in word_frequency:
            word_score.setdefault(item, 0)
            if method=='rake':
                word_score[item] = word_degree[item] / (word_frequency[item] * 1.0)  #orig.
            else:
                word_score[item] = word_degree[item] 
        
        return word_score