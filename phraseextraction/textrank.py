# Copyright FMR LLC <opensource@fidelity.com>
# SPDX-License-Identifer: Apache-2.0


import re
import time
import json
import math
import spacy
import string
import typing
import pathlib
import graphviz 
import itertools
import numpy as np
import unicodedata
import networkx as nx
from icecream import ic
from dataclasses import dataclass
from spacy.matcher import PhraseMatcher
from spacy.tokens import Doc, Span, Token  
from settings import embedding_type,embedding_dim
from sklearn.metrics.pairwise import cosine_similarity
from collections import Counter, defaultdict, OrderedDict

nlp = spacy.load('en_core_web_lg')

def extract_word_vectors(gloveDimension):
        
        word_embeddings = {}
        
        if gloveDimension == '300':
            gloveFileName = '../glove/glove.6B.300d.txt'
        elif gloveDimension == '200':
            gloveFileName = '../glove/glove.6B.200d.txt'
        elif gloveDimension == '50':
            gloveFileName = '../glove/glove.6B.50d.txt'
        else:
            gloveFileName = '../glove/glove.6B.100d.txt'
            
        with open(gloveFileName, encoding='utf-8') as f:
            for line in f:
                values = line.split()
                word = values[0]
                coefs = np.asarray(values[1:], dtype='float32')
                word_embeddings[word] = coefs
        
        return word_embeddings

word_embeddings = extract_word_vectors(embedding_dim)


def groupby_apply (data,keyfunc,applyfunc) :
    data = sorted(data, key=keyfunc)

    accum: typing.List[typing.Tuple[typing.Any, typing.Any]] = [
        (k, applyfunc(g),)
        for k, g in itertools.groupby(data, keyfunc)
        ]

    return accum

@dataclass(order=True, frozen=True)
class Lemma:
    lemma: str
    pos: str


    def label (
        self
        ) -> str:
        return str((self.lemma, self.pos,))
    
@dataclass
class Phrase:
    text: str
    chunks: typing.List[Span]
    count: int
    rank: float

class TextRank_WindowSize():
    def __init__(self,original_text,pos_kept,edge_weight,token_lookback):
        self.doc: Doc = nlp(original_text)
        self.edge_weight: float = edge_weight
        self.token_lookback: int = token_lookback
        self.pos_kept: typing.List[str] = pos_kept
        
        
        # effectively, performs the same work as the `reset()` method;
        # called explicitly here for the sake of type annotations
        self.elapsed_time: float = 0.0
        self.lemma_graph: nx.DiGraph = nx.DiGraph()
        self.phrases: dict = defaultdict(list)
        self.ranks: typing.Dict[Lemma, float] = {}
        self.seen_lemma: typing.Dict[Lemma, typing.Set[int]] = OrderedDict()
        
    def reset(self):
        self.lemma_graph = nx.DiGraph()
        self.phrases = defaultdict(list)
        self.ranks = {}
        self.seen_lemma = OrderedDict()
        
    def get_personalization(self):
        return None
    
    def rank_phrases(self,phrases):
        self.reset()
        self.lemma_graph = self.construct_graph()
        self.ranks = nx.pagerank(self.lemma_graph, personalization = self.get_personalization(),)
        all_phrases = self.collect_phrases(self.doc, phrases, self.ranks)
        
        raw_phrase_list: typing.List[Phrase] = self.get_min_phrases(all_phrases)
        #phrase_list: typing.List[Phrase] = sorted(raw_phrase_list, key=lambda p: p.rank, reverse=True)
        rankphrases ={v.text:v.rank for v in raw_phrase_list}
        return rankphrases
    
    def construct_graph (self):
        g = nx.DiGraph()

        # add nodes made of Lemma(lemma, pos)
        node_list = [Lemma(token.lemma_, token.pos_) for token in self.doc if self.keep_token(token)]
        #print(node_list)
        g.add_nodes_from(node_list)

        # add edges between nodes that co-occur within a window,
        # weighted by the count
        g.add_edges_from(self.edge_list())
        
        return g
    
    def keep_token(self,token):
        lemma = token.lemma_.lower().strip()

        if token.pos_ not in self.pos_kept:
            return False

        # also track occurrence of this token's lemma, for later use
        key = Lemma(lemma, token.pos_,)

        if key not in self.seen_lemma:
            self.seen_lemma[key] = set([token.i])
        else:
            self.seen_lemma[key].add(token.i)

        return True
    
    def edge_list (self):
        #print("Inside Edge List")
        edges = []

        for sent in self.doc.sents:
            h = [Lemma(token.lemma_, token.pos_) for token in sent if self.keep_token(token)]
            #print("token lookback",self.token_lookback)
            #print("h", h)
            for hop in range(self.token_lookback):
                #print("hop")
                for idx, node in enumerate(h[: -1 - hop]):
                    #print(idx,node)
                    nbor = h[hop + idx + 1]
                    edges.append((node, nbor))

        # include weight on the edge: (2, 3, {'weight': 3.1415})
        weighted_edges = [(*n, {"weight": w * self.edge_weight}) for n, w in Counter(edges).items()]
        #print("weighted_edges",weighted_edges)
        return weighted_edges
    
    def collect_phrases (self,doc,candidatePhrase,ranks):
        
       
        phrase_patterns = [nlp(phrase) for phrase in candidatePhrase]
        matcher = PhraseMatcher(nlp.vocab)
        matcher.add("Test",None,*phrase_patterns)
        matches = matcher(doc)
        #tokens = [ token for match_id,start,end in matches for span in doc[start:end] for token in span]
        
        phrases= {
             doc[start:end]: sum(
                ranks[Lemma(token.lemma_, token.pos_)]
                for token in doc[start:end] if self.keep_token(token)
            )
            for match_id,start,end in matches
        }
        
        return {
            span: self.calc_discounted_normalised_rank(span, sum_rank)
            for span, sum_rank in phrases.items()
        }
    

    def calc_discounted_normalised_rank (self,span,sum_rank):
    
        non_lemma = len([tok for tok in span if tok.pos_ not in self.pos_kept])
        non_lemma_discount = len(span) / (len(span) + (2.0 * non_lemma) + 1.0)

        # use a root mean square (RMS) to normalize the contributions
        # of all the tokens
        phrase_rank = math.sqrt(sum_rank / (len(span) + non_lemma))
        return phrase_rank * non_lemma_discount
    
    def get_min_phrases(self,all_phrases):
        data = [(span.text, rank, span) for span, rank in all_phrases.items()]
       

        keyfunc = lambda x: x[0]
        applyfunc = lambda g: list((rank, spans) for text, rank, spans in g)

        phrases = groupby_apply(data,keyfunc,applyfunc)
      
        phrase_list = [
            Phrase(
                text = p[0],
                rank = max(rank for rank, span in p[1]),
                count = len(p[1]),
                chunks = list(span for rank, span in p[1]),
            )
            for p in phrases
        ]

        return phrase_list


class TextRank_WordEmbeddings():
    
    def __init__(self):
        
        self.word_embeddings = word_embeddings
        self.candidate_phrase_vectors = None
        self.similarity_matrix = None
        self.candidate_scores = None
        
    def rank_phrases(self,candidatephrases):
    
        self.candidate_phrase_vectors = self.generate_candidate_vectors(candidatephrases,self.word_embeddings)
        self.similarity_matrix = self.generate_similarity_martix(candidatephrases,self.candidate_phrase_vectors)
        self.candidate_scores = self.generate_graph(self.similarity_matrix)
        rankphrases = {s:self.candidate_scores[i]  for i,s in enumerate(candidatephrases)}
        return rankphrases    
        
    
    def generate_candidate_vectors(self,candidate_phrases,word_embeddings):
        
        candidate_phrase_vectors = []
        for i in candidate_phrases:
            if len(i) != 0:
                v = sum([word_embeddings.get(w, np.zeros((int(embedding_dim),))) for w in i.split()])/(len(i.split())+0.001)
            else:
                v = np.zeros((int(embedding_dim),))
            candidate_phrase_vectors.append(v)
        
        return(candidate_phrase_vectors) 
    
    def generate_similarity_martix(self, candidatephrases,candidate_phrase_vectors):
        
        cand_vect_array = np.array(candidate_phrase_vectors)
        sim_mat = cosine_similarity(cand_vect_array, cand_vect_array)
        sim_mat = np.multiply(sim_mat, 1- np.eye(cand_vect_array.shape[0])) # remove i=j elements
        return sim_mat
    
    def generate_graph(self,similarity_matrix):
        
        nx_graph = nx.from_numpy_array(similarity_matrix)
        scores = nx.pagerank(nx_graph)
        return scores
         
    
    def get_top_keyphrase(self,NoofKeyPhrases):
               
        #Extract Top 10 Extraction
        ranked_keyphrase = sorted(((self.candidate_scores[i],s) for i,s in enumerate(self.candidate_phrases)), reverse= True)
        topKeypharses =[] 
        for i in range(NoofKeyPhrases):
            topKeypharses.append(ranked_keyphrase[i][1])
        
        #print(topKeypharses)
        return topKeypharses
          