# Copyright 2021 FMR LLC <opensource@fidelity.com>
# SPDX-License-Identifer: Apache-2.0


# Number Tag Description
# 1.	CC	Coordinating conjunction
# 2.	CD	Cardinal number
# 3.	DT	Determiner
# 4.	EX	Existential there
# 5.	FW	Foreign word
# 6.	IN	Preposition or subordinating conjunction
# 7.	JJ	Adjective
# 8.	JJR	Adjective, comparative
# 9.	JJS	Adjective, superlative
# 10.	LS	List item marker
# 11.	MD	Modal
# 12.	NN	Noun, singular or mass
# 13.	NNS	Noun, plural
# 14.	NNP	Proper noun, singular
# 15.	NNPS	Proper noun, plural
# 16.	PDT	Predeterminer
# 17.	POS	Possessive ending
# 18.	PRP	Personal pronoun
# 19.	PRP Possessive pronoun
# 20.	RB	Adverb
# 21.	RBR	Adverb, comparative
# 22.	RBS	Adverb, superlative
# 23.	RP	Particle
# 24.	SYM	Symbol
# 25.	TO	to
# 26.	UH	Interjection
# 27.	VB	Verb, base form
# 28.	VBD	Verb, past tense
# 29.	VBG	Verb, gerund or present participle
# 30.	VBN	Verb, past participle
# 31.	VBP	Verb, non-3rd person singular present
# 32.	VBZ	Verb, 3rd person singular present
# 33.	WDT	Wh-determiner
# 34.	WP	Wh-pronoun
# 35.	WP Possessive wh-pronoun
# 36.	WRB	Wh-adverb




# Defining Grammar

grammar = r"""
  KEYPHRASES: 
              
              
              # Two nouns followed by adjective or verb 
              
              {<NN|NNS>+<DT|IN|,|CC|;>?<NN|NNS>+<DT|IN|,|CC|;>?<JJ|JJR|JJS>+}  
              {<NN|NNS>+<DT|IN|,|CC|;>?<NN|NNS>+<DT|IN|,|CC|;>?<VB|VBD|VBG|VBN|VBP|VBZ>+}
         
              
              # A noun then an adjective or verb and then noun
              
              {<NN|NNS>+<DT|IN|,|CC|;>?<JJ|JJS|JJR>+<DT|IN|,|CC|;>?<NN|NNS>+}  
              {<NN|NNS>+<DT|IN|,|CC|;>?<VB|VBD|VBG|VBN|VBP|VBZ>+<DT|IN|,|CC|;>?<NN|NNS>+} 
              
              
              # adjective or verb followed by two nouns
              
              {<JJ|JJR|JJS>+<DT|IN|,|CC|;>?<NN|NNS>+<DT|IN|,|CC|;>?<NN|NNS>+}  
              {<VB|VBD|VBG|VBN|VBP|VBZ>+<DT|IN|,|CC|;>?<NN|NNS>+<DT|IN|,|CC|;>?<NN|NNS>+}
              
              
              # Mix of adjective and Verb with one Noun term
              
              {<NN|NNS>+<DT|IN|,|CC|;>?<VB|VBD|VBG|VBN|VBP|VBZ>+<DT|IN|,|CC|;>?<JJ|JJS|JJR>+} 
             
              
              {<NN|NNS>+<DT|IN|,|CC|;>?<JJ|JJS|JJR>+<DT|IN|,|CC|;>?<VB|VBD|VBG|VBN|VBP|VBZ>+} 
             
              
              {<VB|VBD|VBG|VBN|VBP|VBZ>+<DT|IN|,|CC|;>?<NN|NNS>+<DT|IN|,|CC|;>?<JJ|JJS|JJR>+}
             
              
              {<VB|VBD|VBG|VBN|VBP|VBZ>+<DT|IN|,|CC|;>?<JJ|JJS|JJR>+<DT|IN|,|CC|;>?<NN|NNS>+}
              

              {<JJ|JJS|JJR>+<DT|IN|,|CC|;>?<VB|VBD|VBG|VBN|VBP|VBZ>+<DT|IN|,|CC|;>?<NN|NNS>+}
              
              
              {<JJ|JJS|JJR>+<DT|IN|,|CC|;>?<NN|NNS>+<DT|IN|,|CC|;>?<VB|VBD|VBG|VBN|VBP|VBZ>+}
              
              
              # All three Nouns
              {<NN|NNS>+<DT|IN|,|CC|;>?<NN|NNS>+<DT|IN|,|CC|;>?<NN|NNS>+}
              
              
              
"""
