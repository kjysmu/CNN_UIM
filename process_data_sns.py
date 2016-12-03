# -*- coding: utf-8 -*-

import numpy as np
import cPickle
from collections import defaultdict
import sys, re
import pandas as pd

import codecs
from konlpy.corpus import kobill
from konlpy.tag import Twitter; t = Twitter()
from konlpy.utils import pprint

from gensim.models import word2vec

if __name__=="__main__":    
    cdic = {}
    cdicNorm = {}

    fname = "D:/project/python/CNN/LabelResultFB_final.txt"    
    cname = "D:/project/model/CategoryTopicPair.dat"
    
    ct = 0        
    with codecs.open (fname, "rb", "utf-8") as f:
        for line in f:
            tokenList = line.split("\t")
            if len(tokenList) == 7:
                text = tokenList[1]
                cat1 = tokenList[4].replace("/","")
                cat1 = cat1.replace(" ", "")
                cat2 = tokenList[6].replace("/","")
                cat2 = cat2.replace(" ", "")
                cat2 = cat2.replace("\r\n","")
                key = cat1 + " " + cat2
                if key not in cdicNorm:
                    cdicNorm[key] = ct
                    ct += 1

    revs = []
    vocab = defaultdict(float)
    
    pos = lambda d: ['/'.join(p) for p in t.pos(d, norm=True, stem=True)]    
    ct = 0
    uct = -1
    print "loading data...",        
    with codecs.open (fname, "rb", "utf-8") as f:
        for line in f:
            tokenList = line.split("\t")
            if len(tokenList) == 7:
                text = tokenList[1]
                cat1 = tokenList[4].replace("/","")
                cat1 = cat1.replace(" ", "")
                cat2 = tokenList[6].replace("/","")
                cat2 = cat2.replace(" ", "")
                cat2 = cat2.replace("\r\n","")
                key = cat1 + " " + cat2
                text_pos = ""
                words = pos(text)
                if len(words) >= 100:
                    words = words[0:100]
                for word in words:
                    text_pos += word
                    text_pos += '\t'
                    vocab[word] += 1
                datum  = {"y": cdicNorm[key],
                          "text": text,
                          "user_id":uct,
                          "split_user": uct%10,
                          "text_pos" : text_pos.strip(),
                          "num_words": len(words),
                          "split": ct%10}
                ct += 1
                revs.append(datum)
            else:
                if "UserID" in line:
                    uct+= 1
    
    #revs, vocab = build_data_cv(data_folder, cv=10, clean_string=True)
    max_l = np.max(pd.DataFrame(revs)["num_words"])
    
    print "data loaded!"
    print "number of sentences: " + str(len(revs))
    
    print "vocab size: " + str(len(vocab))
    print "max sentence length: " + str(max_l)
    
    print "loading word2vec vectors...",

    model = word2vec.Word2Vec.load('D:/project/model/W2V/ko_word2vec_namu_all.model')
    word_vecs = {}
    for word in vocab:
        if word in model.vocab:
            word_vecs[word] = model[word]
        else:
            word_vecs[word] = np.random.uniform(-0.25,0.25,300).astype(np.float32)
    
    vocab_size = len(word_vecs)
    word_idx_map = dict()

    W = np.zeros(shape=(vocab_size+1, 300), dtype='float32')
    W2 = np.zeros(shape=(vocab_size+1, 300), dtype='float32')
    
    W[0] = np.zeros(300, dtype='float32')
    W2[0] = np.zeros(300, dtype='float32')
    
    i = 1
    
    for word in word_vecs:
        W[i] = word_vecs[word]
        W2[i] = np.random.uniform(-0.25,0.25,300).astype(np.float32)
        word_idx_map[word] = i
        i += 1
    
    cPickle.dump([revs, W, W2, word_idx_map, vocab, cdicNorm], open("D:/project/python/CNN/mr_sns_norm_3.p", "wb"))
    
    print "dataset created!"
    
    
    