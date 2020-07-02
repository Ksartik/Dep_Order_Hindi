## -*- coding: utf-8 -*-
from io import open
from isc_parser import Parser
from isc_tagger import Tagger
from isc_tokenizer import Tokenizer
import re
import pandas as pd
import sys
import numpy as np
import csv
import os

def parseSentence(df, nodeind, sent_id):
    # Based on Sharma et al. 2019 
    # Different verb classes according to dependency relation in CPG scheme
    karaka2class = {frozenset(): 'v0', frozenset({'k4'}): 'v1', frozenset({'k1s'}): 'v2', 
                    frozenset({'k1'}): 'v3', frozenset({'k2'}): 'v4', frozenset({'k4', 'k1s'}): 'v5', 
                    frozenset({'k4', 'k1'}): 'v6', frozenset({'k4', 'k2'}): 'v7', frozenset({'k1s', 'k1'}): 'v8', 
                    frozenset({'k1s', 'k2'}): 'v9', frozenset({'k1', 'k2'}): 'v10', frozenset({'k4', 'k1', 'k1s'}): 'v11', 
                    frozenset({'k4', 'k2', 'k1s'}): 'v12', frozenset({'k4', 'k1', 'k2'}): 'v13', frozenset({'k1s', 'k1', 'k2'}): 'v14', 
                    frozenset({'k2', 'k4', 'k1', 'k1s'}): 'v15'}
    # Head:Tail , Tails Deprel , Sentential Distance : HeadIndex - CurrentIndex
    df["TargetLemma"] = df["LEMMA"]
    df["TargetNodeIndex"] = list(map(lambda x: x + nodeind, df.index))
    df["SententialDistance"] = df["HEAD"] - df["INDEX"]
    df["SourceNodeIndex"] = df["TargetNodeIndex"] + df["SententialDistance"]
    for i in df.index:
        if (df.iloc[i]["UPOS"] in ["VM", "VAUX"]):
            df_v = df.loc[df["SourceNodeIndex"] == df.iloc[i]["TargetNodeIndex"]]
            df.iloc[i, df.columns.get_loc("UPOS")] += ":" + karaka2class[frozenset(set(df_v["DEPREL"]).intersection({'k1', 'k2', 'k1s', 'k4'}))]
    df["TargetUPOS"] = df["UPOS"]
    df["tni"] = df.index
    df["sni"] = df["tni"] + df["SententialDistance"]
    for i in df.index:
        try:
            df.at[i, "SourceLemma"] = df.at[df.at[i, "sni"], "LEMMA"]
            df.at[i, "SourceUPOS"] = df.at[df.at[i, "sni"], "UPOS"]
        except:
            pass
    df = df.loc[df["HEAD"] != 0]
    df = df.loc[df["UPOS"] != "SYM"]
    df["sent_id"] = sent_id
    df = df[["sent_id", "SourceLemma", "SourceUPOS", "TargetLemma", "TargetUPOS", "DEPREL", 
             "SententialDistance", "SourceNodeIndex", "TargetNodeIndex"]]
    return df

def parse_line (dict1, line):
    description = line.split()
    index = int(description[0])
    if ((index >= 1) and (description[2] != "_") and (len(description) == 10)):
        dict1['INDEX'].append(int(description[0]))
        dict1['FORM'].append(description[1])
        dict1['LEMMA'].append(description[2])
        dict1['UPOS'].append(description[3])
        dict1['XPOS'].append(description[4])
        dict1['FEATS'].append(description[5])
        dict1['HEAD'].append(int(description[6]))
        dict1['DEPREL'].append(description[7])
        dict1['DEPS'].append(description[8])
        dict1['MISC'].append(description[9])
    return dict1

tk = Tokenizer(lang='hin', split_sen=True)
tagger = Tagger(lang='hin')
parser = Parser(lang='hin')

cols = ['INDEX', 'FORM', 'LEMMA', 'UPOS', 'XPOS', 
        'FEATS', 'HEAD', 'DEPREL', 'DEPS', 'MISC']

edgeCols = ["sent_id", "SourceLemma", "SourceUPOS", "TargetLemma", "TargetUPOS", "DEPREL", 
            "SententialDistance", "SourceNodeIndex", "TargetNodeIndex"]

max_sentences = int(sys.argv[1]) if (len(sys.argv) > 1) else 10000000
node_index = 0
prev_node_index = 0
sent_id = 1
# Hindi data taken from IITB - Kunchukuttan et al. 2017
with open("hindi-data/IITB/monolingual.hi", "r", encoding= "utf-8") as f, open("edges_IITB_parsed.csv", "w+", encoding="utf-8", newline='') as dataEdgeF:
    edge_writer = csv.DictWriter(dataEdgeF, fieldnames=edgeCols)
    edge_writer.writeheader()
    count_sentences = 0
    for text in f :
        if (count_sentences >= max_sentences):
            print(count_sentences)
            break
        else:
            try:
                token_sentences = tk.tokenize(text)
                for sentence in token_sentences:
                    tree = parser.parse(sentence)
                    dict1 = {'INDEX': [], 'FORM': [], 'LEMMA': [], 'UPOS': [], 'XPOS': [], 
                            'FEATS': [], 'HEAD': [], 'DEPREL': [], 'DEPS': [], 'MISC': []}
                    sentence = ['\t'.join(node) for node in tree]
                    for line in sentence:
                        try:
                            dict1 = parse_line(dict1, line)
                            node_index += 1
                        except:
                            pass
                    NodeDF = pd.DataFrame(dict1, columns=cols)
                    if (len(NodeDF.index) >= 5):    # sentence length at least 5
                        try:
                            count_sentences += 1
                            EdgeDF = parseSentence(NodeDF, prev_node_index, count_sentences)
                            for i in range(len(EdgeDF.index)):
                                edge_writer.writerow(dict(EdgeDF.iloc[i]))
                            prev_node_index = node_index
                            if ((count_sentences % 1000) == 0):
                                print(count_sentences)
                        except:
                            node_index = prev_node_index
                            pass
                    else:
                        node_index = prev_node_index
            except:
                pass
