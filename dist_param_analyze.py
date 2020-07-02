from gensim.models import KeyedVectors
import numpy as np
import pandas as pd
import csv
import sys
from collections import defaultdict
import os

write_fields = ["sent_id", "sent_pos", "head", "dep_dist", "case_present", "raw_deprel", "deprel", "accessibility", "cosine_dist", "hdmi"]
hdmi_dist_df = {}
integer_fields = ["sent_id", "SententialDistance", "SourceNodeIndex", "TargetNodeIndex"]

considered_deprels = ["k1", "k2", "k4"]

# HDMI generated for different dependency relations using hdmi_online_pos.py
hdmi_df = pd.read_csv("hdmi_dist_df_pos.csv", index_col="dist")

# Word vectors obtained from fasttext - Grave et al. 2018
wv_model = KeyedVectors.load_word2vec_format('cc.hi.300.vec', limit=40000000)

animate_nouns = []
# list of animate nouns extracted from the annotated data Jena et al. 2013
with open("animate_nouns.txt") as f:
    for noun in f:
        animate_nouns.append(noun[:-1])

print("Loaded models")

def similarity_preverb_deps (dep, head_ni, sent_df, norm = "l1"):
    global wv_model
    preverb_deps = sent_df.loc[(sent_df["SourceNodeIndex"] == head_ni) & (sent_df["TargetNodeIndex"] < head_ni)].reset_index()
    max_similarity = 0.0
    for i in preverb_deps.index:
        depi = preverb_deps.iloc[i]["TargetLemma"]
        if (depi != dep):
            try:
                simi = wv_model.similarity(dep, depi)
                if (simi > max_similarity):
                    max_similarity = simi
            except:
                pass
    return max_similarity


def case_present (sent_df, tni):
    noun_mods = sent_df.loc[sent_df["SourceNodeIndex"] == tni]
    noun_mods_psp = noun_mods.loc[(noun_mods["DEPREL"] == "lwg__psp") & (noun_mods["TargetNodeIndex"] == (tni + 1))]
    if (len(noun_mods_psp) > 0):
        return 1
    else:
        return 0

def readable_form(deprel):
    if ((deprel == "k1") or (deprel == "k1s")):
        return "sub"
    elif (deprel == "k2"):
        return "dobj"
    elif (deprel == "k4"):
        return "iobj"
    else:
        return "adj"

def write_analyze (writer, sent_df, sent_base):
    global hdmi_df
    sent_deprel_df = sent_df.loc[(sent_df["TargetUPOS"].isin(["NN", "NNP"])) & (sent_df["SourceUPOS"].str.startswith("VM"))].reset_index()
    sent_length = len(sent_df.index) + 1
    for i in sent_deprel_df.index:
        hdmi_dist_df = {}
        try:
            sent_id = sent_deprel_df.iloc[i]["sent_id"]
            hdmi_dist_df["raw_deprel"] = sent_deprel_df.iloc[i]["DEPREL"]
            deprel = readable_form(sent_deprel_df.iloc[i]["DEPREL"])
            dist = sent_deprel_df.iloc[i]["SententialDistance"]
            tni = sent_deprel_df.iloc[i]["TargetNodeIndex"]
            noun = sent_deprel_df.iloc[i]["TargetLemma"]
            hni = sent_deprel_df.iloc[i]["SourceNodeIndex"]
            ind = sent_deprel_df.iloc[i]["index"]
            # Filling the row
            hdmi_dist_df["dep_dist"] = dist
            hdmi_dist_df["sent_pos"] = round((1.0*(ind+1))/(1.0*sent_length), 4)
            hdmi_dist_df["head"] = hni
            if (noun in animate_nouns):
                hdmi_dist_df["accessibility"] = 1
            else:
                hdmi_dist_df["accessibility"] = 0
            casep = case_present (sent_df, tni)
            hdmi_dist_df["case_present"] = casep
            hdmi_dist_df["hdmi"] = hdmi_df.loc[dist][deprel + "case_hdmi"] if (casep) else hdmi_df.loc[dist][deprel + "noncase_hdmi"]
            hdmi_dist_df["deprel"] = deprel
            hdmi_dist_df["cosine_dist"] = similarity_preverb_deps(noun, hni, sent_df)
            hdmi_dist_df["sent_id"] = sent_id
            # write row
            writer.writerow(hdmi_dist_df)
        except:
            pass

last_sent_id = 0
with open("edges_IITB_parsed.csv", "r", encoding="utf-8", newline='') as f, open("dist_params.csv", "w+") as writef:
    reader = csv.DictReader(f)
    fields = reader.fieldnames
    writer = csv.DictWriter(writef, fieldnames=write_fields)
    sent_df = defaultdict(list)
    # writer.writeheader()
    sent_base = 0
    for row in reader:
        sent_id = int(row['sent_id'])
        if ((sent_id != last_sent_id) and (last_sent_id != 0)):
            if ((last_sent_id % 1000) == 0):
                print(last_sent_id)
            sent_df = pd.DataFrame(sent_df)
            write_analyze(writer, sent_df, sent_base)
            sent_base += len(sent_df.index) + 1
            sent_df = defaultdict(list)
            for field in fields:
                if (field in integer_fields):
                    sent_df[field].append(int(row[field]))
                else:
                    sent_df[field].append(row[field])
        else:
            for field in fields:
                if (field in integer_fields):
                    sent_df[field].append(int(row[field]))
                else:
                    sent_df[field].append(row[field])
        last_sent_id = sent_id
