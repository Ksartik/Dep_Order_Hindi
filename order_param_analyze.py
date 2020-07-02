from gensim.models import KeyedVectors
import numpy as np
import pandas as pd
import csv
import sys
from collections import defaultdict
import os

write_fields = ["sent_id", "head_verb", "d1-d2_dist", "d1_accessibility", "d1_hdmi",
                "d1_cosdist", "d1_case", "d1_deprel", "d2_accessibility", "d2_hdmi",
                "d2_cosdist", "d2_case", "d2_deprel"]

integer_fields = ["sent_id", "SententialDistance", "SourceNodeIndex", "TargetNodeIndex"]

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

def write_info (writer, dep1, dep2, sent_id, head_verb):
    info_df = {}
    info_df["sent_id"] = sent_id
    info_df["head_verb"] = head_verb
    info_df["d1_deprel"] = dep1["DEPREL"]
    info_df["d2_deprel"] = dep2["DEPREL"]
    for param in ["accessibility", "hdmi", "cosdist", "case"]:
        info_df["d1_" + param] = dep1[param]
    for param in ["accessibility", "hdmi", "cosdist", "case"]:
        info_df["d2_" + param] = dep2[param]
    info_df["d1-d2_dist"] = dep1["SententialDistance"] - dep2["SententialDistance"]
    writer.writerow(info_df)

def write_analyze (writer, sent_df, sent_base):
    global hdmi_df
    sent_deprel_df = sent_df.loc[(sent_df["TargetUPOS"].isin(["NN", "NNP"])) & (sent_df["SourceUPOS"].str.startswith("VM"))].reset_index()
    sent_deprel_df["rel_type"] = np.vectorize(readable_form)(sent_deprel_df["DEPREL"])
    sent_deprel_df["case"] = np.vectorize(lambda x : case_present(sent_df, x))(sent_deprel_df["TargetNodeIndex"])
    sent_deprel_df["accessibility"] = np.vectorize(lambda noun: 1 if (noun in animate_nouns) else 0)(sent_deprel_df["TargetLemma"])
    sent_deprel_df["cosdist"] = np.vectorize(lambda x, y: similarity_preverb_deps(x, y, sent_df))(sent_deprel_df["TargetLemma"], sent_deprel_df["SourceNodeIndex"])
    sent_deprel_df["hdmi"] = np.vectorize(lambda dist, cp, rel: hdmi_df.loc[dist][rel + "case_hdmi"] if (cp == 1) else hdmi_df.loc[dist][rel + "noncase_hdmi"]) (sent_deprel_df["SententialDistance"], sent_deprel_df["case"], sent_deprel_df["rel_type"])
    sent_head_gp = sent_deprel_df.groupby(by="SourceNodeIndex")
    for head in sent_head_gp.groups:
        dep_df_head = sent_head_gp.get_group(head)
        try:
            head_verb = np.array(dep_df_head["SourceLemma"]).item()
        except:
            head_verb = dep_df_head.iloc[0]["SourceLemma"]
        for ind, i in enumerate(dep_df_head.index):
            for j in dep_df_head.index[(ind+1):]:
                write_info (writer, dep_df_head.loc[i], dep_df_head.loc[j], sent_id, head_verb)


last_sent_id = 0
with open("edges_IITB_parsed.csv", "r", encoding="utf-8", newline='') as f, open("all_order_params.csv", "w+") as writef:
    reader = csv.DictReader(f)
    fields = reader.fieldnames
    writer = csv.DictWriter(writef, fieldnames=write_fields)
    writer.writeheader()
    sent_df = defaultdict(list)
    sent_base = 0
    for row in reader:
        sent_id = int(row['sent_id'])
        if ((sent_id != last_sent_id) and (last_sent_id != 0)):
            if ((last_sent_id % 1000) == 0):
                print(last_sent_id)
            #if (last_sent_id == 10):
                #break
            sent_df = pd.DataFrame(sent_df)
            #write_analyze (writer, sent_df, sent_base)
            try:
                write_analyze(writer, sent_df, sent_base)
            except:
                pass
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
