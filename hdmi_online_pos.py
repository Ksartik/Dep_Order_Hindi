import numpy as np
import sys
from collections import defaultdict
import os
import re
import pandas as pd
import csv


def compute_mutual_info(fdist_x_y, fdist_x, fdist_y):
    mi = 0
    total_freq = float(np.sum(list(fdist_x_y.values())))
    for x_y in fdist_x_y:
        x, y = x_y.split('\t')
        mi += (fdist_x_y[x_y] / float(total_freq)) * (
                    np.log2(fdist_x_y[x_y]) + np.log2(total_freq) - np.log2(fdist_x[x]) - np.log2(fdist_y[y]))
    return mi

class DistInfoModel:
    def __init__(self):
        """
        self.fdist_<condition><i> 
        where i is 
            1 when the head (verb)
            2 when the dependent (noun)
        """
        self.fdist_argcase1 = defaultdict(int)
        self.fdist_argcase2 = defaultdict(int)
        self.fdist_argcase1_argcase2 = defaultdict(int)

        self.fdist_argnoncase1 = defaultdict(int)
        self.fdist_argnoncase2 = defaultdict(int)
        self.fdist_argnoncase1_argnoncase2 = defaultdict(int)

        self.fdist_subcase1 = defaultdict(int)
        self.fdist_subcase2 = defaultdict(int)
        self.fdist_subcase1_subcase2 = defaultdict(int)

        self.fdist_dobjcase1 = defaultdict(int)
        self.fdist_dobjcase2 = defaultdict(int)
        self.fdist_dobjcase1_dobjcase2 = defaultdict(int)

        self.fdist_iobjcase1 = defaultdict(int)
        self.fdist_iobjcase2 = defaultdict(int)
        self.fdist_iobjcase1_iobjcase2 = defaultdict(int)

        self.fdist_subnoncase1 = defaultdict(int)
        self.fdist_subnoncase2 = defaultdict(int)
        self.fdist_subnoncase1_subnoncase2 = defaultdict(int)

        self.fdist_dobjnoncase1 = defaultdict(int)
        self.fdist_dobjnoncase2 = defaultdict(int)
        self.fdist_dobjnoncase1_dobjnoncase2 = defaultdict(int)

        self.fdist_iobjnoncase1 = defaultdict(int)
        self.fdist_iobjnoncase2 = defaultdict(int)
        self.fdist_iobjnoncase1_iobjnoncase2 = defaultdict(int)

        self.fdist_adjcase1 = defaultdict(int)
        self.fdist_adjcase2 = defaultdict(int)
        self.fdist_adjcase1_adjcase2 = defaultdict(int)

        self.fdist_adjnoncase1 = defaultdict(int)
        self.fdist_adjnoncase2 = defaultdict(int)
        self.fdist_adjnoncase1_adjnoncase2 = defaultdict(int)
        self.num_of_datapoints = 0

    def total_freq_argcase (self):
        return np.sum(list(self.fdist_argcase1_argcase2.values()))
    
    def get_argcase_mutual_info(self):
        return compute_mutual_info(self.fdist_argcase1_argcase2, self.fdist_argcase1, self.fdist_argcase2)

    def total_freq_argnoncase (self):
        return np.sum(list(self.fdist_argnoncase1_argnoncase2.values()))

    def get_argnoncase_mutual_info(self):
        return compute_mutual_info(self.fdist_argnoncase1_argnoncase2, self.fdist_argnoncase1, self.fdist_argnoncase2)

    def total_freq_subcase (self):
        return np.sum(list(self.fdist_subcase1_subcase2.values()))

    def get_subcase_mutual_info(self):
        return compute_mutual_info(self.fdist_subcase1_subcase2, self.fdist_subcase1, self.fdist_subcase2)

    def total_freq_dobjcase (self):
        return np.sum(list(self.fdist_dobjcase1_dobjcase2.values()))

    def get_dobjcase_mutual_info(self):
        return compute_mutual_info(self.fdist_dobjcase1_dobjcase2, self.fdist_dobjcase1, self.fdist_dobjcase2)

    def total_freq_iobjcase (self):
        return np.sum(list(self.fdist_iobjcase1_iobjcase2.values()))

    def get_iobjcase_mutual_info(self):
        return compute_mutual_info(self.fdist_iobjcase1_iobjcase2, self.fdist_iobjcase1, self.fdist_iobjcase2)

    def total_freq_subnoncase (self):
        return np.sum(list(self.fdist_subnoncase1_subnoncase2.values()))

    def get_subnoncase_mutual_info(self):
        return compute_mutual_info(self.fdist_subnoncase1_subnoncase2, self.fdist_subnoncase1, self.fdist_subnoncase2)

    def total_freq_dobjnoncase (self):
        return np.sum(list(self.fdist_dobjnoncase1_dobjnoncase2.values()))

    def get_dobjnoncase_mutual_info(self):
        return compute_mutual_info(self.fdist_dobjnoncase1_dobjnoncase2, self.fdist_dobjnoncase1, self.fdist_dobjnoncase2)

    def total_freq_iobjnoncase (self):
        return np.sum(list(self.fdist_iobjnoncase1_iobjnoncase2.values()))

    def get_iobjnoncase_mutual_info(self):
        return compute_mutual_info(self.fdist_iobjnoncase1_iobjnoncase2, self.fdist_iobjnoncase1, self.fdist_iobjnoncase2)

    def total_freq_adjcase (self):
        return np.sum(list(self.fdist_adjcase1_adjcase2.values()))
    
    def get_adjcase_mutual_info(self):
        return compute_mutual_info(self.fdist_adjcase1_adjcase2, self.fdist_adjcase1, self.fdist_adjcase2)

    def total_freq_adjnoncase (self):
        return np.sum(list(self.fdist_adjnoncase1_adjnoncase2.values()))

    def get_adjnoncase_mutual_info(self):
        return compute_mutual_info(self.fdist_adjnoncase1_adjnoncase2, self.fdist_adjnoncase1, self.fdist_adjnoncase2)

    def update_model_params_from_sentence(self, sent_dist_df, sent_df):
        for i in sent_dist_df.index:
            head_lemma = sent_dist_df.loc[i]["SourceLemma"]
            dep_lemma = sent_dist_df.loc[i]["TargetLemma"]
            head_pos = sent_dist_df.loc[i]["SourceUPOS"]
            dep_pos = sent_dist_df.loc[i]["TargetUPOS"]
            deprel = sent_dist_df.loc[i]["DEPREL"]
            if ((dep_pos in ["NN", "NNP"]) and (re.match("VM:*", head_pos))):
                noun_mods = sent_df.loc[sent_df["SourceNodeIndex"] == sent_df.iloc[i]["TargetNodeIndex"]]
                noun_mods_psp = noun_mods.loc[(noun_mods["DEPREL"] == "lwg__psp") & (noun_mods["TargetNodeIndex"] == (sent_df.iloc[i]["TargetNodeIndex"] + 1))]
                if (len(noun_mods_psp) > 0):
                    # Case on noun
                    # Supertagging with the case marker
                    dep_pos = dep_pos + ":" + noun_mods_psp.iloc[0]["TargetLemma"]
                    if (deprel in ["k1", "k1s", "k2", "k4"]):
                        # arguments for the verb
                        self.fdist_argcase1[head_pos] += 1
                        self.fdist_argcase2[dep_pos] += 1
                        self.fdist_argcase1_argcase2[head_pos + '\t' + dep_pos] += 1
                        if ((deprel == "k1") or (deprel == "k1s")):
                            # subjects
                            self.fdist_subcase1[head_pos] += 1
                            self.fdist_subcase2[dep_pos] += 1
                            self.fdist_subcase1_subcase2[head_pos + '\t' + dep_pos] += 1
                        elif (deprel == "k2"):
                            # direct objects
                            self.fdist_dobjcase1[head_pos] += 1
                            self.fdist_dobjcase2[dep_pos] += 1
                            self.fdist_dobjcase1_dobjcase2[head_pos + '\t' + dep_pos] += 1
                        elif (deprel == "k4"):
                            # indirect objects
                            self.fdist_iobjcase1[head_pos] += 1
                            self.fdist_iobjcase2[dep_pos] += 1
                            self.fdist_iobjcase1_iobjcase2[head_pos + '\t' + dep_pos] += 1
                        else:
                            pass
                    else:
                        self.fdist_adjcase1[head_pos] += 1
                        self.fdist_adjcase2[dep_pos] += 1
                        self.fdist_adjcase1_adjcase2[head_pos + '\t' + dep_pos] += 1
                else:
                    # No case on the noun
                    if (deprel in ["k1", "k1s", "k2", "k4"]):
                        # arguments for the verb
                        self.fdist_argnoncase1[head_pos] += 1
                        self.fdist_argnoncase2[dep_pos] += 1
                        self.fdist_argnoncase1_argnoncase2[head_pos + '\t' + dep_pos] += 1
                        if ((deprel == "k1") or (deprel == "k1s")):
                            # subjects
                            self.fdist_subnoncase1[head_pos] += 1
                            self.fdist_subnoncase2[dep_pos] += 1
                            self.fdist_subnoncase1_subnoncase2[head_pos + '\t' + dep_pos] += 1
                        elif (deprel == "k2"):
                            # direct objects
                            self.fdist_dobjnoncase1[head_pos] += 1
                            self.fdist_dobjnoncase2[dep_pos] += 1
                            self.fdist_dobjnoncase1_dobjnoncase2[head_pos + '\t' + dep_pos] += 1
                        elif (deprel == "k4"):
                            # indirect objects
                            self.fdist_iobjnoncase1[head_pos] += 1
                            self.fdist_iobjnoncase2[dep_pos] += 1
                            self.fdist_iobjnoncase1_iobjnoncase2[head_pos + '\t' + dep_pos] += 1
                        else:
                            pass
                    else:
                        # adjuncts
                        self.fdist_adjnoncase1[head_pos] += 1
                        self.fdist_adjnoncase2[dep_pos] += 1
                        self.fdist_adjnoncase1_adjnoncase2[head_pos + '\t' + dep_pos] += 1
                    

write_fields = ['dist', 'argcase_freq', 'argcase_hdmi', 'argnoncase_freq', 'argnoncase_hdmi',
                'adjcase_freq', 'adjcase_hdmi', 'adjnoncase_freq', 'adjnoncase_hdmi',
                'subcase_freq', 'subcase_hdmi', 'subnoncase_freq', 'subnoncase_hdmi',
                'dobjcase_freq', 'dobjcase_hdmi', 'dobjnoncase_freq', 'dobjnoncase_hdmi',
                'iobjcase_freq', 'iobjcase_hdmi', 'iobjnoncase_freq', 'iobjnoncase_hdmi',
                'argcase_freq', 'argcase_hdmi', 'argnoncase_freq', 'argnoncase_hdmi',
                'argcase_freq', 'argcase_hdmi', 'argnoncase_freq', 'argnoncase_hdmi']
hdmi_dist_df = {}
integer_fields = ["sent_id", "SententialDistance", "SourceNodeIndex", "TargetNodeIndex"]

model = DistInfoModel()
disti = int(sys.argv[1])

last_sent_id = 0
with open("edges_IITB_parsed.csv", "r", encoding="utf-8") as f, open("hdmi_dist_df_pos.csv", "a") as writef:
    reader = csv.DictReader(f)
    fields = reader.fieldnames
    writer = csv.DictWriter(writef, fieldnames=write_fields)
    sent_df = defaultdict(list)
    # writer.writeheader()
    for row in reader:
        try:
            sent_id = int(row['sent_id'])
            if ((sent_id != last_sent_id) and (last_sent_id != 0)):
                if ((last_sent_id % 1000) == 0):
                    print(disti, last_sent_id)
                sent_df = pd.DataFrame(sent_df)
                sent_dist_df = sent_df.loc[sent_df['SententialDistance'] == disti]
                model.update_model_params_from_sentence(sent_dist_df, sent_df)
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
        except:
            pass
    hdmi_dist_df["dist"] = disti
    hdmi_dist_df["argcase_freq"] = model.total_freq_argcase()
    hdmi_dist_df["argcase_hdmi"] = model.get_argcase_mutual_info()
    hdmi_dist_df["argnoncase_freq"] = model.total_freq_argnoncase()
    hdmi_dist_df["argnoncase_hdmi"] = model.get_argnoncase_mutual_info()
    hdmi_dist_df["subcase_freq"] = model.total_freq_subcase()
    hdmi_dist_df["subcase_hdmi"] = model.get_subcase_mutual_info()
    hdmi_dist_df["dobjcase_freq"] = model.total_freq_dobjcase()
    hdmi_dist_df["dobjcase_hdmi"] = model.get_dobjcase_mutual_info()
    hdmi_dist_df["iobjcase_freq"] = model.total_freq_iobjcase()
    hdmi_dist_df["iobjcase_hdmi"] = model.get_iobjcase_mutual_info()
    hdmi_dist_df["subnoncase_freq"] = model.total_freq_subnoncase()
    hdmi_dist_df["subnoncase_hdmi"] = model.get_subnoncase_mutual_info()
    hdmi_dist_df["dobjnoncase_freq"] = model.total_freq_dobjnoncase()
    hdmi_dist_df["dobjnoncase_hdmi"] = model.get_dobjnoncase_mutual_info()
    hdmi_dist_df["iobjnoncase_freq"] = model.total_freq_iobjnoncase()
    hdmi_dist_df["iobjnoncase_hdmi"] = model.get_iobjnoncase_mutual_info()
    hdmi_dist_df["adjcase_freq"] = model.total_freq_adjcase()
    hdmi_dist_df["adjcase_hdmi"] = model.get_adjcase_mutual_info()
    hdmi_dist_df["adjnoncase_freq"] = model.total_freq_adjnoncase()
    hdmi_dist_df["adjnoncase_hdmi"] = model.get_adjnoncase_mutual_info()
    writer.writerow(hdmi_dist_df)
