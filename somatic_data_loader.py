import os
import pickle
import random
import shutil
import subprocess
import sys
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), './utils'))
from datautil import *

import numpy as np
import pandas as pd
import torch
from numpy.lib.shape_base import split
from sklearn import preprocessing
from sklearn.model_selection import train_test_split

#from features import features_to_index as fe2i, fvc_selected_features as fvc_sf
from features import som_features_to_index as fe2i
from features import som_selected_features as fvc_sf

varLabel_to_label_onehot = {
    "Germline"      : [1, 0, 0, 0, 0, 0, 0],
    "StrongLOH"     : [0, 1, 0, 0, 0, 0, 0],
    "LikelyLOH"     : [0, 0, 1, 0, 0, 0, 0],
    "StrongSomatic" : [0, 0, 0, 1, 0, 0, 0],
    "LikelySomatic" : [0, 0, 0, 0, 1, 0, 0],
    "AFDiff"        : [0, 0, 0, 0, 0, 1, 0],
    "SampleSpecific": [0, 0, 0, 0, 0, 0, 1]
}
varLabel_to_label = {
    "Germline"      : 0,
    "StrongLOH"     : 1,
    "LikelyLOH"     : 2,
    "StrongSomatic" : 3,
    "LikelySomatic" : 4,
    "AFDiff"        : 5,
    "SampleSpecific": 6,
}
label_to_varLabel = {
   0 :  "Germline"      ,
   1 :  "StrongLOH"     ,
   2 :  "LikelyLOH"     ,
   3 :  "StrongSomatic" ,
   4 :  "LikelySomatic" ,
   5 :  "AFDiff"        ,
   6 :  "SampleSpecific",
}

indels_label_onehot = {
                "Deletion": [1, 0, 0], 
                "Insertion":[0, 1, 0], 
                "Complex":  [0, 0, 1], 
                }
types_to_label = {
                "SNV": 0,
                "Deletion": 1, 
                "Insertion":2, 
                "Complex":  3, 
                }
label_to_types = {
        0: "SNV",
        1: "Deletion", 
        2: "Insertion", 
        3: "Complex", 
        }
snv_label = "SNV"

#SOM_INDEL_FEATURES =  43 #46+3
SOM_INDEL_FEATURES =  2 + len(fvc_sf) + 3 + 7
SOM_SNV_FEATURES = 41 + 7 #41 + 7

def format_indel_data_item(jri, fisher = True):
    #[FIXME] fisher should always be true, otherwish the map is wrong
    data = list()
    # key is chrom:pos like "chr1:131022:A:AT"
    key = jri[2] + ":" + jri[3] + ":" + jri[5] + ":" + jri[6]
    data.append(len(jri[fe2i["Ref"]])) #refallele len
    data.append(len(jri[fe2i["Alt"]])) #varallel len
    for sf in fvc_sf:
        data.append(jri[fe2i[sf]])
     
    if fisher:
        data.extend(indels_label_onehot[jri[fe2i["VarType"]]]) #vartype
        data.extend(varLabel_to_label_onehot[jri[fe2i["VarLabel"]]])#varlabel
    else:
        print("not support to train if you not run rabbitvar without --fiser!!")
        exit(-1)

    if len(data) != SOM_INDEL_FEATURES:
        print("fvc data length error: \n", len(data), "-", SOM_INDEL_FEATURES , data, " ori\n", jri)
        exit(-1)
    #print("format:", key, data)
    return key, data

def format_snv_data_item(jri, fisher = True):
    if not fisher:
        print("not support to train if you not run rabbitvar without --fiser!!")
        exit(-1)
    data = list()
    # key is chrom:pos like "chr1:131022:A:T"
    key = jri[2] + ":" + jri[3] + ":" + jri[5] + ":" + jri[6] #TODO: case sensitive
    for sf in fvc_sf:
        data.append(jri[fe2i[sf]])
    data.extend(varLabel_to_label[jri[fe2i["VarLabel"]]])#varlabel
    if len(data) != SOM_SNV_FEATURES:
        print("fvc data length error: \n", len(data), data, " ori\n", jri)
        exit(-1)
    return key, data

def get_data(fvc_result_path, vtype = 'SNV'):
    if vtype.upper() == 'SNV':
        return get_snv_data(fvc_result_path)
    elif vtype.upper() == 'INDEL':
        return get_indel_data(fvc_result_path)
    else:
        print("unrecognized variant type: {} !".format(vtype))
        exit(-1)

def get_indel_data(fvc_result_path):
    #--- read fastvc result file and format ---#
    fastvc_indel_dict = dict()
    index = -1
    with open(fvc_result_path, 'r') as f:
        for line in f:
            index += 1
            items = line.strip().split("\t")
            #for i, it in enumerate(items):
            #    print(i, "--", it)
            #exit(0)
            if items[fe2i['VarType']] not in indels_label_onehot:
                #print(items[fe2i['VarType']])
                continue
            if len(items) == 55:
                k, d = format_indel_data_item(items, False)
                fastvc_indel_dict[k] = [d, index]
            elif len(items) == 61 :
                k, d = format_indel_data_item(items, True)
                fastvc_indel_dict[k] = [d, index]
            else:
                print("your train file should be 55 or 61 items! but you have {} items".format(len(items)))
    print("get fastvc indels data done: ", len(fastvc_indel_dict))
    return fastvc_indel_dict

def get_snv_data(fvc_result_path):
    fastvc_snv_dict = dict()
    index = -1
    with open(fvc_result_path, 'r') as f:
        for line in f:
            index += 1
            items = line.strip().split("\t")
            if items[fe2i['VarType']] != snv_label:
                continue
            if len(items) == 55:
                k, d = format_snv_data_item(items, False)
                fastvc_snv_dict[k] = [d, index]
            elif len(items) == 61 :
                k, d = format_snv_data_item(items, True)
                fastvc_snv_dict[k] = [d, index]
    print("get fastvc SNV data done: ", len(fastvc_snv_dict))
    return fastvc_snv_dict

def run_tools_and_get_data(fastvc_cmd, gen_cmd, strelka_cmd, base_path):
    tmpspace = os.path.join(base_path, "tmpspace") 
    if not os.path.exists(tmpspace):
        os.mkdir(tmpspace)
        os.mkdir(os.path.join(tmpspace, "strelka_space"))
        os.mkdir(os.path.join(tmpspace, "fastvc_space"))
    else:
        print("tmpspace exists! delete it first!")
        exit(-1)
    ret = subprocess.check_call(fastvc_cmd, shell = True)
    if not ret:
        print("fastvc runing error!!")
        exit(-1)
    #--- read fastvc result file and format ---#
    fastvc_dict = dict()
    with open(os.path.join(base_path, "tmpspace/fastvc_space/out.txt"), 'r') as f:
        for line in f:
            items = line.split("\t")
            if len(items) == 36 :
                k, d= format_data_item(items, False)
                fastvc_dict[k] = d
            elif len(items) == 38 :
                k, d = format_data_item(items, True)
                fastvc_dict[k] = d

    return fastvc_dict           

def get_labels_dict_SJZP(data_dict, truth_path):
    #truth_vars = dict()
    truth_vars = set()
    with open(truth_path, 'r') as f:
        for line in f:
            var = line.split('\t')
            chrom = ""
            if(var[0][:3] == "chr"):
                chrom = var[0][3:]
            else:
                chrom = var[0]

            pos, ref, alt, vaf = var[1], var[3], var[4], var[7]       
            site = chrom + ":" + pos + ":" + ref.upper() + ":" + alt.upper()
            #print("truth site:", site)
            truth_vars.add(site)
    #-------process detected result--------#
    print("totally {} truth site".format(len(truth_vars)))
    labels_dict = {}
    positive_num = 0
    negtive_num = 0
    for k, v in data_dict.items():
        v = v[0]
        [chrom, pos, ref, alt] = k.split(':')
        if(chrom == "chr"):
            chrom = chrom[0][3:]
        var_type = ""
        site = k
        if len(v) == SOM_INDEL_FEATURES:
            if v[50] == 1:
                var_type = "Deletion"
                #print("find deletion", k)
            elif v[51] == 1:
                var_type = "Insertion"
                #print("find insertion", k)
        if var_type == "Insertion":
            ref = "."
            alt = alt[1:]
            site = chrom + ":" + pos + ":" + ref.upper() + ":" + alt.upper()
        elif var_type == "Deletion":
            pos = str(int(pos) + 1)
            ref = ref[1:]
            alt = "."
            site = chrom + ":" + pos + ":" + ref.upper() + ":" + alt.upper()
        #print("site:", site)
        if site in truth_vars:
            labels_dict[k] = 1
            positive_num += 1
        else:
            labels_dict[k] = 0
            negtive_num += 1
    return positive_num, negtive_num, labels_dict

def get_labels_dict(data_dict, truth_path):
    #truth_vars = dict()
    truth_vars = set()
    with open(truth_path, 'r') as f:
        for var in f:
            items = var.split('\t')
            if(len(items) == 10 ):
                chrom, pos, id, ref, alt, _, filter = items[:7]         
                #if len(chrom) < 6 and filter == "PASS" and (len(ref) > 1 or len(alt) > 1) :
                if len(chrom) < 6 and filter == "PASS":
                #if len(chrom) < 6: #-------just for chm test
                    alts = alt.split(",")
                    for alt in alts:
                        site = chrom + ":" + pos + ":" + ref.upper() + ":" + alt.upper()
                        truth_vars.add(site)
                    #truth_vars[site] = list([ref, alt])  
    print("totally {} truth site".format(len(truth_vars)))
    labels_dict = {}
    positive_num = 0
    negtive_num = 0
    for k, v in data_dict.items():
        if k in truth_vars:
            #labels_dict[k] = [1, 0]
            labels_dict[k] = 1
            positive_num += 1
        else:
            #labels_dict[k] = [0, 1]
            labels_dict[k] = 0
            negtive_num += 1
    return positive_num, negtive_num, labels_dict

class FastvcCallLoader(torch.utils.data.Dataset):

    def __init__(self, data):
        #self.labels = data[1]
        #self.raw_indexs = data[2]
        self.inputs = data

    def __getitem__(self, index):
        #input, label = self.inputs[index], self.labels[index]
        #raw_index = self.raw_indexs[index]
        #return input, np.asarray(label), raw_index
        return self.inputs[index]

    def __len__(self):
        return len(self.inputs)

class FastvcTrainLoader(torch.utils.data.Dataset):
    def __init__(self, data):
        self.inputs = data[0]
        self.labels = data[1]

    def __getitem__(self, index):
        input, label = self.inputs[index], self.labels[index]
        return input, np.asarray(label)

    def __len__(self):
        return len(self.labels)

class Dataset:
    def __init__(self, is_train, re_exec, vartype, pama_list, base_path, truth_path):
        self.indel_input_header = ["RefLength", "AltLength", *fvc_sf, "Deletion", "Insertion", "Complex","Germline", "StrongLOH", "LikelyLOH", "StrongSomatic", "LikelySomatic", "AFDiff", "SampleSpecific"]
        self.snv_input_header = [*fvc_sf, "Germline", "StrongLOH", "LikelyLOH", "StrongSomatic", "LikelySomatic", "AFDiff", "SampleSpecific"]
        self.data = dict()
        self.inputs = list()
        self.labels = list()
        self.raw_indexs = list()
        self.stdscaler = preprocessing.StandardScaler()
        self.re_exec = re_exec
        if is_train:
            self.prepare_from_tsv(pama_list, vartype)
        else:
            self.prepare_from_txt(pama_list, vartype)
        #self.prepare_data(reload, vartype, pama_list, base_path, truth_path)

    def prepare_from_tsv(self, pama_list, vartype):
        print("[debug], tsv filename: ", pama_list)
        filepath = pama_list[0]
        if vartype == 'INDEL':
            df_header = ["RefLength", "AltLength", "VarType", *fvc_sf, "VarLabel", "label"]
        else:
            df_header = [*fvc_sf, "VarLabel", "label"]

        self.df = pd.read_csv(filepath, header=None)
        self.df.columns = df_header
        ## hard filter first
        print("before hard filter: ", len(self.df))
        self.df = hard_filter(self.df)
        print("after hard filter: ", len(self.df))
        #----- experiments: keep only af >= 0.1
        #self.df = self.df[self.df["Var1AF"] >= 0.1]
        print("high af number:", len(self.df))

        print("truth-false: ", sum(self.df['label'] == 1), sum(self.df['label'] == 0) )

        #process label to onehort
        if vartype == "INDEL":
            columns = ["Deletion", "Insertion", "Complex"]
            tmp = self.df["VarType"].apply(lambda x: indels_label_onehot[label_to_types[x]]).to_list()
            assert len(tmp) == len(self.df) 
            self.df[columns] = tmp

        tmp = self.df["VarLabel"].apply(lambda x: varLabel_to_label_onehot[label_to_varLabel[x]]).to_list()
        columns = ["Germline", "StrongLOH", "LikelyLOH", "StrongSomatic", "LikelySomatic", "AFDiff", "SampleSpecific"]
        self.df[columns] = tmp
        assert len(tmp) == len(self.df) 

        if vartype == "INDEL":
            self.inputs = self.df[self.indel_input_header]
        else:
            self.inputs = self.df[self.snv_input_header]

        assert not self.inputs.isnull().values.any()
        self.labels = self.df['label'].to_numpy()
        #self.inputs = preprocessing.scale(self.inputs, axis = 0, with_mean = True, with_std = True, copy = True)
        print("start normalization...")
        self.inputs = preprocessing.normalize(self.inputs, axis = 0, norm = 'l2') 
        #self.inputs = preprocessing.scale(self.inputs, axis = 0, with_mean = True, with_std = True, copy = True) 
        print("Normalization done")

    def prepare_from_txt(self, pama_list, vartype):
        in_file = pama_list[0]
        cr = pd.read_csv(in_file, delimiter = '\t', header = None, engine = 'c', skipinitialspace = True)
        cr.columns = [*som_features, 'None'] #TODO: i should change the code of c++ to avoid the None colum
        cr['VarLabel'] = cr['VarLabel'].map(varLabel_to_label)
        cr['VarType'] = cr['VarType'].map(types_to_label)
        cr['RefLength'] = cr['Ref'].str.len()
        cr['AltLength'] = cr['Alt'].str.len()
        columns = ["Deletion", "Insertion", "Complex"]
        if vartype == "INDEL":
            cr = cr[cr["VarType"] != 0]
            print("before hard filter: ", len(cr))
            cr = hard_filter(cr)
            print("after hard filter: ", len(cr))
            #----- experiments: keep only af >= 0.1
            #cr = cr[cr['Var1AF'] >= 0.1]
            print("high af variant: ", len(cr))
            tmp = cr["VarType"].apply(lambda x: indels_label_onehot[label_to_types[x]]).to_list()
            assert len(tmp) == len(cr) 
            cr[columns] = tmp
            tmp = cr["VarLabel"].apply(lambda x: varLabel_to_label_onehot[label_to_varLabel[x]]).to_list()
            columns = ["Germline", "StrongLOH", "LikelyLOH", "StrongSomatic", "LikelySomatic", "AFDiff", "SampleSpecific"]
            cr[columns] = tmp

            self.df = cr
            self.inputs = self.df[self.indel_input_header]

        #snv data process 
        elif vartype == "SNV":
            cr = cr[cr['VarType'] == 0]
            print("before hard filter: ", len(cr))
            cr = hard_filter(cr)
            print("after hard filter: ", len(cr))

            tmp = cr["VarLabel"].apply(lambda x: varLabel_to_label_onehot[label_to_varLabel[x]]).to_list()
            columns = ["Germline", "StrongLOH", "LikelyLOH", "StrongSomatic", "LikelySomatic", "AFDiff", "SampleSpecific"]
            cr[columns] = tmp
            self.df = cr
            self.inputs = self.df[self.snv_input_header]
        else:
            print('error variant type: ', vartype)

        #self.inputs = preprocessing.scale(self.inputs, axis = 0, with_mean = True, with_std = True, copy = True)
        self.inputs = self.stdscaler.fit_transform(self.inputs)
        print("Normalization done")

    def prepare_data(self, reload, vartype, pama_list, base_path, truth_path):
        print('[debug]', pama_list, base_path, truth_path)
        if not reload:
            merged_data_dict = {}
            if self.re_exec:
                region_file, fasta_file, bam_file = pama_list
                fastvc_cmd, gen_cmd, sk2_cmd = prepare_cmds(fasta_file, region_file, bam_file, 40, base_path)
                merged_data_dict = run_tools_and_get_data(fastvc_cmd, gen_cmd, sk2_cmd, base_path) 
            else:
                fvc_res_path = pama_list[0]
                merged_data_dict = get_data(fvc_res_path, vtype = vartype) 
            assert(len(merged_data_dict) > 0)

            print("get merged data done, merged data dict size: ", len(merged_data_dict))
            pos_num, neg_num, fastvc_label_dict = get_labels_dict_SJZP(merged_data_dict, truth_path)
            print("get label done, size: {}, pos_num: {}, neg_num: {}".format(len(fastvc_label_dict), pos_num, neg_num))
            keys = list()
            for k, v in merged_data_dict.items():
                if fastvc_label_dict[k] == 0 and random.randint(0, 9) > 10:
                    continue
                keys.append(k)
                self.inputs.append(v[0])
                self.labels.append(fastvc_label_dict[k])
                self.raw_indexs.append(v[1])
            assert len(merged_data_dict) == len(keys)
            #--- load into dataframe ---#
            self.inputs = pd.DataFrame(self.inputs, columns=self.indel_input_header, dtype=float)
            data = self.inputs
            self.hard_flag = ((data['Var1AF'] < 0.01)
                              | (data['Var1QMean'] <= 20)
                              | (data['Var1NM'] >= 6)
                              | (data['Var1NM'] < 0)
                              | ((data['StrongSomatic'] != 1) & (data['LikelySomatic'] != 1)))
            print("hard flag length: ", len(self.hard_flag), self.hard_flag)
            assert len(self.hard_flag) == len(self.inputs)
            #--- standlizaton ---#            
            '''
            min_max_scaler = preprocessing.MinMaxScaler()
            self.inputs = min_max_scaler.fit_transform(self.inputs)
            '''
            #---inputs Normalization ---#
            #self.inputs = np.asfarray(self.inputs)
            print("start normalization...")
            self.inputs = preprocessing.normalize(self.inputs, axis = 0, norm = 'l2') 
            #self.inputs = preprocessing.scale(self.inputs, axis = 0, with_mean = True, with_std = True, copy = True) 
            print("[info] inputs shape:", self.inputs.shape)
            '''
            np.set_printoptions(precision = 3, threshold=1000)
            for i in range(10000):
                print(self.keys[i], self.inputs[i], self.labels[i])
            exit(0)
            '''
            print("normalization done")
            print("FastvcDataset init over")

    def split(self, test_size = 0.1, random_state = 0):
        print("spliting data...")
        x_train, x_test, y_train, y_test = train_test_split(self.inputs, self.labels, 
                                test_size = test_size, random_state = random_state)

        print("split result:", type(x_train), x_train.shape)
        self.data['train'] = [self.stdscaler.fit_transform(x_train), y_train]
        self.data['test'] = [self.stdscaler.fit_transform(x_test), y_test]

    def load(self, store_path):
        if not os.path.exists(store_path):
            print("file {} do not exists!".format(store_path))
            exit(-1)
        with open(store_path, 'rb') as f:
            self.data = pickle.load(f)
        print("load data over!")
        
    def store(self, store_path):
        if os.path.exists(store_path):
            print("file {} exists!".format(store_path))
            exit(-1)
        with open(store_path, 'wb') as f:
            pickle.dump(self.data, f)
        print("dump finished!")
