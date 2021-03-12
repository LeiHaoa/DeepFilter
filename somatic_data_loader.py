import os
import shutil
import subprocess
import sys

import numpy as np
from numpy.lib.shape_base import split
import torch
from sklearn import preprocessing 
from sklearn.model_selection import train_test_split
import pickle
#from features import features_to_index as fe2i, fvc_selected_features as fvc_sf
from features import som_features_to_index as fe2i, som_selected_features as fvc_sf
import random

type_to_label = {"SNV":           [1, 0, 0], 
                "Insertion":      [0, 1, 0], 
                "Deletion":       [0, 0, 1]}
varLabel_to_label = {
    "Germline"      : [1, 0, 0, 0, 0, 0, 0],
    "StrongLOH"     : [0, 1, 0, 0, 0, 0, 0],
    "LikelyLOH"     : [0, 0, 1, 0, 0, 0, 0],
    "StrongSomatic" : [0, 0, 0, 1, 0, 0, 0],
    "LikelySomatic" : [0, 0, 0, 0, 1, 0, 0],
    "AFDiff"        : [0, 0, 0, 0, 0, 1, 0],
    "SampleSpecific": [0, 0, 0, 0, 0, 0, 1]
}
indels_label = {
                "Deletion": [1, 0, 0], 
                "Insertion":[0, 1, 0], 
                "Complex":  [0, 0, 1], 
                }
snv_label = "SNV"

SOM_INDEL_FEATURES =  53 #46+3
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
        data.extend(varLabel_to_label[jri[fe2i["VarLabel"]]])#varlabel
        data.extend(indels_label[jri[fe2i["VarType" ]]]) #vartype
    else:
        print("not support to train if you not run rabbitvar without --fiser!!")
        exit(-1)
        data.extend(varLabel_to_label[jri[fe2i["VarLabel"]]])#varlabel
        data.extend(indels_label[jri[fe2i["VarType"]]])

    if len(data) != SOM_INDEL_FEATURES:
        print("fvc data length error: \n", len(data), data, " ori\n", jri)
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
            if items[fe2i['VarType']] not in indels_label:
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
    print("get input SNV data done: ", len(fastvc_snv_dict))
    return fastvc_snv_dict

def get_all_data(fvc_result_path):
    fastvc_snv_dict = dict()
    index = -1
    with open(fvc_result_path, 'r') as f:
        for line in f:
            index += 1
            items = line.strip().split("\t")
            if len(items) == 61 :
                if items[fe2i['VarType']] != snv_label:
                    k, d = format_indel_data_item(items, False)
                    fastvc_indel_dict[k] = [d, index]
                else:
                    k, d = format_snv_data_item(items, True)
                    fastvc_snv_dict[k] = [d, index]
            else:
                print("incorrect input file, did you add --fisher when run rabbitvar or vardict?")
                exit(-1)
    print("get input data done: ", len(fastvc_snv_dict))
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

def get_labels_dict(data_dict, truth_path):
    #truth_vars = dict()
    truth_vars = set()
    with open(truth_path, 'r') as f:
        for var in f:
            if var[0] == "#":
                continue
            items = var.split('\t')
            chrom, pos, id, ref, alt, _, filter = items[:7]         
            #if len(chrom) < 6 and filter == "PASS" and (len(ref) > 1 or len(alt) > 1) :
            #if len(chrom) < 6 and filter == "PASS":
            if len(chrom) < 6: #-------just for chm test
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

def prepare_cmds(fasta_file, region_file, bam_file, thread_number, base_path):
    #--- fastvc cmd prepareing ---#
    fvc_list = list() 
    fastvc_path = ""
    fvc_list.append(fastvc_path)
    fvc_list.append("-i {}".format(region_file))
    fvc_list.append("-G {}".format(fasta_file))
    fvc_list.append("-f 0.01")
    fvc_list.append("-N NA12878")
    fvc_list.append("-b {}".format(bam_file))
    fvc_list.append("-c 1 -S 2 -E 3 -g 4")
    fvc_list.append("--fisher")
    fvc_list.append("--th {}".format(thread_number))
    fvc_list.append("--out {}".format(os.path.join(base_path, "tmpspace/fastvc_space/out.txt")))
    fastvc_cmd = " ".join(fvc_list)    

    #--- strelka cmd prepareing ---#
    #-- 1. generate workspace and script --#
    sk2_conf_path = ""
    gen_cmd = "{} --bam {} --referenceFasta {} --callRegions {} --runDir {}".format(sk2_conf_path, 
        bam_file, fasta_file, region_file, os.path.join(base_path, "tmpspace/strelka_space")) 

    #-- 2. strelka run command --#
    sk2_cmd = "{}/tmpspace/strelka_space/runWorkflow.py  -m local -j {}".format(base_path, thread_number)

    return fastvc_cmd, gen_cmd, sk2_cmd

class FastvcCallLoader(torch.utils.data.Dataset):

    def __init__(self, data):
        self.inputs = data[0]
        self.labels = data[1]
        self.raw_indexs = data[2]

    def __getitem__(self, index):
        input, label = self.inputs[index], self.labels[index]
        raw_index = self.raw_indexs[index]
        return input, np.asarray(label), raw_index

    def __len__(self):
        return len(self.labels)

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
    def __init__(self, reload, training, re_exec, vartype, pama_list, base_path, truth_path):
        self.training = training
        self.data = dict()
        self.inputs = list()
        self.labels = list()
        self.raw_indexs = list()
        self.re_exec = re_exec
        self.prepare_data(reload, vartype, pama_list, base_path, truth_path)

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
            pos_num, neg_num, fastvc_label_dict = get_labels_dict(merged_data_dict, truth_path)
            print("get label done, size: {}, pos_num: {}, neg_num: {}".format(len(fastvc_label_dict), pos_num, neg_num))
            keys = list()
            for k, v in merged_data_dict.items():
                if self.training and fastvc_label_dict[k] == 0 and random.randint(0, 99) > 0:
                    continue
                keys.append(k)
                self.inputs.append(v[0])
                self.labels.append(fastvc_label_dict[k])
                self.raw_indexs.append(v[1])
            print("data size:", len(merged_data_dict), "after f:", len(keys))
            #--- standlizaton ---#            
            '''
            min_max_scaler = preprocessing.MinMaxScaler()
            self.inputs = min_max_scaler.fit_transform(self.inputs)
            '''
            #---inputs Normalization ---#
            self.inputs = np.asfarray(self.inputs)
            print("start normalization...")
            #self.inputs = preprocessing.normalize(self.inputs, axis = 0, norm = 'l2') 
            self.inputs = preprocessing.scale(self.inputs, axis = 0, with_mean = True, with_std = True, copy = True) 
            print("[info] inputs shape:", self.inputs.shape)
            '''
            np.set_printoptions(precision = 3, threshold=1000)
            for i in range(10000):
                print(self.keys[i], self.inputs[i], self.labels[i])
            exit(0)
            '''
            print("normalization done")
            print("FastvcDataset init over")

    def split(self, test_size = 0.2, random_state = 0):
        print("spliting data...")
        x_train, x_test, y_train, y_test = train_test_split(self.inputs, self.labels, 
                                test_size = test_size, random_state = random_state)

        self.data['train'] = [x_train, y_train]
        self.data['test'] = [x_test, y_test]

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
