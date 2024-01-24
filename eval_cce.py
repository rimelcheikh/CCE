# -*- coding: utf-8 -*-
"""
Created on Thu Oct  5 10:57:14 2023

@author: rielcheikh
"""

import sys, os
import tensorflow as tf
import numpy as np
from scipy.stats import rankdata



import pickle
from os.path import exists
from cce.generate_counterfactuals import run_cce


#import tcav.repro_check as repro_check

import pdb
#pdb.set_trace()


# Computing Spearman's rank correlation coefficient between the sensitivity and the predictio scores
def spearmans_rank(exp, pred):
    R_exp = rankdata(exp)
    R_pred = rankdata(pred)
    
    return np.corrcoef(R_exp, R_pred)



result = {}

#tf.compat.v1.enable_eager_execution() 

"""targets = ['dalmatian','zebra','lion','tiger','hippopotamus','leopard','gorilla','ox','chimpanzee','hamster',
           'weasel','otter','mouse','collie','beaver','skunk']"""


#'plant','papyrus','paper','concrete','soapsuds','chess','crackle','rock','crystal','common marigold',
#'marigold','double knit','knitwear', 'lace','rattlesnake master','track'
#['striped','dotted','blotchy', 'braided', 'bubbly', 'bumpy', 'chequered', 'cobwebbed', 'cracked', 'crystalline', 'frilly', 'knitted', 'lacelike', 'scaly','veined']

"""concepts = ['ocean-s', 'desert-s', 'forest-s','black-c', 'brown-c', 'white-c', 'blue-c', 'orange-c', 'red-c', 'yellow-c']"""

"""dataset = 'imagenet'  
bottleneck = ['mixed10']  #['mixed3a']#,'mixed3b','mixed4a','mixed4b','mixed4c','mixed4d','mixed4e','mixed5a','mixed5b']  
model_name = "inceptionv3"
working_dir = './examples/explanations/'"""



def run_eval_cce(targets, concepts, dataset, concept_dataset, model_name, bottleneck, num_random_exp, alphas, model_cav, res_dir, data_dir):

    if exists(res_dir+"explanations/results.pkl"):
        with open(res_dir+"explanations/results.pkl", 'rb') as fp:
            cce_scores = pickle.load(fp)
    else: 
        cce_scores = run_cce(targets, concepts, dataset, concept_dataset, bottleneck, model_name, res_dir, data_dir, num_random_exp, alphas, model_cav)
#(model_name,targets, concepts)

#cce_scores = run_cce(model_name,targets, concepts)

#build class-wise cce scores 
"""class_cce_scores = {}
for t in targets:
    avg_concept = {}
    class_cce_scores[t] = {}
    for c in concepts:
        avg_concept[c] = []
        #class_cce_scores[t] = {}
        for i in range(len(cce_scores[t])):
            avg_concept[c].append(cce_scores[t][i]['concept_scores'][c])
    
        class_cce_scores[t][c] = sum(avg_concept[c])/len(avg_concept[c])

#computing spearman when concept is fixed
sp_coeff_targets = {}
rationale_dict = {}
for c in concepts:
    rationale_dict[c] = {}
    rationale = []
    cce = []
    for t in targets:
        rationale_dict[c][t] = get_asso_strength(t,c.split('-')[0])
        rationale.append(get_asso_strength(t,c.split('-')[0]))
    
        cce.append(class_cce_scores[t][c])
        
    sp_coeff_targets[c] = spearmans_rank(rationale, cce)[0][1]


with open(working_dir+model_name+'/rationales_dict_concept.pkl', 'wb') as fp:
    pickle.dump(rationale_dict, fp)
    print('Rationales dict saved successfully to file')


with open(working_dir+model_name+'/result_fixed_concept.pkl', 'wb') as fp:
    pickle.dump(sp_coeff_targets, fp)
    print('Spearman coeff on fixed concepts saved successfully to file')


with open(working_dir+model_name+'/result_fixed_concept.pkl', 'rb') as fp:
    print('Results:', pickle.load(fp))






#computing spearman when (target,concept) are fixed
sp_coeff = {}
#tcav_dict = {}
rationale_dict = {}
for target in targets:
    rationale = {}
    for c in concepts:
        #rationale_dict[c] = get_asso_strength(target,c.split('-')[0])
        rationale[c] = (get_asso_strength(target,c.split('-')[0]))
    
    #tcav_dict[target] = class_cce_scores[target]
    rationale_dict[target] = rationale


with open(working_dir+model_name+'/rationales_dict_target.pkl', 'wb') as fp:
    pickle.dump(rationale_dict, fp)
    print('Rationales dict saved successfully to file')

with open(working_dir+model_name+'/rationales_dict_target.pkl', 'rb') as fp:
    rationale_dict = pickle.load(fp)
    
r, s = {}, {}
r_vect, s_vect = [], []
for t in targets:
    for c in concepts:
        r[t,c] = rationale_dict[t][c]
        s[t,c] = class_cce_scores[t][c]
        r_vect.append(rationale_dict[t][c])
        s_vect.append(class_cce_scores[t][c])
sp_coeff = spearmans_rank(r_vect, s_vect)[0][1]


with open(working_dir+model_name+'/result_fixed_target_and_concept.pkl', 'wb') as fp:
    pickle.dump(sp_coeff, fp)
    print('Spearman coeff on fixed targets saved successfully to file')


with open(working_dir+model_name+'/result_fixed_target_and_concept.pkl', 'rb') as fp:
    print('Results:', pickle.load(fp))




#computing spearman when target is fixed
with open(working_dir+model_name+'/rationales_dict_target.pkl', 'rb') as fp:
    rationale_dict = pickle.load(fp)
    
sp_coeff_concepts = {}
for target in targets:
    #rationale_dict = {}
    rationale = []
    for c in concepts:
        #rationale_dict[c] = get_asso_strength(target,c.split('-')[0])
        #rationale.append(get_asso_strength(target,c.split('-')[0]))
        rationale.append(rationale_dict[t][c])

    
    cce = []
    for v in class_cce_scores[target].keys():
        cce.append(class_cce_scores[target][v])
    
    sp_coeff_concepts[target] = spearmans_rank(rationale, cce)[0][1]




with open(working_dir+model_name+'/result_fixed_target.pkl', 'wb') as fp:
    pickle.dump(sp_coeff_concepts, fp)
    print('Spearman coeff on fixed targets saved successfully to file') 


with open(working_dir+model_name+'/result_fixed_target.pkl', 'rb') as fp:
    print('Results:', pickle.load(fp))"""









