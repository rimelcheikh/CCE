import numpy as np
import pandas as pd
import csv

awa_dir = 'C:/Users/rielcheikh/Desktop/XAI/data/AwA2-base/Animals_with_Attributes2'   
current_dir = 'C:/Users/rielcheikh/Desktop/XAI/tcav'

def awa_rationales():
    awa_rationales_mat = np.loadtxt(awa_dir + '/predicate-matrix-continuous.txt')
    classes = np.loadtxt(awa_dir+'/classes.txt',dtype=str, usecols=1)
    attributes = np.loadtxt(awa_dir+'/predicates.txt',dtype=str, usecols=1)
    
    mat_pd = pd.DataFrame(awa_rationales_mat,index=classes,columns=attributes)
    
    
    
    #Cross reference classes with labels from ImageNet
    #Fine tune models trained on ImageNet with AwA
    
    imagenet_labels = pd.read_csv(current_dir+'/tcav/tcav_examples/image_models/imagenet/imagenet_url_map.csv',header=0)
    inception_labels = pd.read_csv(current_dir+'/tcav/tcav_examples/image_models/imagenet/downloaded_data/inception5h/imagenet_comp_graph_label_strings.txt',header=0)
    imagenet = imagenet_labels['class_name'].tolist()
    inception = inception_labels['dummy'].tolist()
    
    
    imagenet_classes = []
    for i in imagenet:
        for a in inception : 
            if i == a and i not in imagenet_classes: 
                imagenet_classes.append(i)
                
    awa_classes = []
    for i in mat_pd.index:
        for a in inception : 
            if i == a and i not in awa_classes: 
                awa_classes.append(i)
                
    
    #Cross reference attributes with concepts from Broden             
    source_dir = current_dir+'/tcav/tcav_examples/image_models/imagenet/downloaded_data/broden1_224'
    broden_concepts_by_cat = {}
    broden_concepts = []
    concept_cat = ['scene','object','color','material','part','texture']
    for c in concept_cat:
        b_c = []
        with open(source_dir+'/c_'+c+'.csv') as file_obj: 
            reader_obj = csv.reader(file_obj) 
            for row in reader_obj:  
                if row[0] != 'code':
                    b_c.append(row[2])
                    broden_concepts.append(row[2])
        broden_concepts_by_cat[c] = b_c
        
        
    awa_attributes = []
    for i in mat_pd.columns:
        for a in broden_concepts : 
            #print(a,i)
            if i == a.split('-')[0] and i not in awa_attributes: 
                awa_attributes.append(i)
                
    
    val_class_att = []
    for att in awa_attributes : 
        val_class_att.append(mat_pd[att].tolist())
        
    val_class_att_df = pd.DataFrame(val_class_att,index=awa_attributes,columns=classes).transpose()
    
    
    
    colors = ['black', 'brown', 'white', 'blue', 'orange', 'red', 'yellow']
    objects = ['ground', 'meat', 'tree', 'water', 'bush']
    scenes = ['ocean', 'desert', 'forest']
    part = ['tail'] 
                
    
    #should remove lean+fish, because not same meaning between AwA and broden
    val_class_att_df_2 = val_class_att_df.drop(['lean','fish'],axis=1)
    mat_col_sce = val_class_att_df_2.drop(objects+part,axis=1)
    
    class_att_max = []
    for i in range(len(mat_col_sce.index)):
        class_att_max.append([mat_col_sce.max(axis=1).index[i],mat_col_sce.max(axis=1)[i],mat_col_sce.idxmax(axis=1)[i]])
        
    class_att_max_df = pd.DataFrame(class_att_max,columns=['class','asso_strength','concept'])
    class_att_max_df_sorted = class_att_max_df.sort_values(['concept','asso_strength'])
    #class_att_max_df_sorted.to_excel('./viz/class_att_max_df_sorted.xlsx')
    
    
    return mat_col_sce


def get_asso_strength(label, concept):
    
    awa_rationales_mat = awa_rationales()
    return awa_rationales_mat.loc[label][concept]
    
    