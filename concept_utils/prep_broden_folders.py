# -*- coding: utf-8 -*-
"""
Created on Wed Dec 13 15:18:59 2023

@author: rielcheikh
"""
import os
from PIL import Image


# Concept images are expected in the following format:
# args.concept_dir/concept_name/positives/1.jpg, args.concept_dir/concept_name/positives/2.jpg, ...
# args.concept_dir/concept_name/negatives/1.jpg, args.concept_dir/concept_name/negatives/2.jpg, ...

rationales_dataset = 'awa'

#broden_path = 'C:/Users/rielcheikh/Desktop/XAI/XAI-eval/data/imgs'
concepts_path = 'C:/Users/rielcheikh/Desktop/XAI/XAI-Eval/data/imgs'
random_path = 'C:/Users/rielcheikh/Desktop/XAI/XAI-eval/data/imgs'
dest_path = 'C:/Users/rielcheikh/Desktop/XAI/XAI-eval/data/cce_concepts/'+rationales_dataset


    #to fix : leaf, text, leg, ear, screen, metal, plastic, leather, beak, head, nose, eye, torso, hand, arm
    #check if same meaning : wing,
concepts_to_make = ['tail', 'mouth', 'hair', 'face', 'wing', 
                    'wheel', 'door', 'headlight', 'taillight', 'engine', 'horn', 'saddle', 'flower', 'pot', 
                    'skin', 'wood', 'glass']
concepts_to_make = os.listdir(concepts_path)

concepts_to_make = ['ocean-s', 'desert-s', 'forest-s', 'water-s', 'cave-s', 'black-c', 'brown-c', 'white-c', 'blue-c', 'orange-c', 'red-c', 'yellow-c']



random_i = 0

if not os.path.exists(dest_path):
    os.makedirs(dest_path)
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

for concept in concepts_to_make:
    
    if not os.path.exists(dest_path+'/'+concept+ "/positives/"):
        os.makedirs(dest_path+'/'+concept+ "/positives/")

    for (root,dirs,files) in os.walk(concepts_path):
        if (root.split("\\")[-1] == concept):
            i = 1
            for file in files:
                 with Image.open(concepts_path+'/'+concept+'/'+file) as im:
                     im.convert('RGB').save(dest_path+'/'+concept + "/positives/"+str(i)+".jpeg")
                     i+=1
                        
                    
                    
    if not os.path.exists(dest_path+'/'+concept+ "/negatives/"):
        os.makedirs(dest_path+'/'+concept+ "/negatives/")

    for (root,dirs,files) in os.walk(random_path):
        if (root.split("\\")[-1] == 'random500_'+str(random_i)):
            i = 1
            for file in files:
                try:
                     with Image.open(random_path+'/random500_'+str(random_i)+'/'+file).convert('RGB') as im:
                         im.convert('RGB').save(dest_path+'/'+concept + "/negatives/"+str(i)+".jpeg")
                         i+=1
                except:
                    print('error with getting random image : '+ random_path+'/random500_'+str(random_i)+'/'+file)
    random_i += 1    
                        
