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



broden_path = 'C:/Users/rielcheikh/Desktop/XAI/tcav/tcav/tcav_examples/image_models/imagenet/data'
dest_path = 'C:/Users/rielcheikh/Desktop/XAI/cce/concept_utils/broden_img'

concepts_to_make = ['ocean-s', 'desert-s', 'forest-s','black-c', 'brown-c', 'white-c', 'blue-c', 'orange-c', 'red-c', 'yellow-c']

random_i = 0

if not os.path.exists(dest_path):
    os.makedirs(dest_path)


for concept in concepts_to_make:
    
    if not os.path.exists(dest_path+'/'+concept+ "/positives/"):
        os.makedirs(dest_path+'/'+concept+ "/positives/")

    for (root,dirs,files) in os.walk(broden_path):
        if (root.split("\\")[-1] == concept):
            i = 1
            for file in files:
                 with Image.open(broden_path+'/'+concept+'/'+file) as im:
                     im.save(dest_path+'/'+concept + "/positives/"+str(i)+".jpeg")
                     i+=1
                        
                    
                    
    if not os.path.exists(dest_path+'/'+concept+ "/negatives/"):
        os.makedirs(dest_path+'/'+concept+ "/negatives/")

    for (root,dirs,files) in os.walk(broden_path):
        if (root.split("\\")[-1] == 'random500_'+str(random_i)):
            i = 1
            for file in files:
                 with Image.open(broden_path+'/random500_'+str(random_i)+'/'+file).convert('RGB') as im:
                     im.save(dest_path+'/'+concept + "/negatives/"+str(i)+".jpeg")
                     i+=1
    random_i += 1    
                        
