import os
import pickle
import argparse
import torch
import numpy as np

from glob import glob
from tqdm import tqdm
from torch.utils.data import DataLoader

from cce.concept_utils import learn_concept_bank, ListDataset
from cce.model_utils import get_model

## run learn_concepts.py --concept_dir=C:/Users/rielcheikh/Desktop/XAI/cce/concept_utils/broden_img --model_name=resnet18 --C=0.001 --out_dir=examples/CAVs


import ssl
ssl._create_default_https_context = ssl._create_unverified_context
#https://stackoverflow.com/questions/35569042/ssl-certificate-verify-failed-with-python3#answer-49174340


def config():
    parser = argparse.ArgumentParser()
    parser.add_argument("--concept_dir", type=str, default='C:/Users/rielcheikh/Desktop/XAI/cce/concept_utils/broden_img',
                        help="Directory containing concept images. See below for a detailed description.")
    
    parser.add_argument("--out_dir", default="./examples/CAVs", type=str,
                        help="Where to save the concept bank.")
   
    parser.add_argument("--model_name", default="resnet_101", type=str, help="Name of the model to use.")
    parser.add_argument("--device", default="cpu", type=str)
    
    
    parser.add_argument("--batch_size", default=32, type=int)
    parser.add_argument("--num_workers", default=2, type=int)
    parser.add_argument("--seed", default=42, type=int, help="Random seed")
    parser.add_argument("--C", nargs="+", default=[1e-5, 1e-4, 0.001, 0.01, 0.1, 1.0], type=float,  
                        help="Regularization parameter for SVMs. Can specify multiple values.")
    
    parser.add_argument("--n_samples", default=200, type=int, 
                        help="Number of pairs of positive/negative samples used to train SVMs.")
    return parser.parse_args()


def learn_concepts(concepts_dir, res_dir, model_name, alphas, n_samples=50, seed=42, device='cpu', batch_size=32, num_workers=1):
    np.random.seed(seed)
    
    # Concept images are expected in the following format:
    # args.concept_dir/concept_name/positives/1.jpg, args.concept_dir/concept_name/positives/2.jpg, ...
    # args.concept_dir/concept_name/negatives/1.jpg, args.concept_dir/concept_name/negatives/2.jpg, ...
    
    concept_names = os.listdir(concepts_dir)
    
    # Get the backbone
    backbone, _, preprocess = get_model(model_name, device)
    backbone = backbone.to(device)
    backbone = backbone.eval()
    
    print(f"Attempting to learn {len(concept_names)} concepts.")
    concept_lib = {C: {} for C in alphas}
    for concept in concept_names:
        pos_ims = glob(os.path.join(concepts_dir, concept, "positives", "*"))
        neg_ims = glob(os.path.join(concepts_dir, concept, "negatives", "*"))

        pos_dataset = ListDataset(pos_ims, preprocess=preprocess)
        neg_dataset = ListDataset(neg_ims, preprocess=preprocess)
        print(len(pos_dataset), len(neg_dataset))
        if (len(pos_dataset) > 0 and len(neg_dataset) >0 ):
            pos_loader = torch.utils.data.DataLoader(pos_dataset, batch_size=batch_size, shuffle=False)#, num_workers=num_workers)
            neg_loader = torch.utils.data.DataLoader(neg_dataset, batch_size=batch_size, shuffle=False)#, num_workers=num_workers)
            
            """with open(args.model_name+'loader.pkl', 'wb') as fp:
                pickle.dump(pos_loader, fp)
                print('dictionary saved successfully to file')
           
            with open(args.model_name+'loader.pkl', 'rb') as fp:
                hh = pickle.load(fp)"""
            
            """backbone = get_model(args, get_full_model=True)[0]
            backbone.fc = torch.nn.Identity()"""
            cav_info = learn_concept_bank(pos_loader, neg_loader, backbone, n_samples, alphas, device=device)
            # Store CAV train acc, val acc, margin info for each regularization parameter and each concept
            for C in alphas:
                concept_lib[C][concept] = cav_info[C]
                print(f"{concept} with C={C}: Training Accuracy: {cav_info[C][1]:.2f}, Validation Accuracy: {cav_info[C][2]:.2f}")
    
    # Save CAV results 
    os.makedirs(res_dir, exist_ok=True)
    for C in concept_lib.keys():
        lib_path = os.path.join(res_dir, f"{model_name}_{C}.pkl")
        with open(lib_path, "wb") as f:
            pickle.dump(concept_lib[C], f)
        print(f"Saved to: {lib_path}")        
    


