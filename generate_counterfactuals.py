# Simple script to demonstrate CCE
import os
import pickle
import argparse
import torch
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from PIL import Image
from scipy.stats import rankdata
import ast


from model_utils import get_model, ResNetBottom, ResNetTop, GoogLeNetBottom, GoogLeNetTop, InceptionV3Bottom, InceptionV3Top
from model_utils import imagenet_transforms as preprocess
from model_utils import jj as jj
from concept_utils import conceptual_counterfactual, ConceptBank

global logits, expl
logits, expl = [], []


def config(model):
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", default="./examples/models/dog(snow).pth", type=str)
    
    parser.add_argument("--concept-bank", default="./examples/CAVs/"+model+"_1.0_50.pkl", type=str)
    parser.add_argument("--device", default="cpu", type=str)
    parser.add_argument("--seed", default=42, type=int, help="Random seed")
    parser.add_argument("--image-folder", default="../tcav/tcav/tcav_examples/image_models/imagenet/data/", type=str)
    parser.add_argument("--explanation-folder", default="./examples/explanations/"+model+"/", type=str)
    parser.add_argument("--model_name", default=model)
    return parser.parse_args()


def viz_explanation(image, explanation, class_to_idx):
    fig, ax = plt.subplots(1, 1, figsize=(7, 7))
    exp_text = [f"Label: {class_to_idx[explanation.label]}"]
    exp_text.append(f"Prediction: {class_to_idx[explanation.prediction]}")
    exp_text.extend([f"{c}: {explanation.concept_scores[c]:.2f}" for c in explanation.concept_scores_list[:2]])
    exp_text.extend([f"{c}: {explanation.concept_scores[c]:.2f}" for c in explanation.concept_scores_list[-2:]])
    exp_text = "\n".join(exp_text)
    ax.imshow(image)
    props = dict(boxstyle='round', facecolor='salmon', alpha=0.9)
    ax.axis("off")
    ax.text(0, 1.0, exp_text,
            horizontalalignment='left',
            verticalalignment='top',
            transform=ax.transAxes,
            fontsize=10,
            bbox=props)
    fig.tight_layout()
    return fig
 
   
    
def main(args, model_name, targets, concepts):
    sns.set_context("poster")
    np.random.seed(args.seed)
    #model_name = 'inceptionV3'
    
    """targets = ['skunk','zebra','dalmatian','tiger','hippopotamus','leopard','lion','gorilla',
               'ox','chimpanzee','hamster','weasel','otter','mouse','collie','beaver']"""
    
    # Load the model and Split the model into the backbone and the predictor layer
    if model_name == "googlenet":
        model = get_model(args, get_full_model=True)[0]
        backbone, model_top = GoogLeNetBottom(model), GoogLeNetTop(model)
    elif model_name == "resnet18":
        model = get_model(args, get_full_model=True)[0]
        backbone, model_top = ResNetBottom(model), ResNetTop(model)
    elif model_name == "inceptionv3":
        model = get_model(args, get_full_model=True)[0]
        backbone, model_top = InceptionV3Bottom(model), InceptionV3Top(model)
    else:
        raise ValueError(model_name)
    
    #model = torch.load(args.model_path)
    model = model.to(args.device)
    model = model.eval()
    
    # TODO: Class indices are here, should be adapted based on training dataset labeling/ model output layer index-label matching
    global idx_to_class, cls_to_idx
    #idx_to_class = {0: "bear", 1: "bird", 2: "cat", 3: "dog", 4: "elephant"}
    
    #imagenet as training dataset
    idx_to_class = ast.literal_eval(open('./examples/models/imagenet1k_idx_to_label.txt','r').read())
    #idx_to_class = np.load('./examples/models/imagenet1k_idx_to_label.txt',allow_pickle=True)
    cls_to_idx = {v: k for k, v in idx_to_class.items()}
    
   
    
    # Load the concept bank
    concept_bank = ConceptBank(pickle.load(open(args.concept_bank, "rb")), device=args.device)

    os.makedirs(args.explanation_folder, exist_ok=True)
    
    spearman_scores = {}
    expl = {}
    
    for target in targets:#os.listdir(args.image_folder):
        expl[target] = []
        # Read the image and label
        for image_path in os.listdir(args.image_folder+target):
            image = Image.open(os.path.join(args.image_folder, target, image_path)).convert('RGB')
            image_tensor = preprocess(model_name)(image).to(args.device)
        
            cl = target#image_path.split("_")[0]
        
            """if cl != 'cat':
                continue"""
            label = cls_to_idx[cl]*torch.ones(1, dtype=torch.long).to(args.device)
            
            # Get the embedding for the image
            embedding = backbone(image_tensor.unsqueeze(0))
            # Get the model prediction
            pred = model_top(embedding).argmax(dim=1)
            logits.append(model_top(embedding))

            
            
            # Only evaluate over mistakes
            if pred.item() == label.item():
                print(f"Warning: {image_path} is correctly classified, but we'll still try to increase the confidence if you really want that.")
            
            #else:
            # Get the embedding for the image
            embedding = backbone(image_tensor.unsqueeze(0)).detach()
            # Run CCE
            explanation = conceptual_counterfactual(embedding, label, concept_bank, model_top) 
            #print("__________________________", explanation)
            
            # Get explanations scores vector
            expl[target].append(explanation)
            
            # Visualize the explanation, and save it to a figure
            fig = viz_explanation(image, explanation, idx_to_class) 
            if not os.path.exists(os.path.join(args.explanation_folder,target)):
                os.makedirs(os.path.join(args.explanation_folder,target))
            fig.savefig(os.path.join(args.explanation_folder,target, f"{image_path.split('.')[0]}_explanation.png"))
        
            print("_____________________________________________")

   
    with open(args.explanation_folder+'/results.pkl', 'wb') as fp:
        pickle.dump(expl, fp)
        print('dictionary saved successfully to file')
        
    return expl
        
        
""" pred_v, exp_v = [], []
    
    for i in range(len(logits)):
        pred_v.append([])
        for j in range(0,5):
            pred_v[i].append(logits[i][0][j].item())
    pred = [item[label.item()] for item in pred_v]
            
        
    concepts_list = ['blackness','blueness','redness']
        
    spearman_scores[cl] = {}
    for c in concepts_list:
        exp_v = []
        for i in range(len(expl)):
            exp_v.append(expl[i][c])
    
        R_exp = rankdata(exp_v)
        R_pred = rankdata(pred)
        
        spearman_scores[cl][c] = np.corrcoef(R_exp, R_pred)[0][1]
        
    return spearman_scores
            

def prepare_vects_for_spearman(logits_vec, expl_vec, target, concept):
    pred, exp = [], []
    
    for i in range(len(logits_vec)):
        pred.append([])
        for j in range(1,4):
            pred[i].append(logits_vec[i][0][j].item())
            
    for i in range(len(expl_vec)):
        exp.append(expl_vec[i][concept])

            
    return [item[cls_to_idx[target]] for item in pred], exp

# Computing Spearman's rank correlation coefficient between the sensitivity and the predictio scores
def compute_spearmans_rank(logits_vec, expl_vec, target, concept):
    pred, exp = prepare_vects_for_spearman(logits_vec, expl_vec, target, concept)
    
    R_exp = rankdata(exp)
    R_pred = rankdata(pred)
    
    return np.corrcoef(R_exp, R_pred)"""
        
"""if __name__ == "__main__":
    model_name = 'inceptionv3'
    args = config(model_name)
    main(args, model_name)"""
    
def run_cce(model_name,targets, concepts):
    args = config(model_name)
    main(args, model_name, targets,concepts)
    