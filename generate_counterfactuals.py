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


from cce.model_utils import get_model, ResNetBottom, ResNetTop, GoogLeNetBottom, GoogLeNetTop, InceptionV3Bottom, InceptionV3Top, VGG16Bottom, VGG16Top
from cce.model_utils import imagenet_transforms as preprocess
from cce.model_utils import jj as jj
from cce.concept_utils import conceptual_counterfactual, ConceptBank
from cce.learn_concepts import learn_concepts

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
 
   
    
def main(targets, concept, dataset, concept_dataset, bottleneck, model_name, res_dir, data_dir, num_random_exp, alphas, model_cav, seed=42, device='cpu',):
    sns.set_context("poster")
    np.random.seed(seed)
    
    
    if not os.path.exists(res_dir):
        os.makedirs(res_dir)

    
    # Load the model and Split the model into the backbone and the predictor layer
    if model_name == "googlenet":
        model = get_model(model_name, device = 'cpu', get_full_model=True)[0]
        backbone, model_top = GoogLeNetBottom(model), GoogLeNetTop(model)
    elif model_name == "resnet_101":
        model = get_model(model_name, device = 'cpu', get_full_model=True)[0]
        backbone, model_top = ResNetBottom(model), ResNetTop(model)
    elif model_name == "inceptionv3":
        model = get_model(model_name, device = 'cpu', get_full_model=True)[0]
        backbone, model_top = InceptionV3Bottom(model), InceptionV3Top(model)
    elif model_name == "vgg_16":
        model = get_model(model_name, device = 'cpu', get_full_model=True)[0]
        backbone, model_top = VGG16Bottom(model), VGG16Top(model)
    else:
        raise ValueError(model_name)
    
    #model = torch.load(args.model_path)
    model = model.to(device)
    model = model.eval()
    
    # TODO: Class indices are here, should be adapted based on training dataset labeling/ model output layer index-label matching
    global idx_to_class, cls_to_idx
    #idx_to_class = {0: "bear", 1: "bird", 2: "cat", 3: "dog", 4: "elephant"}
    
    #imagenet as training dataset
    #TODO
    idx_to_class = ast.literal_eval(open('./cce/examples/models/imagenet1k_idx_to_label.txt','r').read())
    
    for k, v in idx_to_class.items():
        idx_to_class[k] = v.split(',')[0]
    #idx_to_class = np.load('./examples/models/imagenet1k_idx_to_label.txt',allow_pickle=True)
    cls_to_idx = {v: k for k, v in idx_to_class.items()}
    
    
    
   
    
    # Load the concept bank
    #TODO : get concept dataset like TCAV
    if not os.path.exists(res_dir+'/CAVs/'+model_name+'_'+str(alphas[0])+'.pkl'):
        learn_concepts(data_dir+'/cce_concepts/'+concept_dataset, res_dir+'/CAVs/', model_name, alphas)
    concept_bank = ConceptBank(pickle.load(open(res_dir+'/CAVs/'+model_name+'_'+str(alphas[0])+'.pkl', "rb")), device=device)

    
    spearman_scores = {}
    expl = {}
    
    for target in targets:#os.listdir(data_dir):
        expl[target] = []
        # Read the image and label
        for image_path in os.listdir(data_dir+'imgs/'+target):
            image = Image.open(os.path.join(data_dir,'imgs/', target, image_path)).convert('RGB')
            image_tensor = preprocess(model_name.split('_')[0])(image).to(device)
        
            cl = target#image_path.split("_")[0]
        
            """if cl != 'cat':
                continue"""
            
            #return self.labels.index(label)
            try : 
                label = cls_to_idx[cl]*torch.ones(1, dtype=torch.long).to(device)
            
            except:
                with open('./data/dict_apy_imagenet_classes.pkl','rb') as f:
                    corr = pickle.load(f)
                for k in corr.keys():
                    if cl == k:
                        label = []
                        for i in corr[k]:
                            label.append(cls_to_idx[i]*torch.ones(1, dtype=torch.long).to(device))
            
            
            # Get the embedding for the image
            embedding = backbone(image_tensor.unsqueeze(0))
            # Get the model prediction
            pred = model_top(embedding).argmax(dim=1)
            logits.append(model_top(embedding))

            
            
            # Only evaluate over mistakes
            if type(label) == list:
                for l in label:
                    if pred.item() == l.item():
                        print(f"Warning: {image_path} is correctly classified, but we'll still try to increase the confidence if you really want that.")
                    
                    # Get the embedding for the image
                    embedding = backbone(image_tensor.unsqueeze(0)).detach()
                    # Run CCE
                    explanation = conceptual_counterfactual(embedding, l, concept_bank, model_top) 
                    #print("__________________________", explanation)
                    
                # Get explanations scores vector
                expl[target].append(explanation)
                
                # Visualize the explanation, and save it to a figure
                fig = viz_explanation(image, explanation, idx_to_class) 
                if not os.path.exists(os.path.join(res_dir,'explanations',target)):
                    os.makedirs(os.path.join(res_dir,'explanations',target))
                fig.savefig(os.path.join(res_dir,'explanations',target, f"{image_path.split('.')[0]}_explanation.png"))
            
                print("_____________________________________________")
   
            
       
        else:
                if pred.item() == label.item():
                    print(f"Warning: {image_path} is correctly classified, but we'll still try to increase the confidence if you really want that.")
           
                
                # Get the embedding for the image
                embedding = backbone(image_tensor.unsqueeze(0)).detach()
                # Run CCE
                explanation = conceptual_counterfactual(embedding, label, concept_bank, model_top) 
                #print("__________________________", explanation)
                
                # Get explanations scores vector
                expl[target].append(explanation)
                
                # Visualize the explanation, and save it to a figure
                fig = viz_explanation(image, explanation, idx_to_class) 
                if not os.path.exists(os.path.join(res_dir,'explanations',target)):
                    os.makedirs(os.path.join(res_dir,'explanations',target))
                fig.savefig(os.path.join(res_dir,'explanations',target, f"{image_path.split('.')[0]}_explanation.png"))
            
                print("_____________________________________________")

   
    with open(res_dir+'/explanations/'+'/results.pkl', 'wb') as fp:
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
    
def run_cce(targets, concepts, dataset, concept_dataset, bottleneck, model_name, res_dir, data_dir, num_random_exp, alphas, model_cav):
    #args = config(model_name)
    main(targets, concepts, dataset, concept_dataset, bottleneck, model_name, res_dir, data_dir, num_random_exp, alphas, model_cav)
    