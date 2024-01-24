import pickle


def prep_cce_res(res_dir, targets, concepts, topk):
   
    with open(res_dir+'/explanations/results.pkl', 'rb') as f:
        cce_scores = pickle.load(f)

    #build class-wise cce scores 
    class_cce_scores = {}
    topk_class_cce_score = {}
    
    for t in targets:
        avg_concept = {}
        topk_class_cce_score[t] = {}
        class_cce_scores[t] = {}
        for c in concepts:
            avg_concept[c] = []
            #class_cce_scores[t] = {}
            for i in range(len(cce_scores[t])):
                avg_concept[c].append(cce_scores[t][i]['concept_scores'][c])
        
            class_cce_scores[t][c] = sum(avg_concept[c])/len(avg_concept[c])
            
            
        topk_concepts = sorted(class_cce_scores[t],key=class_cce_scores[t].__getitem__, reverse=True)[:topk]
        for c in topk_concepts:
            topk_class_cce_score[t][c] = class_cce_scores[t][c]
                
    return topk_class_cce_score, class_cce_scores