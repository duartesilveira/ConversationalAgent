'''
Functions to generate results from different retrieval models
'''

import TRECCASTeval as trec
import ElasticSearchSimpleAPI as es
import numpy as np
import pandas as pd


def get_LMD_Results(isTest = True, numDocs = 1000, filePath= None):
    elastic = es.ESSimpleAPI()
    test_bed = trec.ConvSearchEvaluation()
    
    # Get test bed topics 
    if isTest:
        test_bed_topics = test_bed.test_topics
    else:
        test_bed_topics = test_bed.train_topics
        
    for topic in test_bed_topics:
        
        conv_id = topic['number']
        if isTest:
            if conv_id not in (31, 32, 33, 34, 37, 40, 49, 50, 54, 56, 58, 59, 61, 67, 68, 69, 75, 77, 78, 79):
                continue
        else:
            if conv_id not in (1, 2, 4, 7, 15, 17,18,22,23,24,25,27,30):
                continue
         
        stats = pd.DataFrame(columns=['p10', 'recall', 'ap', 'ndcg5'])
         
        for turn in topic['turn']:
            turn_id = turn['number']
            utterance = turn['raw_utterance']
            topic_turn_id = '%d_%d'% (conv_id, turn_id)
            
            # Get number of relevant docs
            if isTest:
                aux = test_bed.test_relevance_judgments.loc[test_bed.test_relevance_judgments['topic_turn_id'] == (topic_turn_id)]
                num_rel = aux.loc[aux['rel'] != 0]['docid'].count()
            else:
                aux = test_bed.relevance_judgments.loc[test_bed.relevance_judgments['topic_turn_id'] == (topic_turn_id)]
                num_rel = aux.loc[aux['rel'] != 0]['docid'].count()
            
            # Get results from LMD
            if num_rel == 0:
                continue
            result = elastic.search_body(query=utterance, numDocs = numDocs)
            if(filePath):
              result.to_csv(filePath)
            else:
              fileName = "LMD_"+topic_turn_id+".csv" 
              result.to_csv("./Results/" + fileName)
            
            # Create stats files
            p10 = 0
            recall = 0
            ap = 0
            ndcg5 = 0
            if np.size(result) != 0:
                [p10, recall, ap, ndcg5] = test_bed.eval(result[['_id','_score']], topic_turn_id)

            stats = stats.append({'p10': p10, 'recall': recall, 'ap': ap, 'ndcg5': ndcg5}, ignore_index=True)

        stats.to_csv('./results/'+'stats_'+str(conv_id)+'.csv')
            