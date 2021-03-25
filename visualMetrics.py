import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import ElasticSearchSimpleAPI as es
import TRECCASTeval as trec
import os
import sys
import pickle


class GraphGenerator(object):
    ''' Simple Class for generating metric plots to evaluate natural language models'''

    def __init__(self):
        self.precRec = {}

    def ap(self, file):
        data = pd.read_csv(file)
        if (data.shape[0] > 8):
            data = data.iloc[:8]
            data['Conversation Turn'] = range(1, data.shape[0]+1)

        data.plot.scatter(x='ap', y='Conversation Turn', title=f'Per turn score for AP', grid=True)
        #plt.savefig(f'/content/drive/My Drive/ProjectoRI2020/results/{conv_id}_ap.png')

    def ndgc5(self, file):
        data = pd.read_csv(file)
        if (data.shape[0] > 8):
            data = data.iloc[:8]
            data['Conversation Turn'] = range(1, data.shape[0]+1)

        data.plot(x='Conversation Turn', y='ap', title=f'Per turn score for p10')
        #plt.savefig(f'/content/drive/My Drive/ProjectoRI2020/results/{conv_id}_ndcg5.png')
    
    def compareAp(self, conversation_turn):
        data_lmd = pd.read_csv(f'.\\results\\stats_{conversation_turn}.csv')
        data_bert = pd.read_csv(f'.\\results\\stats_{conversation_turn}_BERT.csv')
        if (data_lmd.shape[0] > 8):
            data_lmd = data_lmd.iloc[:8]
        data_lmd['Conversation Turn'] = range(1, data_lmd.shape[0]+1)

        if (data_bert.shape[0] > 8):
            data_bert = data_bert.iloc[:8]
        data_bert['Conversation Turn'] = range(1, data_bert.shape[0]+1)

        ax1 = data_lmd.plot.scatter(x='ap', y='Conversation Turn', title=f'Per turn score for AP', grid=True, c='DarkBlue')
        data_bert.plot.scatter(x='ap', y='Conversation Turn', title=f'Per turn score for AP', grid=True, ax=ax1, c='r')
    
        
        #plt.savefig(f'/content/drive/My Drive/ProjectoRI2020/results/{conv_id}_ap.png')
    
    def plot_precisionRecall_topic_turn_id(self, topic_id, turn_id, models):
        '''Generate a precision-recall plot for a given topic_turn_id, in model.
        
        Input:
          * topic_turn_id (string): topic turn id to evaluate (e.g. "1_1").
          * model (string): model to get metrics from (e.g. "LMD"). A get_PrecisionRecall()
            should've already be ran for model. 
        '''
   
        # Get precision  for current turn_conv_id
        cols = [str(round(x,1)) for x in np.arange(0,1.1,0.1)]
        recall = np.arange(0,1.1,0.1).reshape(-1,1)
        colrs = ["b","g","r","c","m","y","k","gray"]
        
        max_prec = -1
        for model in models:
            modl = self.precRec.get(model)
            precision = modl.loc[((modl["topic_id"] == topic_id) & (modl["turn_id"] == turn_id))][cols].to_numpy().reshape(-1,1)
            if precision.shape[0]==0:
                return None
            plt.plot(recall, precision, colrs.pop(0), label=model)
            if max(precision) > max_prec:
                max_prec = max(precision)
        
        # Plot precision-recall for current turn 
        plt.xticks(np.arange(0, 1.1, 0.1))
        plt.xlim(0,1)
        plt.ylim(0,max_prec*1.1)
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title("Precision-Recall: " + topic_id+'_'+turn_id+"("+"LMD|BERT"+")") 
        plt.grid()
        plt.legend()
        plt.show()
        plt.savefig("prec_rec_" + topic_id+'_'+turn_id+'_'+'BERT'+'_'+'LMD.png')
        
        
    def plot_precisionRecall_avgTopic(self, topic_id, models):
        '''Generate a precision-recall plot for the average of all turns in a given
           topic, in model.
        
        Input:
          * topic_id (string): topic id to evaluate (e.g. "1").
          * model (string): model to get metrics from (e.g. "LMD"). A get_PrecisionRecall()
            should've already be ran for model. 
        '''

        # Get precision  for current turn_conv_id
        cols = [str(round(x,1)) for x in np.arange(0,1.1,0.1)]
        recall = np.arange(0,1.1,0.1).reshape(-1,1)
        colrs = ["b","g","r","c","m","y","k","gray"]
        
        max_prec = -1
        for model in models:
            modl = self.precRec.get(model)
            
            if model == 'BERT-enteties-regOri':
                model = 'BERT-entities-regOri'
            elif model == 'LMD-enteties':
                model = 'LMD-entities'
            
            precision = modl.loc[modl["topic_id"] == topic_id][cols]
            if precision.shape[0]==0:
                return None
            precision_means = precision.mean().to_numpy().reshape(-1,1)
            plt.plot(recall, precision_means, colrs.pop(0), label=model)
            if max(precision_means) > max_prec:
                max_prec = max(precision_means)
        
        
        # Plot precision-recall for current turn 
        plt.xticks(np.arange(0, 1.1, 0.1))
        plt.xlim(0,1)
        plt.ylim(0,max_prec*1.1)
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title("Precision-Recall: Conversation " + topic_id) 
        plt.grid()
        plt.legend()
        plt.savefig("prec_rec_" + topic_id+'_'+'BERT'+'_'+'LMD.png')
        plt.show()
        
        
        
        
        
        
    def plot_precisionRecall_avgModel_byConv(self, models):
        '''Generate a precision-recall plot for the average of all conversations
           in a given model or models.
        
        Input:
          * models (array of string): models to get metrics from (e.g. ["LMD", "RMS"]).
            A get_PrecisionRecall() should've already be ran for all models in the array    
        '''
        
        # Get precision  for current turn_conv_id
        cols = [str(round(x,1)) for x in np.arange(0,1.1,0.1)]
        recall = np.arange(0,1.1,0.1).reshape(-1,1)
        colrs = ["b","g","r","c","m","y","k","gray"]
        
        max_prec = -1
        for model in models:
            modl = self.precRec.get(model)
            
            if model == 'BERT-enteties-regOri':
                model = 'BERT-entities-regOri'
            elif model == 'LMD_enteties':
                model = 'LMD_entities'
            
            precision_means = modl[cols].mean().to_numpy().reshape(-1,1)
            if precision_means.shape[0]==0:
                return None
            else: 
                plt.plot(recall, precision_means, colrs.pop(0), label=model)
                if max(precision_means) > max_prec:
                    max_prec = max(precision_means)
          
        # Plot precision-recall for current turn 
        plt.xticks(np.arange(0, 1.1, 0.1))
        plt.xlim(0,1)
        plt.ylim(0,max_prec*1.1)
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title("Precision-Recall: Average by conversation")   
        plt.grid()
        plt.legend()
        plt.savefig("prec_rec_AVG.png")
        plt.show()
        
    
    
    def get_PrecisionRecall(self, relevance_judgments, conversations):
        ''' Gets the precision-recall values for each topic_turn_id, and for each 
            model in conversations.
        
        Input:
          * conversations (dictionary of array of string): Each key is a different
            model, and each value is an array with all the topic_turn_id's to eval.
            (e.g. conversations = {"LMD": [1_1, 1_2, 1_3, 5_1,7_1]} )  
            
        Return:
          * precRec (dictionary of pd.DataFrame): Each key is a different model, and 
            the value is a pd.DataFrame with the precision values for recalls spaced
            in intervals of 0.1 (from 0 to 1):
                
              topic_id | turn_id  | 0.1  | 0.2  | 0.3  | ...  | 1.0
            0    1     |     1    |  0.4 | 0.35 | 0.2  |  ... |  0.089
            1    1     |     2    |  0.8 | 0.65 | 0.54 |  ... |  0.2 
        '''
        
        # Info to create precision recall dataframe
        recalls = [str(round(x,1)) for x in np.arange(0,1.1,0.1)]     
        columns = ["topic_id", "turn_id"] + recalls
        for model, turns in conversations.items(): 
            
            # Create current model precisionRecall dataframe
            k = 0  # index of item in precisionRecall
            numItems = len(turns)  # total number of turns in current model
            self.precRec[model] = pd.DataFrame(data=np.zeros([numItems,13]), columns=columns)        
            for model_topic_turn_id in turns:
                # Load data, and set turn and conv_ ids
                res_data = pd.read_csv("./results/"+model_topic_turn_id+".csv")
                (model_id, conv_id, turn_id) = model_topic_turn_id.split("_")
                topic_turn_id = conv_id + "_" + turn_id
                  
                if model_id == 'BERT-1stTurn-regOri' or \
                   model_id == 'BERT-enteties-regOri' or \
                   model_id == 'BERT-raw-regOri' or \
                   model_id == 'BERT-T5-regOri':
                       res_data = res_data.sort_values(['logistic regression'], ascending = [False])
                
                # Locate relevant documents in current turn
                GT_docs = relevance_judgments.loc[relevance_judgments["topic_turn_id"] == (topic_turn_id)]
            
                # Compute precision and recall for current curve
                _precision = []
                _recall = []   
                numRel = GT_docs.loc[GT_docs["rel"] != 0]["docid"].count()  # Total number of relevant docs
                relCount = 0                                                # Number of relevant docs found (counter)
                docCount = 0                                                # Number of docs (counter)
                if res_data.size != 0:
                    for doc in res_data["_id"]:
                    
                        # If doc is in GT_truth &&  rel_doc <> 0, then compute prec and recall
                        docCount += 1
                        if doc in GT_docs["docid"].tolist():
                        
                            index_doc = GT_docs.index[GT_docs["docid"] == doc][0] # (index of doc in GT_docs)
                            if GT_docs.at[index_doc, "rel"] != 0:
                                relCount += 1
                                _precision.append(relCount/docCount)
                                _recall.append(relCount/numRel)

                          
                # Transform these precisions into intervals of 0.1 recalls
                precision = np.zeros(11)
                recall = np.arange(0,1.1,0.1)
                for i in range(len(recall)):
                    _y = Intersect_Point_Line(recall[i], _recall, _precision)
                    precision[i] = _y
       
                # Fill precisionRecall dataframe
                self.precRec[model].loc[k, "topic_id"] = conv_id
                self.precRec[model].loc[k, "turn_id"] = turn_id
                self.precRec[model].loc[k, recalls] = precision 
                k += 1
                

def get_recl_index(_recl, recl11):
    for i in range(len(_recl)):
        if recl11 <= _recl[i]:
            return i
        
    
        
      
def get_Conversation_Results(model_id, conversations = []):
    
    # Currently implemented in ./results/
    res_fileNames = []
    for filename in os.listdir("./results/"):
        if filename == 'BERT-raw-regOri_31_1.csv':
            k = 0
        conv_turn_id = os.path.splitext(filename)[0]
        try:
            (model, conv, turn) = conv_turn_id.split("_")
            if int(conv) in conversations and model == model_id:
                res_fileNames.append(conv_turn_id)
        except:
            continue
        
    return {model_id: res_fileNames}
    
def Intersect_Point_Line(x, x_curve, y_curve):
    
    # find "a" and "b"
    a = 0 # default
    b = 0 
    for i in range(len(x_curve)):
        if i == 0:
            if x <= x_curve[i]:
                a = 0
                b = y_curve[i]
        else:
            if (x <= x_curve[i]) and (x >= x_curve[i-1]):
                a = (y_curve[i]-y_curve[i-1])/(x_curve[i]-x_curve[i-1])
                b = y_curve[i]-a*x_curve[i]
    return a*x+b