import pandas as pd
from visualMetrics import get_Conversation_Results
import TRECCASTeval as trec
import csv
import ElasticSearchSimpleAPI as es

def getPassage(res_data, docid):
    index_doc = res_data.index[res_data["_id"] == docid][0]
    return res_data.at[index_doc, "_source.body"]

def getQuery(res_data, topic_turn):
    index_doc = res_data.index[res_data["topic_turn"] == topic_turn][0]
    return res_data.at[index_doc, "utterance"] 
    

def getTriplets():
    ''' Create triplets (topic_turn | doc | rel)

    Input
      * fileNames (dictionary): Dictionary with the result fileNames. 
        These .csv files have the GT for a given topic_turn.
      * model (string): model to get results from.
    Return
      * triplets (pandas dataframe): Dataframe with triplets.
    '''
    tripl = []
    res = es.ESSimpleAPI()
    topic_turns = pd.read_csv("./results/topic_turn.csv", names=["topic_turn", "utterance"])
    # topic_turns = pd.read_csv("./topic_turn_test.csv", names=["topic_turn", "utterance"])
    with open(".\\data\\training\\train_topics_mod.qrel", "r") as aFile:
    # with open(".\\data\\evaluation\\evaluation_topics_mod.qrel", "r") as aFile:
        for line in aFile:
            stripped_line = line.strip()
            (topic_turn, ig, doc_id, _rel) = stripped_line.split(" ")
            try:
                passage= res.get_doc_body(doc_id)
                query =getQuery(topic_turns,topic_turn)
                rel = 1 if int(_rel) > 0 else 0 
                tripl.append([query, passage, rel])
            except:
                continue       
     
    # Convert to pandas dataframe for easier manipulation             
    return pd.DataFrame(tripl,columns=['Query','Passage','rel']) 





#resultsLMD = get_Conversation_Results() 
triplets = getTriplets()
# triplets.to_csv('./results/'+'triplets_test.csv', index=False)


#triplets = pd.read_csv("./results/triplets.csv")
#topic_turns = pd.read_csv("./results/topic_turn.csv", names=["topic_turn", "utterance"]) # We use this to get the query for a given topic_turn id

'''
with open("evaluation_topics_mod.qrel", "r") as a_file:
  for line in a_file:
    stripped_line = line.strip()
    (ida, nothing, doc_id, rel) = stripped_line.split(" ")
    (corp, ida) = doc_id.split("_")
    getQuery()
 '''
 
'''
test_bed = trec.ConvSearchEvaluation() 
topics={}
for topic in test_bed.test_topics:
    conv_id = topic['number']

    if conv_id not in (31, 32, 33, 34, 37, 40, 49, 50, 54, 56, 58, 59, 61, 67, 68, 69, 75, 77, 78, 79):
        continue

    print()
    print(conv_id, "  ", topic['title'])

    for turn in topic['turn']:
        turn_id = turn['number']
        utterance = turn['raw_utterance']
        topic_turn_id = '%d_%d'% (conv_id, turn_id)
        
        print(topic_turn_id, utterance)
        topics[topic_turn_id] = utterance
'''


