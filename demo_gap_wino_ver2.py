# 6c01266 (HEAD -> gapX, origin/gapX) WIP8 <- google nl gelmeden onceki calisan version
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from six.moves import input
import tensorflow as tf
import coref_model as cm
import util
import pandas as pd 
import nltk
nltk.download("punkt")
from nltk.tokenize import sent_tokenize, word_tokenize

def create_example(text):
  raw_sentences = sent_tokenize(text)
  sentences = [word_tokenize(s) for s in raw_sentences]
  speakers = [["" for _ in sentence] for sentence in sentences]
  return {
    "doc_key": "nw",
    "clusters": [],
    "sentences": sentences,
    "speakers": speakers,
  }

def print_predictions(example):
  words = util.flatten(example["sentences"])
  for cluster in example["predicted_clusters"]:
    print(u"Predicted cluster: {}".format([" ".join(words[m[0]:m[1]+1]) for m in cluster]))

def make_predictions(text, model):
  example = create_example(text)
  tensorized_example = model.tensorize_example(example, is_training=False)
  feed_dict = {i:t for i,t in zip(model.input_tensors, tensorized_example)}
  _, _, _, mention_starts, mention_ends, antecedents, antecedent_scores, head_scores = session.run(model.predictions + [model.head_scores], feed_dict=feed_dict)

  predicted_antecedents = model.get_predicted_antecedents(antecedents, antecedent_scores)

  example["predicted_clusters"], _ = model.get_predicted_clusters(mention_starts, mention_ends, predicted_antecedents)
  example["top_spans"] = zip((int(i) for i in mention_starts), (int(i) for i in mention_ends))
  example["head_scores"] = head_scores.tolist()
  return example


def prepare_data(fname):
    df = pd.read_table(fname)
    return df


def wino_evaluate(example,text,pronoun,profession):
    profession_words = list(filter(lambda x:x.lower() not in ['the','a'],[ q.replace(",","").replace(":","") for q in text.split(" ")]))
    words = util.flatten(example["sentences"])
    result = False 
    for cluster in example["predicted_clusters"]:
        elements = []
        for p in cluster:
            elements = elements + list(range(p[0],p[1]+1))
        elements_string = [ words[x] for x in elements]
        if find_in_arr(elements_string,pronoun) > -1:
            for p in profession_words:
                if find_in_arr(elements_string,p) > -1:
                    result = True
                    break
    return result


if __name__ == "__main__":
  config = util.initialize_from_env()
  model = cm.CorefModel(config)
  evala,evalb = [],[]

  name = sys.argv[1]
  fname = sys.argv[2]
  
  with tf.Session() as session:
    model.restore(session)
    fname = 'gapx-merged-nl-head.tsv'
    df = prepare_data(fname)
    df.fillna('',inplace=True)
    coref_results = {}

    for index,row in df.iterrows():      
        print("index:",index)
        source_text = row['source_sentence']
        target_text = row['target_sentence']
        source_p  = row['orjinal_pronoun']
        target_p  = row['bt_pronoun']
        source_profession = row['orjinal_profession']
        target_profession = row['bt_profession']      
        example = make_predictions(source_text,model)
        result_source  = wino_evaluate(example,source_text,source_p,source_profession)
        results_source.append(result_source)

        example_new = {}
        example_new['source_sentences'] = example['sentences']
        example_new['source_predicted_clusters'] = example['predicted_clusters']
        example_new['source_result'] = result_source

        example = make_predictions(target_text,model)        
        example_new['target_sentences'] = example['sentences']
        example_new['target_predicted_clusters'] = example['predicted_clusters']        
        if target_p == -1 or len(target_profession) == 0:
            result_target = - 1
        else:
            result_target  = wino_evaluate(example,target_text,target_p,target_profession)

        example_new['target_result'] = result_target                     
        results_target.append(result_target)          
        coref_results[index] = example_new

    scol = df['source_sentence']
    tcol = df['target_sentence']
    df_new = pd.DataFrame({'source_sentence':scol,'source_coref_result':results_source,'target_sentence':tcol,'target_coref_result':results_target})
    df_new.to_csv(fname+"_results_ver2.tsv",sep='\t',index=False)
    # No need to this because its not gaong to change. (except target_result variable.) Anyway lets do it again
    with open(fname+"_result_ver2.json", 'w') as fp:
        json.dump(coref_results, fp)
        
