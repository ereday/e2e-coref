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

def spans(txt):
    tokens=nltk.word_tokenize(txt)
    offset = 0
    for token in tokens:
        offset = txt.find(token, offset)
        yield token, offset, offset+len(token)
        offset += len(token)

def get_offsets(text):
  result = []
  for token in spans(text):
    result.append(token)
    assert token[0]==text[token[1]:token[2]]
  return result


def gap_evaluate(pix,aix,bix,example,a_coref,b_coref):
    words = util.flatten(example["sentences"])
    result_a = False
    result_b = False
    for cluster in example["predicted_clusters"]:
        #elements = list(set([p[0] for p in cluster] + [p[1] for p in cluster]))
        elements = []
        for p in cluster:
            elements = elements + list(range(p[0],p[1]+1))
            if pix[0] in elements: # Target cluster
                if aix[0] in elements:
                    result_a = True
                if bix[0] in elements:
                    result_b = True
    return result_a,result_b

def _get_indices_google_nl(word_offsets,toff,word):
  res = []
  for ix,word in enumerate(word_offsets):
    if word[1] <= toff and word[2] > toff:
      res.append(ix)
  return res

def get_indices_google_nl(row):
  word_offsets = get_offsets(row['Text'])
  poff,aoff,boff  = row['Pronoun-offset'],row['A_head_offset'],row['B_head_offset']
  pronoun,A,B     = row['Pronoun'],row['A'],row['B']
  pix = _get_indices_google_nl(word_offsets,poff,pronoun)
  aix = _get_indices_google_nl(word_offsets,aoff,A)
  bix = _get_indices_google_nl(word_offsets,boff,B)
  return pix,aix,bix


if __name__ == "__main__":
  config = util.initialize_from_env()
  model = cm.CorefModel(config)
  evala,evalb = [],[]
  with tf.Session() as session:
    model.restore(session)
    fname = 'gapx-merged-nl-head.tsv'
    df = prepare_data(fname)
    for index,row in df.iterrows():      
      print("index:",index)
      text = row['Text']
      a_coref,b_coref = row['A-coref'],row['B-coref']
      pix,aix,bix = get_indices_google_nl(row)
      print("pix:",pix," aix:",aix," bix:",bix)
      example = make_predictions(text,model)
      resulta,resultb = gap_evaluate(pix,aix,bix,example,a_coref,b_coref)
      evala.append(resulta)
      evalb.append(resultb)
    df['A-coref'] = evala
    df['B-coref'] = evalb
    df.to_csv('gapx-predictions_all_fields_lee18.tsv',sep='\t',index=False)
    df['ID','A-coref','B-coref'].to_csv('gapx_predictions_lee18.tsv',sep='\t',index=False,header=False)      
