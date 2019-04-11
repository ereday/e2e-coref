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
        
def spans2(txt):
    tokens=nltk.word_tokenize(txt)
    offset = 0
    res = [] 
    for token in tokens:
        offset = txt.find(token, offset)
        res.append((token, offset, offset+len(token)))
        offset += len(token)
    return res
  
def get_offsets(text):
  result = []
  for token in spans(text):
    result.append(token)
    assert token[0]==text[token[1]:token[2]]
  return result

def _get_indices(word_offsets,toff,word):
  res = []
  is_multi = False
  is_end= False
  for ix,wp in enumerate(word_offsets):
    if wp[1] == toff:
      res.append(ix)      
      if wp[0] == word:
        return res
      else:
        is_multi = True
        is_end = False
    elif is_end:
      return res
    elif is_multi:
      if wp[2]-toff > len(word):
        is_end = True
      else:
        res.append(ix)
  return res

# burayi iste aix[0] i olsun bix[0] i olmasin vs diye yapabilriiz. yapmaliyiz ya da?
def gap_evaluate(pix,aix,bix,example,a_coref,b_coref):
  if a_coref:
    target = aix
  else:
    target = bix  
  words = util.flatten(example["sentences"])
  result = False
  for cluster in example["predicted_clusters"]:
    elements = list(set([p[0] for p in cluster] + [p[1] for p in cluster]))
    if pix[0] in elements: # Target cluster
      if target[0] in elements:
        result = True
        break      
    else:
      continue
  return result
    
  
def get_indices(row):
  word_offsets = get_offsets(row['Text'])
  poff,aoff,boff  = row['Pronoun-offset'],row['A-offset'],row['B-offset']
  pronoun,A,B     = row['Pronoun'],row['A'],row['B']
  pix = _get_indices(word_offsets,poff,pronoun)
  aix = _get_indices(word_offsets,aoff,A)
  bix = _get_indices(word_offsets,boff,B)
  return pix,aix,bix


def _get_indices_google_nl(word_offsets,toff,word):
  res = []
  for ix,word in enumerate(word_offsets):
    if word[1] == toff:
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
  evaluations = []
  with tf.Session() as session:
    model.restore(session)
    fname = 'gapx-merged-nl-head.tsv'
    df = prepare_data(fname)
    for index,row in df.iterrows():      
      print("index:",index)
      text = row['Text']
      a_coref,b_coref = row['A-coref'],row['B-coref']
      if a_coref == False and b_coref == False:
        result = False
        evaluations.append(result)
        continue        
      pix,aix,bix = get_indices_google_nl(row)
      print("pix:",pix," aix:",aix," bix:",bix)
      example = make_predictions(text,model)
      result = gap_evaluate(pix,aix,bix,example,a_coref,b_coref)
      evaluations.append(result)
    df['Result'] = evaluations
    df.to_csv('gapx-merged-evaluation_debug_googlenl.tsv',sep='\t',index=False)
    #print(util.flatten(example['sentences']))
        #print(example['predicted_clusters'])
        #print("aix:",aix," bix:",bix)
        #print("----------------------")
        #print_predictions(example)
        #print("result:",result)
        #text = input("Continue?")
#    while True:
#      text = input("Document text: ")
#      print_predictions(make_predictions(text, model))







