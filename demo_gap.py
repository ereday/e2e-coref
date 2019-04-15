#!/usr/bin/env python

import os
import sys
import time
import json
import numpy as np

import cgi
from BaseHTTPServer import HTTPServer, BaseHTTPRequestHandler
import ssl

import tensorflow as tf
import coref_model as cm
import util

import nltk
nltk.download("punkt")
from nltk.tokenize import sent_tokenize, word_tokenize
import pandas as pd

class CorefRequestHandler(BaseHTTPRequestHandler):
  model = None
  def do_POST(self):
    form = cgi.FieldStorage(
      fp=self.rfile,
      headers=self.headers,
      environ={"REQUEST_METHOD":"POST",
               "CONTENT_TYPE":self.headers["Content-Type"]
      })
    if "text" in form:
      text = form["text"].value.decode("utf-8")
      if len(text) <= 10000:
        print(u"Document text: {}".format(text))
        example = make_predictions(text, self.model)
        print_predictions(example)
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps(example))
        return
    self.send_response(400)
    self.send_header("Content-Type", "application/json")
    self.end_headers()

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
  if a_coref:
    target = aix
  else:
    target = bix  
  words = util.flatten(example["sentences"])
  result = False
  for cluster in example["predicted_clusters"]:
    #elements = list(set([p[0] for p in cluster] + [p[1] for p in cluster]))
    elements = []
    for p in cluster:
      elements = elements + list(range(p[0],p[1]+1))
    if pix[0] in elements: # Target cluster
      if target[0] in elements:
        result = True
        break      
    else:
      continue
  return result

# a_coref,b_coref = either of True,False
def gap_evaluate2(pix,aix,bix,example,a_coref,b_coref):
    words = util.flatten(example["sentences"])
    result_a = False
    result_b = False
    TP="True Positive"
    FP="False Positive"
    FN="False Negative"
    TN="True Negative"
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
                    
    if a_coref == False and b_coref == False:
        # DO something according to mail
        if result_a ==True or result_b == True:
            return FP
        elif result_a == False and result_b == False:
            return TN
        else:
            print("something is wrong, this line should not be printed #0")
            return "__IGNORE__"
    elif a_coref == True and b_coref == True:
        return "__ERROR__"
    elif a_coref == True:
        if   result_a == True and  result_b == True:
            return TP
            # do something according to e-mail
            pass
        elif result_a == True and  result_b == False:
            return TP 
        elif result_a == False and result_b == True:
            return FP
        else: # result_a == False and result_b == False:
            # do something according to e-mail. I think this one is FN
            return FN
    elif b_coref == True:
        if result_a   == True and  result_b == True:
            return TP
            # do something according to e-mail
            pass
        elif result_a == True and  result_b == False:
            return FP
        elif result_a == False and result_b == True:
            return TP
        else: # result_a == False and result_b == False:
            # do something according to e-mail I think this one is FN
            return FN            
    else:
        print("something is wrong, this line should not be printed #1")
        return "__ERROR__"


def _get_indices_google_nl(word_offsets,toff,word):
  res = []
  for ix,word in enumerate(word_offsets):
    #if word[1] == toff:
    if word[1] <= toff and word[2] > toff:
      res.append(ix)
  return res

def get_indices_google_nl(row):
  word_offsets = get_offsets(row['Text'])
  poff,aoff,boff  = row['Pronoun-offset'],row['A_head_offset'],row['B_head_offset']
  pronoun,A,B     = row['Pronoun'],row['A'],row['B']
  #pix = _get_indices_google_nl(word_offsets,poff,pronoun,is_pronoun=True)
  pix = _get_indices_google_nl(word_offsets,poff,pronoun)
  aix = _get_indices_google_nl(word_offsets,aoff,A)
  bix = _get_indices_google_nl(word_offsets,boff,B)
  return pix,aix,bix

if __name__ == "__main__":
  util.set_gpus()

  name = sys.argv[1]
  if len(sys.argv) > 2:
    port = int(sys.argv[2])
  else:
    port = None

  print "Running experiment: {}.".format(name)
  config = util.get_config("experiments.conf")[name]
  config["log_dir"] = util.mkdirs(os.path.join(config["log_root"], name))

  util.print_config(config)
  model = cm.CorefModel(config)

  saver = tf.train.Saver()
  log_dir = config["log_dir"]
  evaluations = []
  with tf.Session() as session:
    checkpoint_path = os.path.join(log_dir, "model.max.ckpt")
    saver.restore(session, checkpoint_path)
    fname = 'gapx-merged-nl-head.tsv'
    df = prepare_data(fname)
    for index,row in df.iterrows():
        print("index:",index)
        text = row['Text']
        a_coref,b_coref = row['A-coref'],row['B-coref']
        #if a_coref == False and b_coref == False:
        #    result = False
        #    evaluations.append(result)
        #    continue
        pix,aix,bix = get_indices_google_nl(row)
        print("pix:",pix," aix:",aix," bix:",bix)
        example = make_predictions(text,model)
        #result = gap_evaluate(pix,aix,bix,example,a_coref,b_coref)
        result = gap_evaluate2(pix,aix,bix,example,a_coref,b_coref)
        evaluations.append(result)
    df['Result'] = evaluations
    df.to_csv('gapx-merged-evaluation_debug_googlenl.tsv',sep='\t',index=False)


# Below code was used before f1 oriented performance calculation

#if __name__ == "__main__":
#  util.set_gpus()
#
#  name = sys.argv[1]
#  if len(sys.argv) > 2:
#    port = int(sys.argv[2])
#  else:
#    port = None
#
#  print "Running experiment: {}.".format(name)
#  config = util.get_config("experiments.conf")[name]
#  config["log_dir"] = util.mkdirs(os.path.join(config["log_root"], name))
#
#  util.print_config(config)
#  model = cm.CorefModel(config)
#
#  saver = tf.train.Saver()
#  log_dir = config["log_dir"]
#  evaluations = []
#  with tf.Session() as session:
#    checkpoint_path = os.path.join(log_dir, "model.max.ckpt")
#    saver.restore(session, checkpoint_path)
#    fname = 'gapx-merged-nl-head.tsv'
#    df = prepare_data(fname)
#    for index,row in df.iterrows():
#        print("index:",index)
#        text = row['Text']
#        a_coref,b_coref = row['A-coref'],row['B-coref']
#        if a_coref == False and b_coref == False:
#            result = False
#            evaluations.append(result)
#            continue
#        pix,aix,bix = get_indices_google_nl(row)
#        print("pix:",pix," aix:",aix," bix:",bix)
#        example = make_predictions(text,model)
#        result = gap_evaluate(pix,aix,bix,example,a_coref,b_coref)
#        evaluations.append(result)
#    df['Result'] = evaluations
#    df.to_csv('gapx-merged-evaluation_debug_googlenl.tsv',sep='\t',index=False)
#    
#    #while True:
#    #    text = raw_input("Document text: ")
#    #    print_predictions(make_predictions(text, model))
