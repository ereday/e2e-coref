#!/usr/bin/env python
import unicodecsv
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

def find_in_arr(arr,var):
    try:
        index_element = arr.index(var)
        return index_element
    except ValueError:
        return -1

def wino_evaluate(example,text,pronoun_ix,profession_ix):
    proffesion = text.split(" ")[profession_ix].replace(",","").replace(":","")
    pronoun = text.split(" ")[pronoun_ix].replace(",","").replace(":","")
    
    words = util.flatten(example["sentences"])
    result = False 
    for cluster in example["predicted_clusters"]:
        if len(example['sentences']) > 1:
            print("SENTENCE LENGTH IS MORE THAN 1 , demo_gap_wino.py WARNINGG")
        tokens = example['sentences'][0]
        elements = []
        for p in cluster:
            elements = elements + list(range(p[0],p[1]+1))
        elements_string = [ tokens[x] for x in elements]
        if find_in_arr(elements_string,proffesion) > -1 and find_in_arr(elements_string,pronoun) > -1:
            result = True
    return result

if __name__ == "__main__":
  util.set_gpus()

  name = sys.argv[1]
  fname = sys.argv[2]
  port = None
  
  print "Running experiment: {}.".format(name)
  config = util.get_config("experiments.conf")[name]
  config["log_dir"] = util.mkdirs(os.path.join(config["log_root"], name))

  util.print_config(config)
  model = cm.CorefModel(config)

  saver = tf.train.Saver()
  log_dir = config["log_dir"]
  results_source,results_target = [],[]
  with tf.Session() as session:
    checkpoint_path = os.path.join(log_dir, "model.max.ckpt")
    saver.restore(session, checkpoint_path)
    df = prepare_data(fname)
    for index,row in df.iterrows():
        print("index:",index)
        source_text = row['source_sentence']
        target_text = row['target_sentence']
        source_pix  = row['orjinal_pronoun_index']
        target_pix  = row['bt_pronoun_index']
        source_profession_ix = row['orjinal_profession_ix']
        target_profession_ix = row['bt_profession_ix']
        if target_pix == -1 or target_profession_ix == -1:
            result_source  = wino_evaluate(example,source_text,source_pix,source_profession_ix)
            results_source.append(result_source)
            resuls_target.append(-1)
            continue
        example = make_predictions(target_text,model)
        result_target  = wino_evaluate(example,target_text,target_pix,target_profession_ix)
        result_source  = wino_evaluate(example,source_text,source_pix,source_profession_ix)
        results_source.append(result_source)
        results_target.append(result_target)
    scol = df['source_sentence']
    tcol = df['target_sentence']
    df_new = pd.DataFrame({'source_sentence':scol,'source_coref_result':results_source,'target_sentence':tcol,'target_coref_result':results_target})
    df_new.to_csv('deneme.tsv',sep='\t',index=False)
    #df_org = prepare_data(fname)        
    #df['Result'] = evaluations
    #df_org['A-coref'] = evala
    #df_org['B-coref'] = evalb
    #df_org.to_csv('gtrans_backtranslation-gapx-predictions_all_fields.tsv',sep='\t',index=False)
    #df_org[['ID','A-coref','B-coref']].to_csv('gtrans_backtranslation-gapx_predictions.tsv',sep='\t',index=False,header=False)
    #df.to_csv('gapx-predictions_all_fields.tsv',sep='\t',index=False)
    #df[['ID','A-coref','B-coref']].to_csv('gapx_predictions.tsv',sep='\t',index=False,header=False)


    
