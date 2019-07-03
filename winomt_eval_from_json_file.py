import json
import pandas as pd 
import pdb

def eval_instance(clusters,sentences,pronoun,profession,ix):
    result = False
    male_ps =["he","him","his"]
    female_ps = ["she","her"]
    neutral_ps = ["it","they","their","them"]
    flat_sentences = [item for sentence in sentences for item in sentence]
    print('ix:',ix,' profession ',profession)
    parts = [p for p in profession.split(' ') if p.lower() not in ['a','the']]
    named_clusters = []
    for cluster in clusters:
        named_cluster = []
        for entity in cluster:
            named_entity = [flat_sentences[i] for i in range(entity[0],entity[1]+1)]
            named_cluster.append(named_entity)
        named_clusters.append(named_cluster)
    # find the cluster having the profession
    if i == 1:
        pdb.set_trace()
    check_double_enterence = False
    for cluster in named_clusters:
        if any([ part in cluster for part in parts]):
            if check_double_enterence:
                print("warning double_enterence for ix:",ix)
            check_double_enterence = True
            if pronoun.lower() == "male" and any([ p in cluster for p in male_ps]):
                result = True
            elif pronoun.lower() == "female" and any([ p in cluster for p in female_ps]):
                result = True
            elif pronoun.lower() == "neutral" and any([ p in cluster for p in neutral_ps]):
                result = True
            else: # meaning pronoun -1 or pronoun and profession is not in the same cluster 
                result = False
    return result


if __name__ == '__main__':
    lang = 'ar'
    system = 'google'
    tsv_input_fn   = 'translations/{}/ver2_en_org-en_{}_bt.tsv'.format(system,lang)
    tsv_output_fn  = 'translations/{}/ver2_en_org-en_{}_bt.tsv_results_ver2.tsv'.format(system,lang)
    json_output_fn = 'translations/{}/ver2_en_org-en_{}_bt.tsv_result_ver2.json'.format(system,lang)
    
    df_input   = pd.read_table(tsv_input_fn,sep='\t')
    df_input.fillna('',inplace=True)
    df_output  = pd.read_table(tsv_output_fn,sep='\t')
    json_output = json.load(open(json_output_fn,'rb'))
    
    result_sources,result_targets = [],[]
    for i,row in df_input.iterrows():
        output         = json_output[str(i)]
        src_clusters   = output['source_predicted_clusters']
        tgt_clusters   = output['target_predicted_clusters']
        src_sentences  = output['source_sentences']
        tgt_sentences  = output['target_sentences']
        src_pronoun    = row['orjinal_pronoun']
        tgt_pronoun    = row['bt_pronoun']
        src_profession = row['orjinal_profession']
        tgt_profession = row['bt_profession']
        result_src     = eval_instance(src_clusters,src_sentences,src_pronoun,src_profession,i)
        result_tgt     = eval_instance(tgt_clusters,tgt_sentences,tgt_pronoun,tgt_profession,i)
        result_sources.append(result_src)
        result_targets.append(result_tgt)

    pdb.set_trace()
