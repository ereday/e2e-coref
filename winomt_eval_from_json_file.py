import json
import pandas as pd 
import pdb

def is_in(cluster,prof):
    for c in cluster:
        if prof in c:
            return True
    return False

def eval_instance(clusters,sentences,pronoun,profession,ix):
    result = False
    male_ps =["he","him","his"]
    female_ps = ["she","her"]
    neutral_ps = ["it","they","their","them"]
    flat_sentences = [item for sentence in sentences for item in sentence]
    # 54,900,1638,1931,2342,2484
    #print('ix:',ix,' profession ',profession)
    parts = [p for p in profession.split(' ') if p.lower() not in ['a','the']]
    named_clusters = []
    for cluster in clusters:
        named_cluster = []
        for entity in cluster:
            named_entity = [flat_sentences[i] for i in range(entity[0],entity[1]+1)]
            named_cluster.append(named_entity)
        named_clusters.append(named_cluster)
    # find the cluster having the profession                
    check_double_enterence = False
    for cluster in named_clusters:
        if any([ is_in(cluster,part) for part in parts]):
            if check_double_enterence:
                print("warning double_enterence for ix:",ix)                
                if result == True:
                    print("is not considered")                    
                    continue
            check_double_enterence = True                
            if pronoun.lower() == "male" and any([is_in(cluster,p) for p in male_ps]):
                result = True
            elif pronoun.lower() == "female" and any([is_in(cluster,p) for p in female_ps]):
                result = True
            elif pronoun.lower() == "neutral" and any([is_in(cluster,p) for p in neutral_ps]):
                result = True
            else: # meaning pronoun -1 or pronoun and profession is not in the same cluster 
                result = False
    return result


if __name__ == '__main__':
    #lang = 'ar'
    #system = 'google'
    langs = ["de","ru","it","fr","es","uk","he","ar"]
    systems = ["google","aws","bing"]
    for lang in langs:
        for system in systems:
            print("############ LANGUAGE: {} SYSTEM:{} ##############".format(lang,system))
            if (system == "aws" and lang == "uk") or (system == "bing" and lang == "uk"):
                continue            
            tsv_input_fn   = 'translations/{}/ver2_en_org-en_{}_bt.tsv'.format(system,lang)
            tsv_output_fn  = 'translations/{}/ver2_en_org-en_{}_bt.tsv_results_ver2.tsv'.format(system,lang)
            json_output_fn = 'translations/{}/ver2_en_org-en_{}_bt.tsv_result_ver2.json'.format(system,lang)
    
            df_input   = pd.read_table(tsv_input_fn,sep='\t')
            df_input.fillna('',inplace=True)
            df_output  = pd.read_table(tsv_output_fn,sep='\t')
            json_output = json.load(open(json_output_fn,'rb'))
    
            result_srcs,result_tgts = [],[]
            prof_srcs,prof_tgts = [],[]
            pro_srcs,pro_tgts = [],[]
            sent_srcs,sent_tgts = [],[]
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
                result_srcs.append(result_src)
                result_tgts.append(result_tgt)
                pro_srcs.append(src_pronoun)
                pro_tgts.append(tgt_pronoun)
                prof_srcs.append(src_profession)
                prof_tgts.append(tgt_profession)
                sent_srcs.append(" ".join([item for sentence in src_sentences for item in sentence]))
                sent_tgts.append(" ".join([item for sentence in tgt_sentences for item in sentence]))        

            df_final = pd.DataFrame({'src_sentence':sent_srcs,'src_pronoun':pro_srcs,'src_profession':pro_srcs,'tgt_sentence':sent_tgts,'tgt_pronoun':pro_tgts,'tgt_profession':pro_tgts,'src_result':result_srcs,'tgt_result':result_tgts})
            # the input tsv_output_fn is not used anywhere and it is wrong, therefore It can be overwritten.
            df_final.to_csv(tsv_output_fn,index=False,sep='\t')

    
