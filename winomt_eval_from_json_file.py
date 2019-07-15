import json
import pandas as pd 
import pdb

def is_in(cluster,prof):
    for c in cluster:
        if (prof in c) or (prof.lower() in c) or (prof.upper() in c) or (prof.capitalize() in c):        
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
                #print("warning double_enterence for ix:",ix)                
                if result == True:
                    #print("is not considered")                    
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

def _evaluate(df,side):
    result = {'male':{'T':0,'F':0},'female':{'T':0,'F':0},'neutral':{'T':0,'F':0}}
    for i,row in df.iterrows():
        if row['{}_pronoun'.format(side)] == "male":
            if row['{}_result'.format(side)] == False:
                result['male']['F'] += 1
            else:
                result['male']['T'] += 1
        elif row['{}_pronoun'.format(side)] == "female":
            if row['{}_result'.format(side)] == False:
                result['female']['F'] += 1
            else:
                result['female']['T'] += 1
        else:
            if row['{}_result'.format(side)] == False:
                result['neutral']['F'] += 1
            else:
                result['neutral']['T'] += 1
    return result

def get_acc(result):
    male_acc   = result["male"]['T']/sum(result["male"].values())
    female_acc = result["female"]['T']/sum(result["female"].values())
    full_acc   = (result["female"]['T']+result["male"]['T'])/(sum(result["female"].values())+sum(result["male"].values()))
    bias       = female_acc/male_acc
    #print("male acc:{:.2f} female acc:{:.2f} full acc:{:.2f} bias:{:.2f}".format(male_acc,female_acc,full_acc,bias))
    print("male acc\tfemale acc\tfull acc\tbias")
    print("{:.2f}\t{:.2f}\t{:.2f}\t{:.2f}".format(male_acc,female_acc,full_acc,bias))
    return male_acc,female_acc,full_acc,bias

def evaluate(df):
    # original source
    result_org_source = _evaluate(df,"src")                    
    # filter rows where at least one of tgt_profession or tgt_pronoun is -1 
    df_filtered = df[(df['tgt_profession']!='-1') & (df['tgt_pronoun']!='-1')]
    result_filtered_source=  _evaluate(df_filtered,"src")
    result_filtered_target = _evaluate(df_filtered,"tgt")
    print("full dataset source side:")
    male_acc,female_acc,full_acc,bias = get_acc(result_org_source)    
    #print(result_org_source)

    print("filtered dataset source side:")
    male_acc,female_acc,full_acc,bias = get_acc(result_filtered_source)
    #print(result_filtered_source)
    
    print("filtered dataset target side:")
    male_acc,female_acc,full_acc,bias = get_acc(result_filtered_target)
    #print(result_filtered_target)

if __name__ == '__main__':
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

            df_final = pd.DataFrame({'src_sentence':sent_srcs,'src_pronoun':pro_srcs,'src_profession':prof_srcs,'tgt_sentence':sent_tgts,'tgt_pronoun':pro_tgts,'tgt_profession':prof_tgts,'src_result':result_srcs,'tgt_result':result_tgts})
            evaluate(df_final)
            df_final.to_csv(tsv_output_fn,index=False,sep='\t')

    

# This part is for sampling some instance in terminal

def sample_from_wrong_instances(df_filtered):    
    srctgt_ft,srctgt_tf,srctgt_ff,srctgt_tt = [],[],[],[]
    for i,row in df_filtered.iterrows():
        if row['src_result'] == False and row['tgt_result'] == True:
            srctgt_ft.append(i)
        elif row['src_result'] == True and row['tgt_result'] == False:
            srctgt_tf.append(i)
        elif row['src_result'] == False and row['tgt_result'] == False:
            srctgt_ff.append(i)
        else: # true,true
            srctgt_tt.append(i)
    return srctgt_ft,srctgt_tf,srctgt_ff,srctgt_tt

def sample(df,n=3):
    s = df.sample(n)
    for i,row in s.iterrows():
        print('{}\t{}\t{}'.format(row['src_sentence'],row['src_profession'],row['src_result']))
        print('{}\t{}\t{}\n'.format(row['tgt_sentence'],row['tgt_profession'],row['tgt_result']))
        print('')

# USAGE
# import pandas as pd 
# fname = 'ver2_en_org-en_de_bt.tsv_results_ver2.tsv'
# df = pd.read_table(fname,sep='\t')
# df_filtered = df[(df['tgt_profession']!='-1') & (df['tgt_pronoun']!='-1')]
# df_filtered = df_filtered.reset_index(drop=True)
# srctgt_ft_ix,srctgt_tf_ix,srctgt_ff_ix,srctgt_tt_ix = sample_from_wrong_instances(df_filtered)
# st_ft = df_filtered.iloc[srctgt_ft_ix]
# st_tf = df_filtered.iloc[srctgt_tf_ix]
# st_ff = df_filtered.iloc[srctgt_ff_ix]
# st_tt = df_filtered.iloc[srctgt_tt_ix]
# sample(st_ft,n=3)


#import  requests
#import json
#app_id = '49cfde7a'
#app_key = '3066064114ce83fc92d5fa66fc7c5cd7'
#language = 'en'
#word_id = 'Ace'
#url = 'https://od-api.oxforddictionaries.com:443/api/v2/entries/'  + language + '/'  + word_id.lower()
#urlFR = 'https://od-api.oxforddictionaries.com:443/api/v2/stats/frequency/word/'  + language + '/?corpus=nmc&lemma=' + word_id.lower()
#r = requests.get(url, headers = {'app_id' : app_id, 'app_key' : app_key})
#print("code {}\n".format(r.status_code))
#print("text \n" + r.text)
#print("json \n" + json.dumps(r.json()))
            
