import pandas as pd
import jieba
data=pd.read_csv('data/atec_nlp_sim_train.csv',sep='\t',index_col=0,header=None)
data.columns=['s1','s2','label']
print(data.head())
stopwords=['\ufeff','，',' ']
jieba.load_userdict('data/dict.txt')
def seg_sen(sen_str):
    seg_sen=jieba.cut(sen_str)
    seg_sen=[ele for ele in seg_sen if ele not in stopwords]
    return seg_sen

def process_data(s1_train,s2_train):
    '''
    :param s1_train:
    :param s2_train:
    :return:
    '''
    s1_all=[]
    s2_all=[]
    all_data=[]
    for s1,s2 in zip(s1_train,s2_train):
        seg_s1=seg_sen(s1)
        seg_s2=seg_sen(s2)
        s1_all.append(seg_s1)
        s2_all.append(seg_s2)
        all_data.extend(seg_s1)
        all_data.extend(seg_s2)
    all_data=list(set(all_data))
    #unk 标记不认识词语，pad补充到特定长度
    all_data.extend(['<UNK>'])
    all_data.extend(['<PAD>'])
    id2word={k:v for k,v in enumerate(all_data)}
    word2id={v:k for k,v in id2word.items()}
    return s1_all,s2_all,word2id,id2word

s1_data=data['s1'].tolist()
s2_data=data['s2'].tolist()
s1_all,s2_all,word2id,id2word=process_data(s1_data,s2_data)
data['s1_seg']=s1_all
data['s2_seg']=s2_all
#maping_to_fixed_length
max_sen_len=15
def transform_word2id(data,word2id):
    wordlist=[]
    for i in data:
        if i in word2id.keys():
            wordlist.append(word2id[i])
        else:
            wordlist.append(word2id['<UNK>'])
    if len(wordlist)<max_sen_len:
        wordlist.extend([word2id['<PAD>']]*(max_sen_len-len(wordlist)))
        return wordlist
    else:
        return wordlist[:max_sen_len]

data['s1_seg_map']=data['s1_seg'].apply(transform_word2id,word2id=word2id)
data['s2_seg_map']=data['s2_seg'].apply(transform_word2id,word2id=word2id)

all_data=[]
for i in range(len(data['s1_seg_map'])):
    all_data.append([data['s1_seg_map'].iloc[i],data['s2_seg_map'].iloc[i],data['label'].iloc[i]])
import pickle        
ratio=int(len(all_data)*0.8)
train_data=all_data[:ratio]
test_data=all_data[ratio:]
with open('data.pkl','wb') as f:
    pickle.dump((train_data,test_data,word2id,id2word),f)            






