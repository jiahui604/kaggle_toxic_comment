#!/usr/bin/env python
# coding: utf-8
#Importing necessary Libraries
import pandas as pd
import numpy as np
from collections import Counter
from afinn import Afinn

#text cleaning
import re
import string
import collections
import nltk
import emoji
from nltk.stem import *
from nltk import pos_tag
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer, HashingVectorizer


#missing value computation
def cal_missing_val(df):
    data_dict = {}
    for col in df.columns:
        data_dict[col] = (df[col].isnull().sum()/df.shape[0])*100
    return pd.DataFrame.from_dict(data_dict, orient='index', columns=['MissingValueInPercentage'])


## Reducing memory
## Function to reduce the DF size
def reduce_memory(df, verbose=True):
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    start_mem = df.memory_usage().sum() / 1024**2    
    for col in df.columns:
        col_type = df[col].dtypes
        if col_type in numerics:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)  
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)    
    end_mem = df.memory_usage().sum() / 1024**2
    if verbose: print('Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)'.format(end_mem, 100 * (start_mem - end_mem) / start_mem))
    return df

def feature_creation_before_cleaning(df):
    ###### Decode the emojis ######
    df['comment_text'] = df['comment_text'].apply(lambda x: emoji.demojize(x))

    ###### Split the comment into words ######
    df['comment_text_split'] = df['comment_text'].apply(lambda x: x.split())
    
    ###### Count of words in each comment ######
    df['num_words'] = df['comment_text_split'].apply(lambda comment: len(comment))
    
    df  = df[df['num_words']>0.0]
    ###### Count of stopwords percentage within in that comment ######
    eng_stopwords             = stopwords.words("english")
    df['num_stopwords']       = df['comment_text_split'].apply(lambda comment: sum(comment.count(w) for w in eng_stopwords))
    df['percentage_non_stop'] = 1-(df['num_stopwords'] / df['num_words']) #non-stopword matters more than stopwords
    
    ###### Count of caps and percentage within in that comment ######
    df['num_caps'] = df['comment_text_split'].apply(lambda comment: sum(1 for c in comment if c.isupper()))
    df['percentage_cap'] = df['num_caps']/df['num_words']

    ###### Count of symbol and percentage within in that comment ######
    df['num_toxic_symbol']  = df['comment_text_split'].apply(lambda comment: sum(comment.count(w) for w in '@!#$%&'))
    df['percentage_symbol'] = df['num_toxic_symbol'] / df['num_words']
    
    ###### Count of happy and percentage within in that comment ######
    df['num_happy'] = df['comment_text'].apply(lambda comment: 
                                               sum(comment.count(w) for w in (':-)', ':)', ';-)', ';)',':d',":p",":dd","8)","(-:","(:",
                                                   ':smile:',':simple_smile:',':smiley:',':sweat_smile:',':smiley_cat:',
                                                   ':smile_cat:',':smirk:',':smirk_cat:',':grin:',':grinning:',':laughing:',
                                                   ':bowtie:',':relaxed:',':heart_eyes:',':kissing_heart:',':joy:',':thumbsup:',
                                                   ':heart_eyes:',':kissing_heart:',':yellow_heart:',':blue_heart:',':purple_heart:',
                                                   ':heart:',':green_heart:',':broken_heart:',':heartbeat:',':heartpulse:',
                                                   ':two_hearts:',':revolving_hearts:',':sparkling_heart:',':heart_eyes_cat:',
                                                   ':gift_heart:',':hearts:',':heart_decoration:')))
    
    df['percentage_happy'] = df['num_happy'] / df['num_words']
    
    ###### Count of sad and percentage within in that comment ######
    df['num_sad'] = df["comment_text"].apply(lambda comment: 
                                         sum(comment.count(w) for w in (":')",":-(",":(",":s",":-s",":/",":')",":-(",":(",':angry:',':cry:',':crying_cat_face:',
                                                                        ':sob:',':disappointed:',':weary:',':fearful:',':frowning:',':disappointed:',':sweat:',
                                                                        ':cold_sweat:',':rage:',':rage1:',':rage2:',':rage3:',':rage4:',':triumph:',':thumbsdown:',
                                                                        ':fu:',':poop:',':shit:',':hankey:')))
    df['percentage_sad'] = df['num_sad'] / df['num_words']
    
    ###### Count of unique words and percentage within in that comment ######
    df['num_unique_words'] = df['comment_text_split'].apply(lambda comment: len(set(w for w in comment)))
    df['percentage_unique'] = df['num_unique_words'] / df['num_words']
    
    ###### Count of punctuation and percentage within in that comment ######
    df['punctuation'] = df['comment_text'].apply(lambda comment: sum(comment.count(w) for w in string.punctuation))
    df['percentage_punctuation'] = df['punctuation'] / df['num_words']
    
    ###### Count of exclamation marks ######
    df['exclamation_marks'] = df['comment_text'].apply(lambda comment: comment.count('!'))
    df['exclamation_marks_vs_length'] = df['exclamation_marks'] / df['num_words']
    
    ###### Count of new lines ######
    df['num_newlines'] = df["comment_text"].apply(lambda comment: 
                                         sum(comment.count(new_line) for new_line in ("\n","\\n")))
    
    ###### new features ######
    ###### Count of titles ######
    df['num_words_title'] = df['comment_text'].apply(lambda x: len([w for w in str(x).split() if w.istitle()]))

    ###### Count of character ######
    df['num_chars'] = df['comment_text'].apply(lambda x: len(str(x)))
    
    ###### character per word ######
    df['chars_per_word'] = df['num_chars'] / df['num_words']

    ###### Count of sentences in comment ######
    df['sentence'] = df['comment_text'].apply(lambda x: [s for s in re.split(r'[.!?\n]+', str(x))])
    df['num_sentence'] = df['sentence'].apply(lambda x: len(x))

    ###### sentences mean in comment ######
    df['sentence_mean'] = df.sentence.apply(lambda xs: [len(x) for x in xs]).apply(lambda x: np.mean(x))
    
    ###### sentences max in comment ######
    df['sentence_max'] = df.sentence.apply(lambda xs: [len(x) for x in xs]).apply(lambda x: max(x) if len(x) > 0 else 0)
    
    ###### sentences min in comment ######
    df['sentence_min'] = df.sentence.apply(lambda xs: [len(x) for x in xs]).apply(lambda x: min(x) if len(x) > 0 else 0)

    ###### sentences std deviation in comment ######
    df['sentence_std'] = df.sentence.apply(lambda xs: [len(x) for x in xs]).apply(lambda x: np.std(x))

    ###### word per sentences in comment ######
    df['words_per_sentence'] = df['num_words'] / df['num_sentence']

    ###### number of repeated sentences in comment ######
    df['num_repeated_sentences'] = df['sentence'].apply(lambda x: len(x) - len(set(x)))
    df.drop('sentence', inplace=True, axis=1)
    
    # From https://www.kaggle.com/ogrellier/lgbm-with-words-and-chars-n-gram
    ###### starts with columns ######
    df['start_with_columns'] = df['comment_text'].apply(lambda x: 1 if re.search(r'^\:+', x) else 0)
   
    ###### has_timestamp ######
    df['has_timestamp'] = df['comment_text'].apply(lambda x: 1 if re.search(r'\d{2}|:\d{2}', x) else 0)

    ###### has_date_long ######
    df['has_date_long'] = df['comment_text'].apply(lambda x: 1 if re.search(r'\D\d{2}:\d{2}, \d{1,2} \w+ \d{4}', x) else 0)
    
    ###### has_date_short ######
    df['has_date_short'] = df['comment_text'].apply(lambda x: 1 if re.search(r'\D\d{1,2} \w+ \d{4}', x) else 0)

    ###### has_email ######
    df['has_email'] = df['comment_text'].apply(lambda x: 1 if re.search(r'[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+', x) else 0)
    
    ###### find ip - two columns created as 'ip, 'is_ip' ######
    df['ip']    = [re.findall('[0-9]+(?:\.[0-9]+){3}', i)for i in df.comment_text.tolist()]
    df['ip']    = df['ip'].astype(str).str.replace('[','').str.replace(']','').str.replace("'",'')
    df['is_ip'] = np.where(df['ip'] == '', 0, 1)
    
    ###### sentiment analysis ######
    afinn = Afinn()
    
    df['afinn'] = df.comment_text.apply(lambda xs: [afinn.score(x) for x in xs.split()])
    
    df['afinn_sum'] = df.afinn.apply(lambda x: sum(x) if len(x) > 0 else 0)
   
    df['afinn_mean'] = df.afinn.apply(lambda x: np.mean(x))
    
    df['afinn_max'] = df.afinn.apply(lambda x: max(x) if len(x) > 0 else 0)
    
    df['afinn_min'] = df.afinn.apply(lambda x: min(x) if len(x) > 0 else 0)
    
    df['afinn_std'] = df.afinn.apply(lambda x: np.std(x))
    
    df['afinn_num'] = df.afinn.apply(lambda xs: len([x for x in xs if x != 0]))
    
    df['afinn_num_pos'] = df.afinn.apply(lambda xs: len([x for x in xs if x > 0]))
    
    df['afinn_num_neg'] = df.afinn.apply(lambda xs: len([x for x in xs if x < 0]))
    
    df['afinn_per_word'] = df['afinn_num'] / (df['num_words'] + 0.0001)
    
    df['afinn_pos_per_word'] = df['afinn_num_pos'] / (df['num_words'] + 0.0001)
    
    df['afinn_neg_per_word'] = df['afinn_num_neg'] / (df['num_words'] + 0.0001)
    
    df['afinn_neg_per_pos'] = df['afinn_num_pos'] / (df['afinn_num_neg'] + 0.0001)
    
    df.drop('afinn', inplace=True, axis=1)
    
    return df


def other_clean(comment):
    #Convert to lower case , so that Hi and hi are the same
    comment=comment.lower()
    # remove leaky elements like ip,user
    comment=re.sub("\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}","",comment)
    #removing usernames
    comment=re.sub("\[\[.*\]","",comment)
    return (comment)

def glove_preprocess(text):
    """
    adapted from https://nlp.stanford.edu/projects/glove/preprocess-twitter.rb
	thanks to Dieter from https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge/discussion/50350#287297
    """
    # Different regex parts for smiley faces
    eyes = "[8:=;]"
    nose = "['`\-]?"
    text = re.sub("https?:* ", "<URL>", text)
    text = re.sub("www.* ", "<URL>", text)
    text = re.sub("\[\[User(.*)\|", '<USER>', text)
    text = re.sub("<3", '<HEART>', text)
    text = re.sub("[-+]?[.\d]*[\d]+[:,.\d]*", "<NUMBER>", text)
    text = re.sub(eyes + nose + "[Dd)]", '<SMILE>', text)
    text = re.sub("[(d]" + nose + eyes, '<SMILE>', text)
    text = re.sub(eyes + nose + "p", '<LOLFACE>', text)
    text = re.sub(eyes + nose + "\(", '<SADFACE>', text)
    text = re.sub("\)" + nose + eyes, '<SADFACE>', text)
    text = re.sub(eyes + nose + "[/|l*]", '<NEUTRALFACE>', text)
    text = re.sub("/", " / ", text)
    text = re.sub("[-+]?[.\d]*[\d]+[:,.\d]*", "<NUMBER>", text)
    text = re.sub("([!]){2,}", "! <REPEAT>", text)
    text = re.sub("([?]){2,}", "? <REPEAT>", text)
    text = re.sub("([.]){2,}", ". <REPEAT>", text)
    pattern = re.compile(r"(.)\1{2,}")
    text = pattern.sub(r"\1" + " <ELONG>", text)
    return text

def data_cleaning(df):
    
    repl = {
    "yay!": " good ",
    "yay": " good ",
    "yaay": " good ",
    "yaaay": " good ",
    "yaaaay": " good ",
    "yaaaaay": " good ",
    ":')": " sad ",
    ":-(": " sad ",
    ":(": " sad ",
    ":s": " sad ",
    ":-s": " sad ",
    ":/": "sad",
    ":')": " sad ",
    ":-(": " sad ",
    ":(": " sad ",
    ":d": " happy ",
    ":p": " happy ",
    ":dd": " happy ",
    "8)": " happy ",
    ":-)": " happy ",
    ":)": " happy ",
    ";)": " happy ",
    "(-:": " happy ",
    "(:": " happy ",
    r"\br\b": "are",
    r"\bu\b": "you",
    r"\bhaha\b": "ha",
    r"\bhahaha\b": "ha",
    r"\bdon't\b": "do not",
    r"\bdoesn't\b": "does not",
    r"\bdidn't\b": "did not",
    r"\bhasn't\b": "has not",
    r"\bhaven't\b": "have not",
    r"\bhadn't\b": "had not",
    r"\bwon't\b": "will not",
    r"\bwouldn't\b": "would not",
    r"\bcan't\b": "can not",
    r"\bcannot\b": "can not",
    r"\bi'm\b": "i am",
    "haha": "smile",
    "hahaha": "smile",
    "aren't" : "are not",
    "can't" : "cannot",
    "couldn't" : "could not",
    "didn't" : "did not",
    "doesn't" : "does not",
    "don't" : "do not",
    "hadn't" : "had not",
    "hasn't" : "has not",
    "haven't" : "have not",
    "he'd" : "he would",
    "he'll" : "he will",
    "he's" : "he is",
    "she'd" : "she would",
    "she'll" : "she will",
    "she's" : "she is",
    "i'd" : "I would",
    "i'd" : "I had",
    "i'll" : "I will",
    "i'm" : "I am",
    "isn't" : "is not",
    "it's" : "it is",
    "it'll":"it will",
    "i've" : "I have",
    "let's" : "let us",
    "mightn't" : "might not",
    "mustn't" : "must not",
    "shan't" : "shall not",
    "she'd" : "she would",
    "she'll" : "she will",
    "she's" : "she is",
    "shouldn't" : "should not",
    "that's" : "that is",
    "there's" : "there is",
    "they'd" : "they would",
    "they'll" : "they will",
    "they're" : "they are",
    "they've" : "they have",
    "we'd" : "we would",
    "we're" : "we are",
    "weren't" : "were not",
    "we've" : "we have",
    "what'll" : "what will",
    "what're" : "what are",
    "what's" : "what is",
    "what've" : "what have",
    "where's" : "where is",
    "who'd" : "who would",
    "who'll" : "who will",
    "who're" : "who are",
    "who's" : "who is",
    "who've" : "who have",
    "won't" : "will not",
    "wouldn't" : "would not",
    "you'd" : "you would",
    "you'll" : "you will",
    "you're" : "you are",
    "you've" : "you have",
    "'re": "are",
    "wasn't": "was not",
    "we'll":"we will"
    }
    
    df['comment_text_clean'] = df['comment_text_split'].apply(lambda splits_s: [repl[splits_s[i].lower()] if splits_s[i].lower() in repl.keys() else splits_s[i] for i in range(0,len(splits_s))])
    
    ##### Remove stopword after counting them #####
    eng_stopwords = stopwords.words("english")
    df['comment_text_clean'] = df['comment_text_clean'].apply(lambda x: [word for word in x if word not in (eng_stopwords)])
    
    ##### Remove digits #####
    df['comment_text_clean'] = df['comment_text_clean'].apply(lambda x: [i for i in x if not i.isdigit()])
    
    ##### Remove tense of a word #####
    df['comment_text_clean'] = df['comment_text_clean'].apply(lambda x : [WordNetLemmatizer().lemmatize(word,'v') for word in x])
    
    # join clean comment
    df['comment_text_clean'] = df['comment_text_clean'].apply(lambda x : ' '.join(x)) # join clean comment
    
    ##### Remove puncuation after counting them #####
    df["comment_text_clean"] = df["comment_text_clean"].str.translate(str.maketrans('','',string.punctuation))
    
    ##### Remove line breaker symbol after counting them #####
    df["comment_text_clean"] = df["comment_text_clean"].str.replace("\n"," ")
    
     ##### glove preprocess #####
    df["comment_text_clean"]= df["comment_text_clean"].apply(glove_preprocess)
                      
    ##### other cleaning of comment such as lowering case, removing username #####
    df['comment_text_clean'] = df['comment_text_clean'].apply(lambda x : other_clean(x))
    
    df.drop('comment_text_split', axis=1, inplace=True)
    return (df)

def correction(word, WORDS):
    "Most probable spelling correction for word."
    
#     def words(text): return re.findall(r'\w+', text.lower())

    def P(word): 
        "Probability of `word`."
        # use inverse of rank as proxy
        # returns 0 if the word isn't in the dictionary
        return - WORDS.get(word, 0)

    def candidates(word): 
        "Generate possible spelling corrections for word."
        return (known([word]) or known(edits1(word)) or known(edits2(word)) or [word])

    def known(words): 
        "The subset of `words` that appear in the dictionary of WORDS."
        return set(w for w in words if w in WORDS)

    def edits1(word):
        "All edits that are one edit away from `word`."
        letters    = 'abcdefghijklmnopqrstuvwxyz'
        splits     = [(word[:i], word[i:])    for i in range(len(word) + 1)]
        deletes    = [L + R[1:]               for L, R in splits if R]
        transposes = [L + R[1] + R[0] + R[2:] for L, R in splits if len(R)>1]
        replaces   = [L + c + R[1:]           for L, R in splits if R for c in letters]
        inserts    = [L + c + R               for L, R in splits for c in letters]
        return set(deletes + transposes + replaces + inserts)

    def edits2(word): 
        "All edits that are two edits away from `word`."
        return (e2 for e1 in edits1(word) for e2 in edits1(e1))
    
    return max(candidates(word), key=P)

def get_spell_check(sentence, wordcloud):
    correct_words = []
    for word in sentence:
        correct_words.append(wordcloud[word])
    return (' '.join(correct_words))

###### Count repeated words after cleaning ######
def word_repeated(comment):
    repeated_threshold = 10 # 3,5,15....????
    word_counts = collections.Counter(comment)
    return sum(count for word, count in sorted(word_counts.items()) if count > repeated_threshold)

def tag_part_of_speech(comment):
    pos_list   = pos_tag(comment)
    all_tags   = [i[1] for i in pos_list]
    all_tags_count = Counter(all_tags)
    noun_count  = [v for k, v in all_tags_count.items() if k in ('NN','NNP','NNPS','NNS')]
    adjective_count  = [v for k, v in all_tags_count.items() if k in ('JJ','JJR','JJS')]
    verb_count  = [v for k, v in all_tags_count.items() if k in ('VB','VBD','VBG','VBN','VBP','VBZ')]
    return ([sum(noun_count),sum(adjective_count), sum(verb_count)])

def count_website(comment):
    return len([j for j in comment if j[:4] == 'http' or j[:3] == 'www'])
    
def feature_creation_after_cleaning(df): 
    
    df['comment_text_clean_split'] = df['comment_text_clean'].apply(lambda x: x.split())
    
    ###### count of website mentioned######
    df['num_website']    = df['comment_text_clean_split'].apply(lambda comment: count_website(comment))
    
    ###### count of repeated words ######
    df['num_rep_words']            = df['comment_text_clean_split'].apply(lambda comment: word_repeated(comment))
    df['percentage_repeated_word'] = df['num_rep_words'] / df['num_words']
    
    ###### Count number of adjectives, noun, verb ######
    df['num_nouns'], df['num_adjectives'], df['num_verbs'] = zip(*df['comment_text_clean_split'].apply
                                                                 (lambda comment: tag_part_of_speech(comment)))
    df['percentage_nouns'] = df['num_nouns'] / df['num_words']
    df['percentage_adjs'] = df['num_adjectives'] / df['num_words']
    df['percentage_verbs'] = df['num_verbs'] / df['num_words']
    
    df = df[df['comment_text_clean'].apply(lambda x: type(x)==str)]
    df.drop('comment_text_clean_split', axis=1, inplace=True)
    return (df)

def word_freq(category, df):
    corpus = df[df[category]==1]['comment_text_clean']
    
    '''in the countvectorizer, get rid of english stop words, use lower case (default), use binary so 
    when count >1 it will be 1 ''' 
    vectorizer = CountVectorizer(stop_words = 'english', binary = True)
    x = vectorizer.fit_transform(corpus)
    word_freq = pd.DataFrame(x.toarray(), columns = vectorizer.get_feature_names())
    return word_freq

def top_freq_word(category,df):
    freq = word_freq(category, df)
    top_freq = pd.DataFrame(freq.sum(), columns = ['Freq']).sort_values(by = 'Freq', ascending = False)
 
    return top_freq.head(50)

def run_tfidf(train, test, ngram_min=1, ngram_max=2, min_df=5,
              max_features=20000, rm_stopwords=True, analyzer='word',
              sublinear_tf=False, token_pattern=r'(?u)\b\w\w+\b', binary=False,
              tokenize=False, tokenizer=None):
    
    rm_stopwords = 'english' if rm_stopwords else None
    strip_accents = 'unicode' if tokenize else None
    tfidf_vec = TfidfVectorizer(ngram_range=(ngram_min, ngram_max),
                                analyzer=analyzer,
                                stop_words=rm_stopwords,
                                strip_accents=strip_accents,
                                token_pattern=token_pattern,
                                tokenizer=tokenizer,
                                min_df=min_df,
                                max_features=max_features,
                                sublinear_tf=sublinear_tf,
                                binary=binary)
#     print('TFIDF ngrams ' + str(ngram_min) + ' to ' + str(ngram_max) + ' on ' +
#                str(analyzer) + ' with strip accents = ' + str(strip_accents) +
#                ', token_pattern = ' + str(token_pattern) + ', tokenizer = ' + str(tokenizer) +
#                ', rm_stopwords = ' + str(rm_stopwords) + ', min_df = ' + str(min_df) +
#                ', max_features = ' + str(max_features) + ', sublinear_tf = ' +
#                str(sublinear_tf) + ', binary = ' + str(binary))
    train_tfidf = tfidf_vec.fit_transform(train['comment_text_clean'])
    vocab       = tfidf_vec.get_feature_names()
    print (vocab)
    train_tfidf = pd.DataFrame(train_tfidf.toarray(), columns=vocab)

    if test is not None:
        test_tfidf = tfidf_vec.transform(test['comment_text_clean'])
        print('TFIDF train shape: {}'.format(train_tfidf.shape))
        print('TFIDF test shape: {}'.format(test_tfidf.shape))
    else:
        print('TFIDF train shape: {}'.format(train_tfidf.shape))
        test_tfidf = None
    return train_tfidf, test_tfidf

# LGB Model Definition
def runLGB(train_X, train_y, test_X, test_y, test_X2, label, dev_index, val_index):
    d_train = lgb.Dataset(train_X, label=train_y)
    d_valid = lgb.Dataset(test_X, label=test_y)
    watchlist = [d_train, d_valid]
    params = {'learning_rate': 0.05,
              'application': 'binary',
              'num_leaves': 31,
              'verbosity': -1,
              'metric': 'auc',
              'data_random_seed': 3,
              'bagging_fraction': 1.0,
              'feature_fraction': 0.4,
              'nthread': min(mp.cpu_count() - 1, 6),
              'lambda_l1': 1,
              'lambda_l2': 1}
    rounds_lookup = {'toxic': 1400,
                     'severe_toxic': 500,
                     'obscene': 550,
                     'threat': 380,
                     'insult': 500,
                     'identity_hate': 480}
    model = lgb.train(params,
                      train_set=d_train,
                      num_boost_round=rounds_lookup[label],
                      valid_sets=watchlist,
                      verbose_eval=10)
    print(model.feature_importance())
    pred_test_y = model.predict(test_X)
    pred_test_y2 = model.predict(test_X2)
    return pred_test_y, pred_test_y2