import numpy as np
import pandas as pd
import sys
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.stem.porter import *
from nltk.corpus import stopwords
import string
import spacy
from numba import jit
import functools
import io
from sklearn.preprocessing import StandardScaler
# nltk.download('averaged_perceptron_tagger')
from nltk import pos_tag
import spacy.cli
# spacy.cli.download("en_core_web_lg")
import en_core_web_lg
embeddings = en_core_web_lg.load()
# nltk.download('stopwords')
# nltk.download('punkt')
ps = PorterStemmer()
stopwords_list = set(stopwords.words('english'))
from model import predict

"""#Preprocessing"""

contractions = { 
"ain't": "am not / are not / is not / has not / have not",
"aren't": "are not / am not",
"can't": "cannot",
"can't've": "cannot have",
"'cause": "because",
"could've": "could have",
"couldn't": "could not",
"couldn't've": "could not have",
"didn't": "did not",
"doesn't": "does not",
"don't": "do not",
"hadn't": "had not",
"hadn't've": "had not have",
"hasn't": "has not",
"haven't": "have not",
"he'd": "he had / he would",
"he'd've": "he would have",
"he'll": "he shall / he will",
"he'll've": "he shall have / he will have",
"he's": "he has / he is",
"how'd": "how did",
"how'd'y": "how do you",
"how'll": "how will",
"how's": "how has / how is / how does",
"I'd": "I had / I would",
"I'd've": "I would have",
"I'll": "I shall / I will",
"I'll've": "I shall have / I will have",
"I'm": "I am",
"I've": "I have",
"isn't": "is not",
"it'd": "it had / it would",
"it'd've": "it would have",
"it'll": "it shall / it will",
"it'll've": "it shall have / it will have",
"it's": "it has / it is",
"let's": "let us",
"ma'am": "madam",
"mayn't": "may not",
"might've": "might have",
"mightn't": "might not",
"mightn't've": "might not have",
"must've": "must have",
"mustn't": "must not",
"mustn't've": "must not have",
"needn't": "need not",
"needn't've": "need not have",
"o'clock": "of the clock",
"oughtn't": "ought not",
"oughtn't've": "ought not have",
"shan't": "shall not",
"sha'n't": "shall not",
"shan't've": "shall not have",
"she'd": "she had / she would",
"she'd've": "she would have",
"she'll": "she shall / she will",
"she'll've": "she shall have / she will have",
"she's": "she has / she is",
"should've": "should have",
"shouldn't": "should not",
"shouldn't've": "should not have",
"so've": "so have",
"so's": "so as / so is",
"that'd": "that would / that had",
"that'd've": "that would have",
"that's": "that has / that is",
"there'd": "there had / there would",
"there'd've": "there would have",
"there's": "there has / there is",
"they'd": "they had / they would",
"they'd've": "they would have",
"they'll": "they shall / they will",
"they'll've": "they shall have / they will have",
"they're": "they are",
"they've": "they have",
"to've": "to have",
"wasn't": "was not",
"we'd": "we had / we would",
"we'd've": "we would have",
"we'll": "we will",
"we'll've": "we will have",
"we're": "we are",
"we've": "we have",
"weren't": "were not",
"what'll": "what shall / what will",
"what'll've": "what shall have / what will have",
"what're": "what are",
"what's": "what has / what is",
"what've": "what have",
"when's": "when has / when is",
"when've": "when have",
"where'd": "where did",
"where's": "where has / where is",
"where've": "where have",
"who'll": "who shall / who will",
"who'll've": "who shall have / who will have",
"who's": "who has / who is",
"who've": "who have",
"why's": "why has / why is",
"why've": "why have",
"will've": "will have",
"won't": "will not",
"won't've": "will not have",
"would've": "would have",
"wouldn't": "would not",
"wouldn't've": "would not have",
"y'all": "you all",
"y'all'd": "you all would",
"y'all'd've": "you all would have",
"y'all're": "you all are",
"y'all've": "you all have",
"you'd": "you had / you would",
"you'd've": "you would have",
"you'll": "you shall / you will",
"you'll've": "you shall have / you will have",
"you're": "you are",
"you've": "you have"
}

contractions_re = re.compile('(%s)' % '|'.join(contractions.keys()))
def expand_contractions(s, contractions_dict=contractions):
    def replace(match):
        return contractions_dict[match.group(0)]
    return contractions_re.sub(replace, s)

# function to calculate the cosine
@jit(nopython=True)
def cosine_similarity_calc(vec_1:np.ndarray, vec_2:np.ndarray):
    sim = np.dot(vec_1, vec_2) / (np.linalg.norm(vec_1) * np.linalg.norm(vec_2))
    return sim

@jit(nopython=True)
def euclidean_dist(vec1:np.ndarray, vec2:np.ndarray):
    dist = np.linalg.norm(vec1 - vec2)
    return dist
# find the average cosine score of a sentence
def get_similarity_scores(i, sent_embeddings, sentences_in_text):
    score, sum_score = 0, 0
    dist, sum_dist = 0, 0
    ctr = 0
    if len(sentences_in_text[i]) == 0 or len(sentences_in_text) == 0:
        return (np.nan, np.nan, np.nan, np.nan)
    
    for j in range(len(sent_embeddings)):
        if i != j:
            # Calculate cosine similarity
            try:
              cos_sim = cosine_similarity_calc(sent_embeddings[i], sent_embeddings[j])
            except:
              return (np.nan, np.nan, np.nan, np.nan)
            score += (cos_sim / (len(sentences_in_text[i]) * len(sentences_in_text[j]))) # Normalized score
            sum_score += cos_sim
            
            # Calculate euclidean distance
            euclid = euclidean_dist(sent_embeddings[i], sent_embeddings[j])
            dist += (euclid / (len(sentences_in_text[i]) * len(sentences_in_text[j])))
            sum_dist += euclid
            
            ctr += 1
    if ctr == 0:
      return (score, np.nan, dist, np.nan)

    return (score, sum_score / ctr, dist, sum_dist / ctr)
# Calculate the TF-ISF score for the sentence
def tfisf(sentence, sentences):
    if len(sentence) == 0:
      return np.nan

    words_in_sentence = word_tokenize(sentence)
    words_in_sentence = [ps.stem(token) for token in words_in_sentence]
    
    freq = {}
    for word in words_in_sentence:
        if word in freq:
            freq[word] += 1
        else:
            freq[word] = 1
    score = 0
    for word in words_in_sentence:
        if word in string.punctuation:
            continue
            
        fs = 0
        for s in sentences:
            if word in s:
                fs += 1
        score += (freq[word] * np.log(len(sentences) / fs))
        
    return score / len(sentence)

def count_stopwords(sentence):
    ctr = 0
    words = word_tokenize(sentence)
    words = [ps.stem(word) for word in words if word not in string.punctuation]

    if len(words) == 0:
      return np.nan

    for word in words:
        if word in stopwords_list:
            ctr += 1
    
    return ctr / len(words)

def compute_similarity(dataset):
    df = {
        'title': [],
        'sentence': [],
        'sentence_position': [],
        'sentence_index': [],
        'length_ratio': [],
        'similarity_score': [],
        'mean_cosine_similarity': [],
        'euclid_dist': [],
        'mean_euclid_dist': [],
        'tfisf': [],
        'stopwords': [],
    }

    embeddings = spacy.load('en_core_web_lg')

    for i in range(0, dataset.shape[0]):
        row = dataset.iloc[i, :].values
        title = row[0].strip()
        text = row[1].strip()

        sentences_in_text = sent_tokenize(text)
        sentences_in_text = [expand_contractions(s) for s in sentences_in_text]
        sent_embeddings = [embeddings(s).vector for s in sentences_in_text]

        print(f'Processing row {i + 1} with {len(sentences_in_text)} sentences.')

        # Find the length of the longest sentence
        longest_sent_len = 0
        for s in sentences_in_text:
          cur = len(word_tokenize(s))
          if cur > longest_sent_len:
            longest_sent_len = cur
        
        # Perform stemming on the sentences after tokenizing them
        stemmed_sentences = []
        for s in sentences_in_text:
            s = expand_contractions(s)
            stemmed_sentences.append(set())
            tokens = word_tokenize(s)
            for token in tokens:
                stemmed_sentences[-1].add(ps.stem(token))
        
        for i in range(len(sentences_in_text)):
            s = sentences_in_text[i]
            df['title'].append(title)
            df['sentence'].append(s)

            # Find length ration
            df['length_ratio'].append(len(word_tokenize(s)) / longest_sent_len)
            
            # Find the similarity scores
            similarity_scores = get_similarity_scores(i, sent_embeddings, sentences_in_text)
            df['similarity_score'].append(similarity_scores[0])
            df['mean_cosine_similarity'].append(similarity_scores[1])
            
            df['euclid_dist'].append(similarity_scores[2])
            df['mean_euclid_dist'].append(similarity_scores[3])
            
            # Calculate TF-ISF
            df['tfisf'].append(tfisf(s, stemmed_sentences))
            
            # Count the number of stopwords in the sentence
            df['stopwords'].append(count_stopwords(s))
            
            
            df['sentence_index'].append((((i + 1) / len(sentences_in_text)) - 0.5) ** 2)
            if i == 0 or i == (len(sentences_in_text) - 1):
                df['sentence_position'].append(1)
            else:
                df['sentence_position'].append(0)
                
    return df

def tokenize_and_remove_punctuation(sentence):
    tokens = word_tokenize(sentence)
    return [t for t in tokens if len(t) > 1 or t.isalnum()]

def get_title_score(title, sentence):
    title_tokens = word_tokenize(title)
    title_tokens = [(ps.stem(s)).lower() for s in title_tokens if s not in string.punctuation]
    
    sent_tokens = word_tokenize(sentence)
    
    ctr = 0
    for s in sent_tokens:
        if (ps.stem(s)).lower() in title_tokens:
            ctr += 1
    return ctr / len(title_tokens)

def pos_tagging(sentence, counts):
    # verbs(VB), adjectives(JJ), nouns(NN), adverbs (RB)
    # verb - VB	verb (ask)
    # VBG - verb gerund (judging)
    # VBD - verb past tense (pleaded)
    # VBN - verb past participle (reunified)
    # VBP - verb, present tense not 3rd person singular(wrap)
    # VBZ
    pos_to_be_considered=['VB', 'JJ', 'NN', 'RB', 'WRB', 'PR', 'WP']
    text = word_tokenize(sentence)
    tokens_tag = pos_tag(text)
    num_verb=0
    num_adj=0
    num_noun=0
    num_pronoun=0
    num_adverb=0
    for token in tokens_tag:
        if token[1][:2] in pos_to_be_considered:
            if token[1][:2] == 'VB':
                num_verb+=1
            elif token[1][:2] == 'PR' or token[1][:2] == 'WP':
                num_pronoun+=1
            elif token[1][:2] == 'JJ':
                num_adj+=1
            elif token[1][:2] == 'NN':
                num_noun+=1
            else:
                num_adverb+=1
                
    counts['verbs'].append(num_verb / len(text))
    counts['adjectives'].append(num_adj / len(text))
    counts['nouns'].append(num_noun / len(text))
    counts['pronouns'].append(num_pronoun / len(text))
    counts['adverbs'].append(num_adverb / len(text))

def preprocess(paragraph,title):
  training_set = { 'Title': [], 'Text': []}
  training_set['Title'].append(title)
  training_set['Text'].append(paragraph)
  test = pd.DataFrame(training_set)
  dataset = pd.DataFrame(compute_similarity(test))
  dataset['sentence_length'] = [len(tokenize_and_remove_punctuation(s)) for s in dataset['sentence']]
  title_scores = []
  for i in range(dataset.shape[0]):
      title_scores.append(get_title_score(dataset.iloc[i, 0], dataset.iloc[i, 1]))
  dataset['title_feature'] = title_scores
  counts = {'verbs': [], 'adjectives': [], 'nouns': [], 'pronouns': [], 'adverbs': []}
  for i in range(0,dataset.shape[0]):
    pos_tagging(dataset.iloc[i, 1], counts)
  dataset['verbs'] = counts['verbs']
  dataset['adjectives'] = counts['adjectives']
  dataset['nouns'] = counts['nouns']
  dataset['pronouns'] = counts['pronouns']
  dataset['adverbs'] = counts['adverbs']
  return dataset
title = sys.argv[1].replace('\n', ' ')
para = sys.argv[2].replace('\n', ' ')
data = preprocess(para,title)
x_test = data.iloc[:,2:].values
def generate_summaries(df, y_pred):
    current_title = df['title'].iloc[0]
    current_text = ""
    current_pred_summary = ""
    pred_summary=[]
    for t, s, ps in zip(df.iloc[: , 0].values, df.iloc[:, 1].values, y_pred):

        if t==current_title:
            current_text+=s
            if ps==1:
                current_pred_summary+=s
        else: 
            pred_summary.append(current_pred_summary)
            current_title=t
            current_text=s

            if ps==1:
                current_pred_summary=s
            else:
                current_pred_summary=""

    if len(current_text)!=0:    
        pred_summary.append(current_pred_summary)
    
    return pred_summary

y_pred = predict(x_test)

summary = generate_summaries(data,y_pred)
fs = open("Text/summary.txt", "w")
fs.write(summary[0])
fs.close()