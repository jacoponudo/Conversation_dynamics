import sys
module_path = '/Users/jacoponudo/Documents/thesis/src/PRO'
sys.path.append(module_path)

import pandas as pd

def add_percentile_column(group):
    group['percentile'] = (group['created_at'].rank(pct=True) * 100).round(2)
    return group

from tqdm import tqdm

def count_unique_words(text):
    if isinstance(text, str):
        words = text.split()
        return len(set(words))
    else:
        return 0




import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from nltk.tag import pos_tag
from tqdm import tqdm
import pandas as pd
import numpy as np

nltk.download('stopwords')
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')

ps = PorterStemmer()
stop_words = set(stopwords.words('english'))



def preprocess_text(text):
    # Tokenize the text into words
    words = word_tokenize(text.lower())
    
    # Tag words with their part of speech
    tagged_words = pos_tag(words)
    
    # Define a function to filter out unwanted words
    def is_word_of_interest(word):
        # Retain adjectives, nouns, verbs, adverbs, and technical terms
        if word[1] in ['JJ', 'JJR', 'JJS', 'NN', 'NNS', 'NNP', 'NNPS', 'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ', 'RB', 'RBR', 'RBS']:
            return True
        # Exclude pronouns, prepositions, conjunctions, and stopwords
        elif word[1] in ['PRP', 'PRP$', 'WP', 'WP$', 'IN', 'CC'] or word[0].lower() in stop_words:
            return False
        # Retain words that are all uppercase (likely to be technical terms or acronyms)
        elif word[0].isupper():
            return True
        # Retain words that contain numbers (likely to be technical terms)
        elif any(char.isdigit() for char in word[0]):
            return True
        else:
            return False
    
    # Filter out unwanted words
    filtered_words = [word[0] for word in tagged_words if is_word_of_interest(word)]
    
    # Stem the filtered words
    stemmed_words = [ps.stem(word) for word in filtered_words if word.isalnum()]
    
    return stemmed_words



from tqdm import tqdm

def calculate_unique_word_ratio(data):
    unique_word_ratios = {}
    unique_words_users={}
    for user in tqdm(data['user'].unique()):
        df = data[data['user'] == user]
        unique_words = set()
        total_words = 0
        texts = df['text'][df['text'].apply(lambda x: isinstance(x, str))]
        processed_texts = texts.apply(preprocess_text)
        unique_words.update([word for sublist in processed_texts for word in sublist])
        total_words = sum(len(text) for text in processed_texts)
        unique_words = len(unique_words) 
        unique_words_users[user] = unique_words
        total_words_users[user]=total_words
    
    return unique_words_users,total_words_users

