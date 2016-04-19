from bs4 import BeautifulSoup as bs
import re
import os
import operator
import numpy as np
from stemming.porter2 import stem
import pickle

stopwords = set(['doi','wikipedia','all', 'just', 'being', 'over', 'both', 'through', 'yourselves', 'its', 'before', 'herself', 'had', 'should', 'to', 'only', 'under', 'ours', 'has', 'do', 'them', 'his', 'very', 'they', 'not', 'during', 'now', 'him', 'nor', 'did', 'this', 'she', 'each', 'further', 'where', 'few', 'because', 'doing', 'some', 'are', 'our', 'ourselves', 'out', 'what', 'for', 'while', 'does', 'above', 'between', 't', 'be', 'we', 'who', 'were', 'here', 'hers', 'by', 'on', 'about', 'of', 'against', 's', 'or', 'own', 'into', 'yourself', 'down', 'your', 'from', 'her', 'their', 'there', 'been', 'whom', 'too', 'themselves', 'was', 'until', 'more', 'himself', 'that', 'but', 'don', 'with', 'than', 'those', 'he', 'me', 'myself', 'these', 'up', 'will', 'below', 'can', 'theirs', 'my', 'and', 'then', 'is', 'am', 'it', 'an', 'as', 'itself', 'at', 'have', 'in', 'any', 'if', 'again', 'no', 'when', 'same', 'how', 'other', 'which', 'you', 'after', 'most', 'such', 'why', 'a', 'off', 'i', 'yours', 'so', 'the', 'having', 'once'
])

feature_size = 101
train_ratio = 0.6
valid_ratio = 0.3
test_ratio = 0.1

# Extract the text from html
def gettextonly(soup):
    v = soup.body.findAll(text = True)
    return v

# Separate the words by any non-whitespace character
def separatewords(text):
    text_s = ''
    for str in text:
        text_s += str
        splitter=re.compile('[^A-Za-z]')
    return [s.lower() for s in splitter.split(text_s) if s!='']

# Load each page, extract the text, separate into words, and save into a dictionary
def load_file(folder):
    """Load the data for a single letter label."""
    page_files = os.listdir(folder)
    
    word_dic_set = []
    
    print(folder)
    for page in page_files:
        page_file = os.path.join(folder, page)
        print(page_file)
        try:
            if page_file == "training/positive/all-urls.txt":
                continue
            f = open(page_file, 'r')
            page_info = f.read()
            f.close()
        
            soup = bs(page_info, "html.parser")

            text = gettextonly(soup)
            words = separatewords(text)
            word_dic = {}
            for i in range(len(words)):
                if words[i] in stopwords:
                    continue
                else:
                    stem_word = stem(words[i])
                    if stem_word in word_dic:
                        word_dic[stem_word] += 1
                    else:
                        word_dic[stem_word] = 1
            
            word_dic_set.append(word_dic)
        except IOError as e:
            print('Could not read:', page_file, ':', e, '- it\'s ok, skipping.')
    return word_dic_set

# Count the frequency of each word among all postive/negative pages, and sort words by frequency
# return sorted words 
def find_high_frq_words(dic_list):
    word_dic_total = {}
    
    for dic in dic_list:
        for word in dic:
            if word in word_dic_total:
                word_dic_total[word] += dic[word]
            else:
                word_dic_total[word] = dic[word]
    
    high_frq_words = sorted(word_dic_total.keys(), key = word_dic_total.get, reverse = True)                 
    
    return high_frq_words

# Generate keywords by finding first N (N is controlled by feature_size) words that appeared in
# sorted positive words list, not in sorted negative words list
def generate_key_words(high_frq_p, high_frq_n):
    key_words = []
    high_frq_n = set(high_frq_n)
    for word in high_frq_p:
        if word not in high_frq_n and len(word) > 1:
            key_words.append(word)
        if len(key_words) == feature_size - 1:
            return key_words

# Generate features for each sample (page) by counting the frequency of each keyword appeared in the page
def form_features(dic_list, keywords):
    dataset = np.ndarray(shape=(len(dic_list), feature_size), dtype=np.float32)
    for ind in range(0, len(dic_list)):
        for f_i in range(0, feature_size - 1):
            if keywords[f_i] in dic_list[ind]:
                dataset[ind, f_i] = dic_list[ind][keywords[f_i]]
            else:
                dataset[ind, f_i] = 0
                
        # Add one more feature that evaluate the uniquness of the page
        frq_words = sorted(dic_list[ind].keys(), key = dic_list[ind].get, reverse = True) 
        high_frq_words = []
        for i in range(0, 10):
            if (len(frq_words[i]) > 1):
                high_frq_words.append(frq_words[i])
        dif_words = set(high_frq_words) - set(keywords);
        dataset[ind, feature_size - 1] = len(dif_words)
    return dataset

# Merge postive and negative dataset, and generate label for each sample
def merge_datasets_and_labels(d_p, d_n):
    
    dataset = np.concatenate((d_p, d_n))
    labels = np.zeros(d_p.shape[0] + d_n.shape[0])
    labels[0 : d_p.shape[0]] = 1
    
    return dataset, labels

# Shuffle dataset
def randomize(dataset, labels):
    permutation = np.random.permutation(labels.shape[0])
    shuffled_dataset = dataset[permutation,:]
    shuffled_labels = labels[permutation]
    return shuffled_dataset, shuffled_labels

# Given ratio, split dataset into train, valid, and test dataset
def split_dataset(shuffled_dataset, shuffled_labels, train_ratio, valid_ratio, test_ratio):
    n_train = int(shuffled_dataset.shape[0] * train_ratio)
    n_valid = int(shuffled_dataset.shape[0] * valid_ratio)
    train_dataset = shuffled_dataset[0 : n_train, :]
    train_labels = shuffled_labels[0 : n_train]
    valid_dataset = shuffled_dataset[n_train : n_train + n_valid, :]
    valid_labels = shuffled_labels[n_train : n_train + n_valid]
    test_dataset = shuffled_dataset[n_train + n_valid :, :]
    test_labels = shuffled_labels[n_train + n_valid :]
    
    return train_dataset, train_labels, valid_dataset, valid_labels, test_dataset, test_labels


# Prepare data for classification
def main():
    data_p = load_file('training/positive')
    data_n = load_file('training/negative')
    
    high_frq_words_p = find_high_frq_words(data_p)
    high_frq_words_n = find_high_frq_words(data_n)

    key_words = generate_key_words(high_frq_words_p, high_frq_words_n[0 : feature_size])
    
    # Save keywords list into file
    pickle_file = 'key_words.pickle'
    try:
        f = open(pickle_file, 'wb')
        save = {
            'keywords': key_words,
        }
        pickle.dump(save, f, pickle.HIGHEST_PROTOCOL)
        f.close()
    except Exception as e:
        print('Unable to save data to', pickle_file, ':', e)
        raise
    
    dataset_p = form_features(data_p, key_words)
    dataset_n = form_features(data_n, key_words)
    
    dataset, labels = merge_datasets_and_labels(dataset_p, dataset_n)
    
    shuffled_dataset, shuffled_labels = randomize(dataset, labels)
    
    train_dataset, train_labels, valid_dataset, valid_labels, test_dataset, test_labels = split_dataset(
    shuffled_dataset, shuffled_labels, train_ratio, valid_ratio, test_ratio)

    pickle_file = 'data.pickle'

    try:
        f = open(pickle_file, 'wb')
        save = {
            'train_dataset': train_dataset,
            'train_labels': train_labels,
            'valid_dataset': valid_dataset,
            'valid_labels': valid_labels,
            'test_dataset': test_dataset,
            'test_labels': test_labels,
        }
        pickle.dump(save, f, pickle.HIGHEST_PROTOCOL)
        f.close()
    except Exception as e:
        print('Unable to save data to', pickle_file, ':', e)
        raise
    
    


