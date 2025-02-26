import re
import numpy as np
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer, WordNetLemmatizer
import nltk
from tqdm import tqdm_notebook
import pandas as pd


def process_dataset(name):
    path = f'./datasets/{name}.csv'

    pd_all = pd.read_csv(path)
    pd_all = pd_all.sample(frac=1, random_state=999)  # Shuffle

    pd_all['Title+Body'] = pd_all.apply(
        lambda row: row['Title'] + '. ' + row['Body'] if pd.notna(row['Body']) else row['Title'],
        axis=1
    )

    pd_tplusb = pd_all.rename(columns={
        "Unnamed: 0": "id",
        "class": "sentiment",
        "Title+Body": "text"
    })
    pd_tplusb.to_csv('Title+Body.csv', index=False, columns=["id", "Number", "sentiment", "text"])


def remove_stopwords(text):
    """Remove stopwords from the text."""
    NLTK_stop_words_list = stopwords.words('english')
    custom_stop_words_list = ['...']  # You can customize this list as needed
    final_stop_words_list = NLTK_stop_words_list + custom_stop_words_list
    return " ".join([word for word in str(text).split() if word not in final_stop_words_list])


def clean_str(string):
    """
    Clean text by removing non-alphanumeric characters,
    and convert it to lowercase.
    """
    string = re.sub(r"[^A-Za-z0-9(),.!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"\)", " ) ", string)
    string = re.sub(r"\(", " ( ", string)
    string = re.sub(r"\?", " ? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    string = re.sub(r"\\", "", string)
    string = re.sub(r"\'", "", string)
    string = re.sub(r"\"", "", string)
    # Todo: add `.lower()` to the code below to convert words to lowercase
    return string.strip()


def tokenize_words(texts, preprocess=""):
    nltk.download('punkt_tab', quiet=True)
    nltk.download('wordnet', quiet=True)
    max_len = 0
    tokenized_texts = []
    word2idx = {'<pad>': 0, '<unk>': 1}
    stemmer = SnowballStemmer('english')
    lemmatizer = WordNetLemmatizer()

    idx = 2
    for i, sentence in enumerate(texts):
        tokenized_sentence = word_tokenize(sentence)

        if preprocess == "stem":
            preprocessed_sentence = [stemmer.stem(word) for word in tokenized_sentence]
        elif preprocess == "lemmatize":
            preprocessed_sentence = [lemmatizer.lemmatize(word) for word in tokenized_sentence]
        else:
            preprocessed_sentence = tokenized_sentence

        tokenized_texts.append(preprocessed_sentence)

        for token in preprocessed_sentence:
            if token not in word2idx:
                word2idx[token] = idx
                idx += 1

        max_len = max(max_len, len(preprocessed_sentence))
    return tokenized_texts, word2idx, max_len


def pad_and_encode(tokenized_texts, word2idx, max_len):
    input_ids = []
    for tokenized_sentence in tokenized_texts:
        if len(tokenized_sentence) < max_len:
            tokenized_sentence += ['<pad>'] * (max_len - len(tokenized_sentence))
        else:
            tokenized_sentence = tokenized_sentence[:max_len]

        input_id = [word2idx.get(token) for token in tokenized_sentence]
        input_ids.append(input_id)

    return np.array(input_ids)


def load_pretrained_vectors(word2idx, file):
    print("loading vectors...")
    file_in = open(file, 'r', encoding='utf-8', newline='\n', errors='ignore')
    n, d = map(int, file_in.readline().split())

    embeddings = np.random.uniform(-0.25, 0.25, (len(word2idx), d))
    embeddings[word2idx['<pad>']] = np.zeros((d,))

    count = 0
    for line in tqdm_notebook(file_in):
        tokens = line.rstrip().split(' ')
        word = tokens[0]
        if word in word2idx:
            count += 1
            embeddings[word2idx[word]] = np.array(tokens[1:], dtype=np.float32)

    print(f"{count}/{len(word2idx)} pretrained vectors found")

    return embeddings
