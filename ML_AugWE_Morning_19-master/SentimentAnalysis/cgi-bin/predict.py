from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import pandas as pd
import pickle as pkl

def textProcessing(df):
#     1. Tokenization
    tokens = []
    for i in range(len(df)):
        tokens.append(word_tokenize(df['Reviews'].iloc[i].lower()))
    
#     2. Remove Stopwords
    words = []
    eng_stopwords = stopwords.words('english')
    eng_stopwords.extend([',','?','!','@','.','-','&'])
    for tokenList in tokens:
        list_1 = []
        for i in range(len(tokenList)):
            if tokenList[i] not in eng_stopwords:
                list_1.append(tokenList[i])
        words.append(list_1)
    
#     3. Lemmatization
    wnet = WordNetLemmatizer()
    for i in range(len(words)):
        for j in range(len(words[i])):
            words[i][j] = wnet.lemmatize(words[i][j], pos='v')
            
    for i in range(len(words)):
        words[i] = ' '.join(words[i])
    
    return words

def test(review):
    file = open('tfidf.pkl','rb')
    tfidf = pkl.load(file)
    file.close()

    file = open('nb.pkl','rb')
    multi_nb = pkl.load(file)
    file.close()
    test_df = pd.DataFrame({"Reviews":[review]})
    test_wordsList = textProcessing(test_df)
    test_vect = tfidf.transform(test_wordsList)
    test_matrix = test_vect.toarray()

    pred = multi_nb.predict(test_matrix)
    return pred[0]