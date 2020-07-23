import pandas as pd 
df=pd.read_csv('movie_data.csv')
df.head(10)
#output- df['review'][0]

import numpy as np 
from sklearn.feature_extraction.text import CountVectorizer 
count = CountVectorizer()

docs=np.array(['The sun is shining',
'The weather is sweet',
'The sun is shining, the weather is sweet, and one and one is two'])


bag = count.fit_transform(docs)
print(count.vocabulary_)
#RELEVANCY USING TF-IF (TERM FREQUENCY-INVERSE DOCUMENT FREQUENCY)
#smooth idf prevents divisions by 0 
#norm is the unit vector that is being used for the layers 
#use idf is classifying it into binary and non binary 

from sklearn.feature_extraction.text import TfidfTransformer 
tfidf=TfidfTransformer(use_idf=True, norm='l2',smooth_idf=True)
print(tfidf.fit_transform(count.fit_transform(docs)).toarray())

df.loc[0, 'review'][-50:]


import re
def preprocessor(text):
    text=re.sub('<[^>]*>','',text)
    emoticons = re.findall('(?::|;|=)(?:-)?(?:\)|\(|D|P)',text)
    text=re.sub('[\W]+',' ',text.lower()) +\
    ' '.join(emoticons).replace('-','')
    return text
    
    #OUTPUT FOR ^^
#preprocessor(df.loc[0,'review'][-50:])
    
   #TOKENIXATON OF DOCUMENTS  
from nltk.stem.porter import PorterStemmer
porter=PorterStemmer()
from nltk.stem.porter import PorterStemmer
porter=PorterStemmer()

def tokenizer(text):
    return text.split()

def tokenizer_porter(text):
    return [porter.stem(word) for word in text.split()]


from nltk.stem.porter import PorterStemmer
porter=PorterStemmer()


import nltk
nltk.download('stopwords')

from nltk.corpus import stopwords 
stop = stopwords.words('english')
[w for w in tokenizer_porter('a runner likes running and runs a lot')[-5:] if w not in stop]


#TRANSFORM TEXT DATA INTO TF IDF VECTORS 
from sklearn.feature_extraction.text import TfidfVectorizer 

tfidf = TfidfVectorizer(strip_accents=None,
                        lowercase=False,
                        preprocessor=None,
                        tokenizer=tokenizer_porter,
                        use_idf=True,
                        norm='l2',
                        smooth_idf=True)
y=df.sentiment.values
X=tfidf.fit_transform(df.review)


#############DOCUMENT CLASSIFICATION USING LOGISTIC REGRESSION"

from sklearn.model_selection import train_test_split 
X_train, X_test, y_train, y_test = train_test_split(X,y,random_state=1, test_size=0.5,
                                                                    shuffle=False)


import pickle 
from sklearn.linear_model import LogisticRegressionCV

clf=LogisticRegressionCV(cv=5,
                         scoring='accuracy',
                         random_state=0,
                         n_jobs=-1,
                         verbose=3,
                         max_iter=300).fit(X_train, y_train)

sent_analysis_model=open('sent_analysis_model.sav','wb')
pickle.dump(clf,sent_analysis_model)
sent_analysis_model.close()



filename='sent_analysis_model.sav'
saved_clf=pickle.load(open(filename,'rb'))

saved_clf.score(X_test, y_test)
