from model.text_keyword_classifier import TextKeywordClassifier
from data.keyword_indexer import KeywordIndexer
from data.text_keyword_data import TextKeywordData

if __name__ == '__main__':
    keyword_indexer = KeywordIndexer()
    classifier = TextKeywordClassifier(keyword_indexer=keyword_indexer)

    # train
    text_keyword_data = TextKeywordData()
    text_keyword_data.filter(keyword_indexer=keyword_indexer)
    classifier.train(text_keyword_data=text_keyword_data)

    # predict
    text = 'I want to buy a new phone'
    keywords = classifier.predict(text=text)

    print(keywords)