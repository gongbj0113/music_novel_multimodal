from model.text_keyword_classifier import TextKeywordClassifier
from data.keyword_indexer import KeywordIndexer
from data.text_keyword_data import TextKeywordData

from model.text_generator_from_representation import TextGeneratorFromRepresentation
from data.representation_text import RepresentationTextData

from model.representation_generator_from_music import RepresentationGeneratorFromMusic
from data.music_representation_data import MusicRepresentationData

import music_representation_data_generator as mr_data_gen

def test_text_keyword_classifier():
    keyword_indexer = KeywordIndexer()
    classifier = TextKeywordClassifier(keyword_indexer=keyword_indexer)

    # train
    text_keyword_data = TextKeywordData()
    text_keyword_data.filter(keyword_indexer=keyword_indexer)
    test_data = classifier.train(text_keyword_data=text_keyword_data)

    # predict
    texts = [
        '나는 오늘 밥을 먹었다.',
        '나는 오늘 밥을 먹지 않았다.',
        '나는 너를 사랑해.',
        '나는 너를 사랑하지 않아.',
        '나는 가슴이 설렌다.',
        '나는 암울하다.',
        '나는 행복하다.',
        '나는 슬프다.',
        '나는 우울하다.',
        '나는 신난다.',
    ]


    for (text, keyword) in test_data:
        output = classifier.predict(text=text)
        print(text + " >> " + keyword + " >> " + output)

    for text in texts:
        keywords = classifier.predict(text=text)
        print(text + " >> " + keywords)

def test_text_generator_from_representation():
    representation_text_data = RepresentationTextData()
    generator = TextGeneratorFromRepresentation()

    # train
    generator.train(representation_text_data=representation_text_data)

    # predict
    representation = '비범'
    text = generator.predict(representation=representation)
    print(representation + " >> " + text)

def test_representation_generator_from_music():
    music_representation_data = MusicRepresentationData()
    generator = RepresentationGeneratorFromMusic()

    # train
    generator.train(music_representation_data=music_representation_data)

    # predict
    music_path = 'data/music/DwTvH.mp3'
    representation = generator.predict(music_path=music_path)
    print(music_path + " >> " + representation)

def test_text_keyword_valid():
    keyword_indexer = KeywordIndexer()
    classifier = TextKeywordClassifier(keyword_indexer=keyword_indexer)

    # train
    text_keyword_data = TextKeywordData()
    text_keyword_data.filter(keyword_indexer=keyword_indexer)
    # classifier.train(text_keyword_data=text_keyword_data)

    # predict
    for i in range(100):
        text, keyword = text_keyword_data[i]
        predict = classifier.predict(text=text)
        print(text + " >> " + keyword + " >> " + predict)

def test_music_representation_data_generator():
    mr_data_gen.generate()