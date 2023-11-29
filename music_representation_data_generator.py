from model.text_keyword_classifier import TextKeywordClassifier
from data.keyword_indexer import KeywordIndexer

from data.music_keyword import MusicKeyword
from data.text_data import TextData
from model.representation_generator_from_music import RepresentationGeneratorFromMusic


def generate():
    keyword_indexer = KeywordIndexer()
    classifier = TextKeywordClassifier(keyword_indexer=keyword_indexer)

    music_keyword = MusicKeyword()
    music_keyword.download()

    text_data = TextData()
    music_cap = RepresentationGeneratorFromMusic()
    music_cap.loaded = True
    
    music_representation_text_data = []


    for i, text in enumerate(text_data):
        try:
            keyword = classifier.predict(text=text)
            if keyword is None:
                continue
            english_keyword = keyword_indexer.transform_keyword_to_english(keyword)
            if english_keyword is None:
                continue
            music_path = music_keyword.get_random_by_keyword(keyword=keyword)
            representation = music_cap.predict(music_path=music_path)

            print("Status: " + str(i) + " / " + str(len(text_data)))
        except:
            continue

        music_representation_text_data.append((music_path, english_keyword + ", " + representation, text))

    
    # save to csv file
    # head : music,representation,text
    # path : data/csv/music_representation_text.csv
    import csv

    csv_file_path = "data/csv/music_representation_text.csv"
    with open(csv_file_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["music", "representation", "text"])
        writer.writerows(music_representation_text_data)