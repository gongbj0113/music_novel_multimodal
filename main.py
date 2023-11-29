from model.text_keyword_classifier import TextKeywordClassifier
from data.keyword_indexer import KeywordIndexer
from data.text_keyword_data import TextKeywordData

from model.text_generator_from_representation import TextGeneratorFromRepresentation
from data.representation_text import RepresentationTextData

from model.representation_generator_from_music import RepresentationGeneratorFromMusic
from data.music_representation_data import MusicRepresentationData

import music_representation_data_generator as mr_data_gen

import test as test

def train_all():
    # Train Text keyword classifier
    print("Train Text keyword classifier")
    keyword_indexer = KeywordIndexer()
    classifier = TextKeywordClassifier(keyword_indexer=keyword_indexer)

    text_keyword_data = TextKeywordData()
    text_keyword_data.filter(keyword_indexer=keyword_indexer)
    classifier.train(text_keyword_data=text_keyword_data)

    # Generate Music representation data
    print("Generate Music representation data")
    mr_data_gen.generate()

    # Train Text generator from representation
    print("Train Text generator from representation")
    representation_text_data = RepresentationTextData()
    generator = TextGeneratorFromRepresentation()
    generator.train(representation_text_data=representation_text_data)

    # Train Representation generator from music
    print("Train Representation generator from music")
    music_representation_data = MusicRepresentationData()
    generator = RepresentationGeneratorFromMusic()
    generator.train(music_representation_data=music_representation_data)

    print("Done")

def run(music_path):
    representation_generator = RepresentationGeneratorFromMusic()
    text_generator = TextGeneratorFromRepresentation()

    representation = representation_generator.predict(music_path=music_path)
    text = text_generator.predict(representation=representation)

    return text


if __name__ == '__main__':
    # train_all()

    # test.test_text_keyword_classifier()
    test.test_music_representation_data_generator()
    # test.test_representation_generator_from_music()
    # test.test_text_generator_from_representation()