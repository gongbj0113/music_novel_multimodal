import csv
from typing import Dict, List, Optional

KEYWORD_DATA_PATH = 'data/csv/keywords.csv'

class KeywordIndexer:
    """
    A class for indexing keywords and transforming them to indices and vice versa.
    """

    def __init__(self) -> None:
        self.keyword_to_index: Dict[str, int] = {}
        self.index_to_keyword: List[str] = []
        self.keyword_to_english: Dict[str, str] = {}
        self.read_csv(KEYWORD_DATA_PATH)

    def read_csv(self, file_path: str) -> None:
        """
        Reads a CSV file containing keywords and adds them to the index.

        Args:
            file_path (str): The path to the CSV file.
        """
        with open(file_path, 'r') as file:
            reader = csv.reader(file)
            for row in reader:
                self.add_keyword(row[0], row[1])

    def add_keyword(self, keyword: str, english:str) -> None:
        """
        Adds a keyword to the index if it doesn't already exist.

        Args:
            keyword (str): The keyword to add.
        """
        if keyword not in self.keyword_to_index:
            index = len(self.index_to_keyword)
            self.keyword_to_index[keyword] = index
            self.index_to_keyword.append(keyword)
            self.keyword_to_english[keyword] = english

    def transform_text_to_index(self, keyword: str) -> Optional[int]:
        """
        Transforms a keyword to its corresponding index.

        Args:
            keyword (str): The keyword to transform.

        Returns:
            Optional[int]: The index of the keyword, or None if the keyword is not found.
        """
        if keyword in self.keyword_to_index:
            return self.keyword_to_index[keyword]
        else:
            return None
        
    def get_num_keywords(self) -> int:
        """
        Returns the number of keywords in the index.

        Returns:
            int: The number of keywords.
        """
        return len(self.index_to_keyword)

    def transform_index_to_text(self, index: int) -> Optional[str]:
        """
        Transforms an index to its corresponding keyword.

        Args:
            index (int): The index to transform.

        Returns:
            Optional[str]: The keyword corresponding to the index, or None if the index is out of range.
        """
        if index < len(self.index_to_keyword):
            return self.index_to_keyword[index]
        else:
            return None

    def transform_keyword_to_english(self, keyword: str) -> Optional[str]:
        """
        Transforms a keyword to its corresponding english.

        Args:
            keyword (str): The keyword to transform.

        Returns:
            Optional[str]: The english of the keyword, or None if the keyword is not found.
        """
        if keyword in self.keyword_to_english:
            return self.keyword_to_english[keyword]
        else:
            return None