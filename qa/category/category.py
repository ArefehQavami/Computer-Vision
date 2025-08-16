import csv
import nltk
import collections
from bson.objectid import ObjectId
from typing import List, Dict
from basics.database import MongoDb
from basics.preprocess import Preprocess
from logger.elasticlogger import ElasticsearchLogger


class Category(Preprocess):
    """
        If the user does not know which category of questions to choose
        and then enter her/his question,
        we will suggest the appropriate category for her/his question.
    """
    _question_column = 'question'
    _answer_column = 'answer'
    _category_name = 'category_name'
    _word_frequency = 'word_frequency'

    def __init__(self, config: dict) -> str:

        """
           Represents a category with a question and associated keywords.

        """
        super().__init__()

        self.config = config

        self.db_obj = MongoDb(config)
        self.logger_error = ElasticsearchLogger(config, index_name='ai-app-logs-stage')

    def create_dict_of_cat_text(self) -> dict:
        """
            Create a dictionary where the keys indicate the category and the values indicate the question + answer.

            Returns:
                Dictionary of documents
        """
        try:
            all_qac = self.create_list_of_question_answer_cat()
            all_category = set(item[2] for item in all_qac)
            qa_of_each_cat = {}
            for cat in all_category:
                for items in all_qac:
                    if cat == items[2]:
                        qa_of_each_cat.setdefault(cat, []).append(self.common_clean(items[0]) + " " + self.common_clean(items[1]))

            return qa_of_each_cat

        except Exception as e:
            self.logger_error.log('error', f"An error occurred during the creation of the dictionary question + answer. {str(e)}")
            raise Exception(f"An error occurred during the creation of the dictionary question + answer. {str(e)}")


    def calculate_freq_of_keywords(self) -> Dict:

        """
            calculate the frequency of each word in documents belonging to each category.
            Then, we create a dictionary where the keys indicate the document category and
            the values indicate the most frequent words in that category.
        """
        try:
            keywords = collections.defaultdict(lambda: collections.defaultdict(int))
            qa_dict = self.create_dict_of_cat_text()
            for keys, values in qa_dict.items():
                normalized_doc = self.common_clean(' '.join(values))
                tokens = self.tokenizer(normalized_doc)
                cleaned_tokens = self.stopword_removal(tokens)
                token_lemmas = self.lemmatizer(cleaned_tokens)
                freq_dist = nltk.FreqDist(token_lemmas)
                for token, frequency in freq_dist.items():
                    if frequency >= self.config['thresh_freq']:
                        keywords[keys][token.replace("\u200c", " ")] += frequency
            return keywords

        except Exception as e:
            self.logger_error.log('ٍerror', f"An error occurred during the calculation of the word's frequency. {str(e)}")
            raise Exception(f"An error occurred during the calculation of the word's frequency. {str(e)}")

    def sorted_main_keywords(self):
        """
            sorted main dictionary by value.
            Returns:
                dict: A dictionary mapping categories to their main keywords.
        """
        keywords = self.calculate_freq_of_keywords()
        main_keywords = {}
        for category, terms in keywords.items():
            sorted_terms = dict(sorted(terms.items(), key=lambda x: x[1], reverse=False))
            main_keywords[str(category)] = sorted_terms
        return main_keywords

    def extract_keywords_of_question(self, user_question: str) -> List:

        """
            split user's question then Normalize and lemmatize the words of user question
            Return:
                   tokenized and lemmatized format of input question's words
        """

        question_qa = self.common_clean(user_question)
        question_tokens = self.tokenizer(question_qa)
        cleaned_question = self.stopword_removal(question_tokens)
        question_lemmas = self.lemmatizer(cleaned_question)
        self.logger_error.log('info', "Keywords Query completed successfully")
        return question_lemmas

    def find_matching_category(self, user_question) -> List:

        """
            Iterate through each category and its associated keywords and
            find the common lemmas between the question and category keywords then
            save the count of common lemmas for each category and finally
            find the category with the maximum number of common lemmas
            Attributes:
                user_question (str): The user's input question.
            Raises:
                ValueError: If no matching category is found.
            Returns:
                category with the maximum number
        """
        self.db_obj.connect(self.config['mongodb_db'])
        keyword_collection = self.db_obj.db[self.config['mongodb_keyword_coll']]
        question_lemmas = self.extract_keywords_of_question(user_question)

        if keyword_collection.count_documents({}) == 0:
            self.insert_all_keywords()
            self.logger_error.log('info', "all keywords inserted to keyword collection")

        keywords_dict = self.read_all_keywords()

        matches = {}
        for category, dict_freq_words in keywords_dict.items():
            keyword_lemmas = set(_keyword for _keyword in dict_freq_words.keys())
            common_lemmas = set(question_lemmas).intersection(keyword_lemmas)

            score = [rep for _wo, rep in list(dict_freq_words.items()) if _wo in list(common_lemmas)]
            matches[category] = sum(score)  # or len(common_lemmas), which one is better?

            max_value = max(matches.values())
            max_categories = [category for category, value in matches.items() if value == max_value and value != 0]

        if len(max_categories) != 0:
            return max_categories
        else:
            self.logger_error.log('info', "The user's question did not match any category!")


    def delete_all_categories(self) -> None:
        """
            Remove all data in the category collection of the MongoDB database.

            Raises:
                ValueError: If the category collection is already empty.
        """
        try:
            self.db_obj.connect(self.config['mongodb_db'])
            cat_collection = self.db_obj.db[self.config['mongodb_category_coll']]

            if cat_collection.count_documents({}) == 0:
                self.logger_error.log('info', "Category collection is already empty.")

            cat_collection.delete_many({})

        except Exception as e:
            self.logger_error.log('error', f'An error occurred while deleting all categories from the database: {e}')
            raise Exception(f'An error occurred while deleting all categories from the database: {e}')

    def show_cat(self) -> None:

        """
            Print all categories from the category collection of MongoDB.
        """

        try:
            self.db_obj.connect(self.config['mongodb_db'])
            cat_collection = self.db_obj.db[self.config['mongodb_category_coll']]

            if cat_collection.count_documents({}) == 0:
                self.logger_error.log('info', "Category collection is already empty.")

            cat_list = [cat for cat in cat_collection.find()]

            return cat_list

        except Exception as e:
            self.logger_error.log('error', f'An error occurred while accessing the category set: {e}')
            raise Exception(f'An error occurred while accessing the category set: {e}')

    def delete_category(self, cat_id: str) -> None:

        """
            remove one data in category collection of mongodb database
            Attributes:
                       cat_id: The category objectId that should be deleted
        """

        try:
            if not isinstance(cat_id, str):
                self.logger_error.log('info', "The input must be a string")

            self.db_obj.connect(self.config['mongodb_db'])
            cat_collection = self.db_obj.db[self.config['mongodb_category_coll']]

            document_id = ObjectId(cat_id)

            cat_collection.delete_one({'_id': document_id})

        except ValueError as e:
            self.logger_error.log('error', f'An error occurred while deleting one category from the database: {e}')
            raise Exception(f'An error occurred while deleting one category from the database: {e}')

    def update_categories_collection(self, cat_id: str, cat_name: str) -> None:

        """
            read category collection and then modified input key and value after that insert modified dict.
            Attributes:
                       cat_id(str): The category id that needs to be corrected
                       cat_name(str): The name of category that needs to be corrected
        """

        try:
            if not (isinstance(cat_id, str) and isinstance(cat_name, str)):
                self.logger_error.log('info', "The input must be a string")

            self.db_obj.connect(self.config['mongodb_db'])
            cat_collection = self.db_obj.db[self.config['mongodb_category_coll']]
            qa_collection = self.db_obj.db[self.config['mongodb_coll']]

            document = cat_collection.find_one({'_id': ObjectId(cat_id)})
            find_value = document[self._category_name]

            cat_collection.update_one({'_id': ObjectId(cat_id)}, {'$set': {self._category_name: cat_name}})
            qa_collection.update_many({self._category_name: find_value}, {'$set': {self._category_name: cat_name}})

        except Exception as e:
            self.logger_error.log('error', f'An error occurred while updating category collection: {e}')
            raise Exception(f'An error occurred while updating category collection: {e}')

    def insert_categories_to_collection(self, cat_name: str) -> None:

        """
            Insert all categories with category name format (str) to the category collection of MongoDB.

            Args:
                cat_name (str): category names.

            Raises:
                ValueError: If any of the category names already exist in the database.
        """
        try:
            if not isinstance(cat_name, str):
                self.logger_error.log('info', "The input must be a string")

            self.db_obj.connect(self.config['mongodb_db'])
            cat_collection = self.db_obj.db[self.config['mongodb_category_coll']]

            existing_categories = cat_collection.distinct(self._category_name)
            if cat_name in existing_categories:
                self.logger_error.log('info', f"Category name '{cat_name}' already exists in the database.")
            else:
                item = {self._category_name: cat_name}
                cat_collection.insert_one(item)

        except Exception as e:
            self.logger_error.log('error', f'An error occurred while inserting data to category collection: {e}')
            raise Exception(f'An error occurred while inserting data to category collection: {e}')

    def get_question_by_category(self, cat_name: str) -> None:

        """
            Returns a list of question texts with the specified category.
            Args:
                cat_name (str): The category to search for.
            Returns:
                list: A list of question texts with the specified category.
        """
        try:
            if not isinstance(cat_name, str):
                self.logger_error.log('info', "The input must be a string")

            self.db_obj.connect(self.config['mongodb_db'])
            questions_collection = self.db_obj.db[self.config['mongodb_coll']]

            questions = questions_collection.find({self._category_name: cat_name})
            question_texts = [question for question in questions]
            return question_texts

        except Exception as e:
            self.logger_error.log('error', f'Error retrieving during get questions by category: {e}')
            raise Exception(f'Error retrieving during get questions by category: {e}')

    def create_list_of_question_answer_cat(self) -> List[List]:
        """
            Create a 2D list that includes all questions, answers, and categories of the saved data,
            Retrieve all documents from the collection,
            Convert documents to a list

            Returns:
                data(list[list]): list of data
        """
        self.db_obj.connect(self.config['mongodb_db_main'])
        coll = self.db_obj.db[self.config['mongodb_coll']]

        if coll.count_documents({}) == 0:
            self.logger_error.log('info', "qa Collection is already empty")

        documents = coll.find()

        data = [[document[self._question_column], document[self._answer_column], document[self._category_name]] for
                document in documents]
        return data

    def insert_all_keywords(self) -> None:

        """
            Insert extracted keywords with the frequency of each document of a category to the keyword collection in the MongoDB database.
        """
        try:
            self.db_obj.connect(self.config['mongodb_db'])
            keyword_dict = self.sorted_main_keywords()

            if not isinstance(keyword_dict, dict):
                self.logger_error.log('info', "The input must be a dictionary")

            for key, value in keyword_dict.items():
                keyword_doc = {self._category_name: key, self._word_frequency: value}
                self.db_obj.db[self.config['mongodb_keyword_coll']].insert_one(keyword_doc)

        except Exception as e:
            self.logger_error.log('error', f"An error occurred during insert keywords to keyword collection:{str(e)}")
            raise Exception(f"An error occurred during insert keywords to keyword collection:{str(e)}")

    def show_keywords(self) -> None:

        """
            Print all keywords from the keyword collection of MongoDB.
        """
        try:
            self.db_obj.connect(self.config['mongodb_db'])
            key_collection = self.db_obj.db[self.config['mongodb_keyword_coll']]

            if key_collection.count_documents({}) == 0:
                self.logger_error.log('info', "Keyword collection is already empty")

            key_list = [key for key in key_collection.find()]
            return key_list

        except Exception as e:
            self.logger_error.log('error', f'Error accessing keyword collection: {e}')
            raise Exception(f'Error accessing keyword collection: {e}')

    def delete_keyword_collection(self) -> None:
        """
            Delete all content of one collection in the MongoDB database:
            Connect to MongoDB, select database and collection, drop the collection.
        """
        try:
            self.db_obj.connect(self.config['mongodb_db'])
            key_collection = self.db_obj.db[self.config['mongodb_keyword_coll']]
            if key_collection.count_documents({}) == 0:
                self.logger_error.log('info', "keyword Collection is already empty")
            else:
                key_collection.drop()

        except Exception as e:
            self.logger_error.log('error', f"An error occurred during the deletion keyword collection: {str(e)}")
            raise Exception(f"An error occurred during the deletion keyword collection: {str(e)}")

    def read_all_keywords(self) -> Dict:
        """
            Read keywords of each category from the keyword collection.
            Returns: dictionary including category and keywords (keys and values).
        """
        try:
            self.db_obj.connect(self.config['mongodb_db'])
            collection = self.db_obj.db[self.config['mongodb_keyword_coll']]

            if collection.count_documents({}) == 0:
                self.logger_error.log('info', "Keyword collection is already empty.")

            data_dict = {item[self._category_name]: item[self._word_frequency] for item in collection.find()}
            return data_dict

        except Exception as e:
            self.logger_error.log('error', f"An error occurred during read keywords collection: {str(e)}")
            raise Exception(f"An error occurred during read keywords collection: {str(e)}")

    def read_category_from_csv_write_to_db(self, path, category_column_name):
        """
            This module reads all categories from the new csv file and writes them to the mongodb database.
            Args:
                path (str): path of csv file.
                category_column_name (str): The name of the column from which the values will be extracted.
        """
        try:
           with open(path, 'r', encoding='utf-8') as csvfile:
               csv_reader = csv.DictReader(csvfile, delimiter=',')

               if category_column_name not in csv_reader.fieldnames:
                    self.logger_error.log('info', "Column name not found in the CSV file.")

               column_values = [row[category_column_name] for row in csv_reader]

           for _cat in column_values:
               self.insert_categories_to_collection(_cat)

        except Exception as e:

           self.logger_error.log('error', f"An error occurred during read category from csv file and write to db: {str(e)}")
           raise Exception(f"An error occurred during read category from csv file and write to db: {str(e)}")




