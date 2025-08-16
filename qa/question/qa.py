from bson import ObjectId
from basics.database import MongoDb
from pymongo.errors import CollectionInvalid
from logger.elasticlogger import ElasticsearchLogger
import csv
class QA:

    def __init__(self, config):

        obj_db = MongoDb(config, use_config=True)
        obj_db.connect(config['mongodb_db'])
        self.collection = obj_db.db[config['mongodb_coll']]
        self._validate_collection_existence()
        self.logger = ElasticsearchLogger(config, config['elastic_index_name'])


    def add_qa(self, question: str, answer: str, category_id: int, category_name: str):
        """
        add question and answer and category_id and category_name to database
        """
        self._validate_string_inputs([question, answer, category_name])
        self._validate_integer_input(category_id)
        self._validate_not_empty_inputs([question, answer, category_name, category_id])
        single_document = {
            'question': question,
            'answer': answer,
            'category_id': category_id,
            'category_name': category_name
        }
        self.collection.insert_one(single_document)

    def edit_qa(self, id, question=None, answer=None, category_id=None, category_name=None):
        """
        edit question or answer or category based on id
        """
        self._validate_object_id(id)
        self._validate_document_existence(id)
        self._validate_string_inputs([question, answer, category_name])
        self._validate_integer_input(category_id)
        update_values = {}
        if question is not None and question != "":
            update_values["question"] = question
        if answer is not None and answer != "":
            update_values["answer"] = answer
        if category_id is not None:
            update_values["category_id"] = category_id
        if category_name is not None and category_name != "":
            update_values["category_name"] = category_name
        if question is None and answer is None and category_id is None and category_name is None:
            self.logger.log("error", "All variables are None.")
            raise ValueError("تمام متغیرها بدون مقدار هستند.")
        self.collection.update_one({"_id": ObjectId(id)}, {"$set": update_values})

    def delete_qa(self, id):
        """
        delete a row of database based on id
        """
        self._validate_object_id(id)
        self._validate_document_existence(id)
        result = self.collection.delete_one({'_id': ObjectId(id)})
        if result.deleted_count == 0:
            self.logger.log("error", "Failed to delete the document")
            raise Exception("حذف رکورد با خطا مواجه شد.")

    def delete_all_qa(self):
        """
        delete all records of database
        """
        self._validate_non_empty_collection()
        self.collection.delete_many({})

    def get_all_qa(self):
        """
        retrieve list of all records
        """
        self._validate_non_empty_collection()
        documents = self.collection.find()
        data_list = []
        for document in documents:
            data_list.append(document)
        return data_list

    def _validate_object_id(self, id):
        if not ObjectId.is_valid(id):
            self.logger.log("error", "Invalid ObjectId")
            raise ValueError("آیدی معتبر نمی باشد.")

    def _validate_document_existence(self, id):
        if not self.collection.find_one({"_id": ObjectId(id)}):
            self.logger.log("error", "Document with ObjectId {} does not exist".format(id))
            raise ValueError(" رکوردی با این آیدی {} وجود ندارد. " .format(id))

    def _validate_collection_existence(self):
        if self.collection is None:
            self.logger.log("error", "Invalid collection")
            raise CollectionInvalid("کالکشن معتبر نمی باشد.")

    def _validate_string_inputs(self, values):
        if any(value is not None and not isinstance(value, str) for value in values):
            self.logger.log("error", "Question, answer, and category_name must be strings")
            raise TypeError("سوال و جواب و نام دسته بندی می بایست رشته متنی باشند.")

    def _validate_integer_input(self, value):
        if value is not None and not isinstance(value, int):
            self.logger.log("error", "Category_id must be an integer")
            raise TypeError("آیدی دسته بندی می بایست عدد باشد.")

    def _validate_not_empty_inputs(self, values):
        if any(value is None or value == "" for value in values):
            self.logger.log("error", "Question, answer, category_id and category_name can not be None")
            raise ValueError("سوال و جواب و آیدی و نام دسته بندی نمی توانند بدون مقدار باشند.")

    def _validate_non_empty_collection(self):
        if self.collection.count_documents({}) == 0:
            self.logger.log("error", "Empty collection")
            raise CollectionInvalid("کالکشن پایگاه داده خالی است.")

    def add_all_qa(self, csv_path):
        """
        add all data in csv to database collection
        """
        required_columns = ['question', 'answer', 'category_id', 'category_name']
        with open(csv_path, 'r',encoding="utf8") as file:
            reader = csv.DictReader(file)
            headers = reader.fieldnames
            if not all(column in headers for column in required_columns):
                missing_columns = set(required_columns) - set(headers)
                raise TypeError(f"Error: The following required column(s) are missing: ", missing_columns)
            for row in reader:
                self.collection.insert_one(row)

    def get_qa_by_id(self, id):
        """
        search a record with id
        """
        self._validate_object_id(id)
        self._validate_document_existence(id)
        qa = self.collection.find_one({'_id': ObjectId(id)})
        return qa

    def search_qa(self, keyword):
        """
        find question-answers based on keywords
        """
        qa = self.collection.find({'$text': {'$search': keyword}})
        search_result = []
        for document in qa:
            search_result.append(document)
        return search_result
