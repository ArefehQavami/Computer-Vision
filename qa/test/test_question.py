import pytest
from bson import ObjectId
from question.qa import QA
import yaml

class Test:
    @pytest.fixture
    def qa_instance(self):
        with open(r'E:\qa\basics\config.yml', 'r') as file:
            config = yaml.load(file, Loader=yaml.FullLoader)
        return QA(config)

    @pytest.mark.parametrize("question, answer, category_id, category_name", [
        ("سوال تست", "جواب تست", 5, "تست")
    ])
    def test_add_qa(self, qa_instance, question, answer, category_id, category_name):
        qa_instance.add_qa(question, answer, category_id, category_name)
        inserted_document = qa_instance.collection.find_one(sort=[("_id", -1)])
        assert inserted_document is not None
        assert inserted_document['question'] == question
        assert inserted_document['answer'] == answer
        assert inserted_document['category_id'] == category_id
        assert inserted_document['category_name'] == category_name

    @pytest.mark.parametrize("question, answer, category_id, category_name", [
        ("سوال تست آپدیت شده", None, None, None),
        (None, "جواب تست آپدیت شده", None, None),
        (None, None, 5, "تست آپدیت شده"),
    ])
    def test_edit_qa(self, qa_instance, question, answer, category_id, category_name):
        inserted_document = qa_instance.collection.find_one(sort=[("_id", -1)])
        qa_instance.edit_qa(inserted_document['_id'], question, answer, category_id, category_name)
        inserted_document = qa_instance.get_qa_by_id(inserted_document['_id'])
        if question is not None:
            assert inserted_document['question'] == question
        if answer is not None:
            assert inserted_document['answer'] == answer
        if category_id is not None:
            assert inserted_document['category_id'] == category_id
        if category_name is not None:
            inserted_document['category_name'] == category_name

    def test_delete_qa(self, qa_instance):
        inserted_document = qa_instance.collection.find_one(sort=[("_id", -1)])
        qa_instance.delete_qa(id=inserted_document['_id'])
        assert qa_instance.collection.find_one({'_id': ObjectId(inserted_document['_id'])}) is None

    def test_get_id(self, qa_instance):
        record = qa_instance.collection.find_one()
        id = record["_id"]
        get_document = qa_instance.get_qa_by_id(id)
        assert get_document == record
