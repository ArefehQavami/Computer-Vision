import pytest
from report.report import Report
import yaml

class Test:

    @pytest.fixture
    def rp_instance(self):
        with open(r'E:\qa\basics\config.yml', 'r') as file:
            config = yaml.load(file, Loader=yaml.FullLoader)
        return Report(config)

    def test_qa_count(self, rp_instance):
        count = rp_instance.qa_count()
        count_col = rp_instance.collection.count_documents({})
        assert count == count_col

    def test_qa_count_category(self, rp_instance):
        count = rp_instance.qa_count_category()
        print(count)
        count_cat = [{'category_id': 3, 'category_name': 'ارزی-ریالی', 'count': 46}, {'category_id': 2, 'category_name': 'حقوقی', 'count': 49}, {'category_id': 1, 'category_name': 'تسهیلات', 'count': 14}, {'category_id': 0, 'category_name': 'بانکداری الکترونیک', 'count': 49}]
        assert count == count_cat