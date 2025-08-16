import pytest
import yaml
from category.category import Category

with open(r'C:\Users\m.gheysari\PycharmProjects\pythonProject24\config.yml', 'r') as file:
    config = yaml.load(file, Loader=yaml.FullLoader)


@pytest.fixture
def category():
    """ Define a fixture to create an instance of the Category class """
    return Category(config)

class TestCategory:


    @pytest.mark.parametrize("exist_cat", [['حقوقی', 'بانکداری الکترونیک', 'ارزی-ریالی', 'تست', 'تسهیلات']])
    def test_create_dict_of_cat_text(self, category, exist_cat):
        """ Test the create_dict_of_cat_text method of Category """
        result = category.create_dict_of_cat_text()
        for key_cat in list(result.keys()):
            assert key_cat in exist_cat

    @pytest.mark.parametrize("exist_cat", [['حقوقی', 'بانکداری الکترونیک', 'ارزی-ریالی', 'تسهیلات']])
    def test_show_cat(self, category, exist_cat):
        """ Test the calculate_freq_of_keywords method of Category """
        result_list = category.show_cat()
        for dict_cat in result_list:
            for _ in dict_cat:
                assert dict_cat['category_name'] in exist_cat

    @pytest.mark.parametrize("exist_cat, user_question",
                             [(['حقوقی', 'بانکداری الکترونیک', 'ارزی-ریالی', 'تسهیلات'],
                             "آیا امکان افتتاح حساب غیر حضوری وجود دارد؟"),
                              (['حقوقی', 'بانکداری الکترونیک', 'ارزی-ریالی', 'تسهیلات'],
                            "برای دریافت رمز دوم همراه بانک چه اقداماتی باید انجام داد؟")])
    def test_find_matching_category(self, category, exist_cat, user_question):
        """ Test the sorted_main_keywords method """
        cat_result = category.find_matching_category(user_question)
        for cat_name in cat_result:
            assert cat_name in exist_cat

    @pytest.mark.parametrize("cat_name", ['حقوقی', 'بانکداری الکترونیک', 'ارزی-ریالی', 'تسهیلات'])
    def test_insert_categories_to_collection(self, category, cat_name):
        """ Test the find_matching_category method of Category """
        category.insert_categories_to_collection(cat_name)
        result_list = category.show_cat()
        list_cat = [dict_cat['category_name'] for dict_cat in result_list]
        assert cat_name in list_cat

    @pytest.mark.parametrize("cat_id", ["654f67f579e98aab34f05301", "654f67f579e98aab34f052fe"])
    def test_delete_category(self, category, cat_id):
        """
            Test the delete_category method of Category,
            assert that the category with id 1 and name 'test_category' is deleted
        """
        category.delete_category(cat_id)
        result_list = category.show_cat()
        list_cat = [dict_cat['category_name'] for dict_cat in result_list]
        assert len(list_cat) == 4

    def test_delete_all_categories(self, category):
        """
            Test the delete_all_categories method of Category,
            assert that the category collection is empty
        """
        category.delete_all_categories()

    @pytest.mark.parametrize("cat_id, cat_name", [("654f6e52eccc32aea6909bc2", "قانونی")])
    def test_update_categories_collection(self, category, cat_id, cat_name):
        """
            Test the update_categories_collection method of Category,
            assert that the category with id 1 is updated with the new name
        """
        category.update_categories_collection(cat_id, cat_name)
        result_list = category.show_cat()
        list_cat = [dict_cat['category_name'] for dict_cat in result_list]
        assert cat_name in list_cat

    @pytest.mark.parametrize("cat_name", ["حقوقی"])
    def test_get_question_by_category(self, category, cat_name):
        """
            Test the insert_categories_to_collection method of Category,
            assert that the categories are inserted into the category collection
        """
        test_res = category.get_question_by_category(cat_name)
        list_cat = [dict_cat['category_name'] for dict_cat in test_res]
        for cat in list_cat:
            assert cat == cat_name

    @pytest.mark.parametrize("exist_cat", [['حقوقی', 'بانکداری الکترونیک', 'ارزی-ریالی', 'تسهیلات']])
    def test_show_keywords(self, category, exist_cat):
        """ Test the calculate_freq_of_keywords method of Category """
        keyword_list = category.show_keywords()
        for dict_cat in keyword_list:
            assert dict_cat['category_name'] in exist_cat

    @pytest.mark.parametrize("exist_cat", [['حقوقی', 'بانکداری الکترونیک', 'ارزی-ریالی', 'تسهیلات']])
    def test_read_all_keywords(self, category, exist_cat):
        """ Test the calculate_freq_of_keywords method of Category """
        keyword_list = category.read_all_keywords()
        for dict_cat in keyword_list:
            assert dict_cat in exist_cat


if __name__ == "__main__":
    pytest.main()