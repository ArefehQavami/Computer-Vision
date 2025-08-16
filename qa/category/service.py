from flask import Flask, request, jsonify
from flask.views import MethodView
import yaml
from basics.database import MongoDb
from category.category import Category
from logger.elastic_search import ElasticsearchLogger

app = Flask(__name__)

class CategoryAPI(MethodView):
    """
        Retrieve all categories,
        Add categories to collection,
        Update category
    """

    def __init__(self, config):
        self.category = Category(config)

    def get(self):
        try:
            categories_serializable = [{'_id': str(cat['_id']), 'name': cat['category_name']} for cat in
                                       self.category.show_cat()]
            self.category.logger_error.log('info', "Category information received successfully.")
            return jsonify(categories_serializable), 200
        except Exception as e:
            return jsonify({'خطا': 'دریافت اطلاعات دسته بندی با خطا مواجه شد.', 'error': str(e)}), 400

    def post(self):

        try:
            cat_name = request.json.get('cat_name')
            self.category.insert_categories_to_collection(cat_name)
            self.category.logger_error.log('info', "The category information has been successfully inserted into the database.")
            return jsonify({'پیام': "اطلاعات دسته بندی با موفقیت در پایگاه داده درج شد."}), 200

        except Exception as e:
            return jsonify({'خطا': 'درج اطلاعات دسته بندی با خطا مواجه شد.', 'error': str(e)}), 400

    def put(self):
        try:
            cat_id = request.json.get('cat_id')
            cat_name = request.json.get('cat_name')
            self.category.update_categories_collection(cat_id, cat_name)
            self.category.logger_error.log('info', "Category information has been successfully updated in the database.")
            return jsonify({'پیام': 'اطلاعات دسته بندی با موفقیت در پایگاه داده به روز رسانی شد.'}), 200
        except Exception as e:
            return jsonify({'خطا': 'به روز رسانی اطلاعات دسته بندی با خطا مواجه شد', 'error': str(e)}), 400

    def delete(self):
        try:
            cat_id = request.json.get('cat_id')
            self.category.delete_category(cat_id)
            self.category.logger_error.log('info', "Category information was successfully deleted.")
            return jsonify({'پیام': "اطلاعات دسته بندی با موفقیت حذف شد."}), 200
        except Exception as e:
            return jsonify({'خطا': 'حذف اطلاعت دسته بندی با خطا مواجه شد', 'error': str(e)}), 400

class CategoryQuestionAPI(MethodView):

    def __init__(self, config):
        self.category = Category(config)

    def get(self):
        try:
            cat_name = request.json.get('cat_name')
            categories_serializable = [{'_id': str(que['_id']), 'question': que['question'], 'answer': que['answer'],
                                        'category_id': que['category_id'], 'category_name': que['category_name']}\
                                       for que in self.category.get_question_by_category(cat_name)]
            self.category.logger_error.log('info', "Information received successfully.")
            return jsonify(categories_serializable), 200
        except Exception as e:
            return jsonify({'خطا': 'دریافت اطلاعات کلاس دسته بندی با خطا مواجه شد', 'error': str(e)}), 400

class MatchingCategoryAPI(MethodView):

    def __init__(self, config):
        self.category = Category(config)

    def get(self):
        try:
            keywords_serializable = [{keyword['category_name']: keyword['word_frequency']} for keyword in
                                     self.category.show_keywords()]
            self.category.logger_error.log('info', "Keyword information received successfully.")
            return jsonify(keywords_serializable), 200
        except Exception as e:
            return jsonify({'خطا': 'دریافت اطلاعات کلمات کلیدی با خطا مواجه شد', 'error': str(e)}), 400

    def delete(self):
        try:
            category_name = request.json.get('category_name')
            self.category.delete_category(category_name)
            self.category.logger_error.log('info', "Keyword information successfully deleted.")
            return jsonify({'پیام': "اطلاعات کلمات کلیدی با موفقیت حذف شد."})
        except Exception as e:
            return jsonify({'خطا': 'حذف اطلاعات کلمات کلیدی با خطا مواجه شد', 'error': str(e)}), 400


if __name__ == '__main__':
    with open(r'C:\Users\m.gheysari\PycharmProjects\pythonProject24\config.yml', 'r') as file:
        config = yaml.load(file, Loader=yaml.FullLoader)

    app.add_url_rule('/categories', view_func=CategoryAPI.as_view('categories', config=config))
    app.add_url_rule('/matching_category', view_func=MatchingCategoryAPI.as_view('matching_category', config=config))
    app.add_url_rule('/category_question', view_func=CategoryQuestionAPI.as_view('category_question', config=config))
    app.run(debug=True)
