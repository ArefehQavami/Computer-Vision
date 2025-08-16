# from settings.setting import Blueprint, BadHost
# question = Blueprint('question', __name__, url_prefix='/question/')
from flask import Flask, request
from question.qa import QA
import yaml
import json

app = Flask(__name__)
with open(r'E:\qa\basics\config.yml', 'r') as file:
    config = yaml.load(file, Loader=yaml.FullLoader)
qa = QA(config=config)

@app.route('/addQuestion',  methods=['POST'])
def add_question():
    try:
        data = request.json
        question = data.get('question')
        answer = data.get('answer')
        category_id = data.get('category_id')
        category_name = data.get('category_name')
        qa.add_qa(question=question, answer=answer, category_id=category_id, category_name=category_name)
        qa.logger.log("info", 'Question added successfully.')
        return {'message': 'سوال با موفقیت اضافه شد.'}, 200
    except Exception as e:
        qa.logger.log("error", 'Error adding question')
        return 'اضافه نمودن سوال با خطا مواجه شد : {}'.format(str(e)), 500

@app.route('/editQuestion/<id>', methods=['PUT'])
def edit_question(id):
    try:
        data = request.json
        question = data.get('question')
        answer = data.get('answer')
        category_id = data.get('category_id')
        category_name = data.get('category_name')
        qa.edit_qa(id=id, question=question, answer=answer, category_id=category_id, category_name=category_name)
        qa.logger.log("info", 'Question updated successfully.')
        return {'message': 'سوال با موفقیت بروزرسانی شد.'}, 200
    except Exception as e:
        qa.logger.log("error", 'Failed to update question with id')
        return f" بروزرسانی سوال با این آیدی با مشکل مواجه شد {id}: {str(e)}", 500


@app.route('/deleteQuestion/<id>', methods=['DELETE'])
def delete_question(id):
    try:
        qa.delete_qa(id=id)
        qa.logger.log("info", 'Question deleted successfully.')
        return {'message': 'سوال با موفقیت حذف شد.'}, 200
    except Exception as e:
        qa.logger.log("error", 'Error deleting question')
        return 'حذف سوال با خطا مواجه شد: {}'.format(str(e)), 500


@app.route('/deleteQuestions', methods=['DELETE'])
def delete_questions():
    try:
        qa.delete_all_qa()
        qa.logger.log("info", 'All Questions deleted successfully.')
        return {'message': 'تمام سوالات با موفقیت حذف شدند.'}, 200
    except Exception as e:
        qa.logger.log("error", 'Error deleting questions')
        return 'حذف کل سوالات با خطا مواجه شد : {}'.format(str(e)), 500


@app.route('/getQuestion/<id>', methods=['GET'])
def get_question(id):
    try:
        question_answer = qa.get_qa_by_id(id=id)
        question_answer['_id'] = str(question_answer['_id'])
        qa.logger.log("info", 'Found Question successfully.')
        return json.dumps({"message": "سوال با موفقیت دریافت شد.", "data": question_answer})
    except Exception as e:
        qa.logger.log("error", 'Error getting question')
        return 'دریافت سوال با این آیدی با خطا مواجه شد: {}'.format(str(e)), 500


@app.route('/getQuestions', methods=['GET'])
def get_all_questions():
    try:
        question_answer = qa.get_all_qa()
        qa.logger.log("info", 'Found Question successfully')
        return json.dumps({"message": "تمام سوالات با موفقیت دریافت شدند.", "data": (question_answer)}, default=str)
    except Exception as e:
        qa.logger.log("error", 'Error getting questions')
        return 'دریافت همه سوالات با خطا مواجه شد: {}'.format(str(e)), 500


app.run()
