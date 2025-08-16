import json
from flask import Flask
from report import Report
import yaml

with open(r'E:\qa\basics\config.yml', 'r') as file:
    config = yaml.load(file, Loader=yaml.FullLoader)
rp = Report(config=config)
app = Flask(__name__)

@app.route('/count', methods=['GET'])
def qa_count():
    try:
        rp_count = rp.qa_count()
        rp.logger.log("info", 'The number of questions has been successfully counted')
        return json.dumps({"message": "تعداد کل سوالات", "data": rp_count})
    except Exception as e:
        rp.logger.log("error", 'Error counting the number of questions')
        return ' محاسبه تعداد کل سوالات با خطا مواجه شد : {}'.format(str(e)), 500


@app.route('/countCategory', methods=['GET'])
def qa_count_category():
    try:
        rp_cat_count = rp.qa_count_category()
        rp.logger.log("info", 'The number of questions based on category has been successfully counted')
        return json.dumps({"message": "تعداد کل سوالات به تفکیک دسته", "data": rp_cat_count})
    except Exception as e:
        rp.logger.log("error", 'Error counting the number of questions based on category')
        return ' محاسبه تعداد کل سوالات به تفکیک دسته با خطا مواجه شد : {}'.format(str(e)), 500


app.run()
