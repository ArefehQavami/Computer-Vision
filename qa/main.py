
import pandas as pd
import time

import pymongo
from report.report import Report
from question.qa import QA
import yaml
from basics.preprocess import Preprocess
# from category.category import Category
# from test_question_calss import TestQA
from basics.database import MongoDb
from pymongo import MongoClient
def main():

    with open(r'E:\qa\basics\config.yml', 'r') as file:
        config = yaml.load(file, Loader=yaml.FullLoader)



    rp = Report(config)
    print(rp.qa_count())
    print(rp.qa_count_category())
    # client = MongoClient("mongodb://10.187.105.206:27017", username ='qa'   , password='Aa1234')
    # db = client["qa"]
    # collection = db["qa"]
    # print(collection)
    # for t in collection.find():
    #     print(t)
    #
    #
    # for t in collection.find():
    #     print(t)

    # client2 = MongoClient("mongodb://10.187.160.101:27017", username ='ai'   , password='ad12bfWqo')
    # db2 = client2["qa"]
    # collection2 = db2["qa"]
    # print(collection2)
    # count = 0
    # for t in collection2.find():
    #     print(t)
    #     count = count + 1
    # print(count)

    # collection2.update_many({}, {"$unset": {"id": ""}})

    # records = list(collection2.find())

    # Insert the records into the destination collection
    # if records:
    #     collection.insert_many(records)
    #     print("Records copied successfully!")
    # else:
    #     print("No records found in the source collection.")
    # """ test QA class """
    qa = QA(config)
    # qa.add_all_qa("E:/qa/data/data.csv")

    all = qa.get_all_qa()
    # prep = Preprocess()
    # count = 0
    for item in all:
    #     text = item['question']
        print(item)
    #     count = count + 1
    #     print(count)
    #     start = time.time()
    #     result = prep.common_clean(item['question'])
    #     result1 = prep.preprocess_text(item['question'])
    #     end = time.time()
    #     print("time", end - start)
    #     print("common clean", result)
    #     print("preprocess text", result1)
    #     space = prep.correct_spacing(text)
    #     print(space)
    #     correct = prep.remove_diacritics(text)
    #     print(correct)
    #     chars = prep.remove_specials_chars(text)
    #     print(chars)
    #     mi = prep.seperate_mi(text)
    #     print("miiiiii", mi)
    #     style = prep.persian_style(text)
    #     print(style)
    #     token = prep.tokenizer(text)
    #     print("token", token)
    #     token_space = prep.token_spacing(text)
    #     print("token_space", token_space)
    #     stopword = prep.stopword_removal(token)
    #     print("stopword", stopword)
    #     lem = prep.lemmatizer(text)
    #     print("lem", lem)


    id = "6523cacb8dc1f603e1e4d2ea"
    # qa.add_qa("2","2","2","2")
    # qa.delete_qa(id)
    # qa.edit_qa(id,None, None, None, None)
    # qa.delete_all_qa()
    # print(qa.get_qa_by_id(id))
    # print(qa.search_qa("حقوق"))
    # df = pd.DataFrame(pd.read_excel("E:/qa/data/new.xlsx"))
    # qa.add_all_qa(df)
    # all = qa.get_all_qa()
    # for item in all:
    #     print(item)


    # prep = Preprocess()
    # # docs = qa.collection.find()
    # df = qa.excel_to_csv("E:/qa/data/QA-All - Copy.xlsx")
    # count = 0
    # new = []
    # # for doc in docs:
    # for i in range(0, 20):
    #     print(count)
    #     print(df['question'].iloc[count])
    #     # clean = prep.clean(df['question'].iloc[count])
    #     norm = prep.normalizer(df['question'].iloc[count])
    #     # print(norm)
    #     space = prep.correct_spacing(norm)
    #     # print(space)
    #     correct = prep.remove_diacritics(space)
    #     # print(correct)
    #     chars = prep.remove_specials_chars(correct)
    #     # print(chars)
    #     style = prep.persian_style(chars)
    #     # print(style)
    #     num = prep.persian_number(style)
    #     # print(num)
    #     clean = prep.remove_punctuations(num)
    #     print(clean)
    #     df['question-clean'].iloc[count] = clean
    #     # print(num)
    #     # new.append(num)
    #     # qa.edit_qa(doc['_id'], question= None, answer=num, category=None)
    #     count = count + 1
    #
    # print(df['question'].iloc[15])
    # print(df['question'].iloc[16])
    # print(df['question'].iloc[17])
    # df.to_excel("new.xlsx")


    # qa = QA(config)
    # qa.get_all_qa()
    # # print(new)
    # new_df = pd.DataFrame(new)
    # print(new_df)
    # new_df.to_csv("E:/qa/data/new.csv")

    text = "#$%&'()*+,-./:;<=>?@[\]^_`{|}~! ،٪×÷»«۰'ٓ ؛“،,”‘۔’’‘–.÷+\:\-\?»\=\{}\*«_…\؟!/ـ]|[\\u\\x  (ﹼ،َُِّ«ٰ»ٖء) - ۀ أ ة ي ؤ إ ٹ ڈ ئ ﻨ ﺠ ﻣ ﷲ  ﻳ   ٻ ڵ ٱ ﭘ ﻪ ﻳ ٻ ں ٶ ہ ﻩ ك ٲ ﺆ ﺪ ترتیب حل، ن   شده است می شده است\u200Cکمبئبکسب ۀ ة ي ؤ إ ٹ ڈ ئ ﻨ ﺠ ﻣ ﷲ ﻳ ٻ ٱ ڵ ﭘ ﻪ ﻳ ٻ ں ٶ ٲ ہ ﻩ ﻩ ك ﺆ أ ﺪ  حَذفِ اِعراب آمدند پیامبر اکرم ﷺ   1 2 3 4   "
    # text = "۰'ٓ ؛“،,”‘۔’’‘–.÷+\:\-\?»\=\{}\*«_…\؟!/ـ]|[\\u\\x  (ﹼ،َُِّ«ٰ»ٖء) - ۀ أ ة ي ؤ إ ٹ ڈ ئ ﻨ ﺠ ﻣ ﷲ  ﻳ   ٻ ڵ ٱ ﭘ ﻪ ﻳ ٻ ں ٶ ہ ﻩ ك ٲ ﺆ ﺪ ترتیب حل، ن   شده است می شده است\u200Cکمبئبکسب   "
    # text = " Arefeh Qavami ² ³ µ  À  Á Â Ç È É Ê Í Ö Û Ü ß à á â ã ä ç è é ê í î ï ñ ó ô õ ö ø ú ü ý ā č ē ę ğ ī ı ř ş š ū ž Ɛ Α Κ ά ή α β ε ζ η ι κ ν  ۰۱۲۳۴۵۶۷۸۹٠"
    # '۰', '۱', '۲', '۳', '۴', '۵', '۶', '۷', '۸', '۹', '٠',
    # '١', '٢', '٣', '٤', '٥', '٦', '٧', '٨', '٩', '۰', '۱', '۲', '۳', '۴', '۵', '۶', '۷', '۸', '۹'
           # "'ο', 'π', 'ρ', 'ς', 'τ', 'υ', 'ό', 'Տ', 'ա', 'ե',
           #   'թ', 'ի', 'ն', 'վ', 'ւ', 'ք', 'ḥ', 'ṭ', 'ῥ', '۰', '۱', '۲', '۳', '۴', '۵', '۶', '۷', '۸', '۹', '٠',

    text = " اَلسَّلامُ عَلَیْکَ یا اَباعَبْدِاللَّهِ اَلسَّلامُ عَلَیْکَ یَابْنَ رَسُولِ اللَّهِ اَلسَّلامُ عَلَیْکَ یا خِیَرَةَ اللَّهِ و َابْنَ خِیَرَتِهِ اَلسَّلامُ عَلَیْکَ یَابْنَ اَمیرِالْمُؤْمِنینَ و َابْنَ سَیِّدِ الْوَصِیّینَ   اَلسَّلامُ عَلَیْکَ یَابْنَ فاطِمَةَ سَیِّدَةِ نِساَّءِ الْعالَمینَ اَلسَّلامُ عَلَیْکَ یا ثارَ اللَّهِ وَ ابْنَ ثارِهِ وَ الْوِتْرَ الْمَوْتُور   10.450   12354555"
    text = "است و بود و شد و می دارد "


    # prep = Preprocess()
    # result = prep.common_clean(text)
    # print("common clean", result)
    # result1 = prep.remove_punctuations(text)
    # print(result1)
    # result2 = prep.clean_cat(text)
    # print("clean qa", result2)
    # t = prep.clean_qa(text)
    # print(" arefeh clean_qa", t)
    # tt = prep.preprocess_text(text)
    # print("preprocess_text", tt)
    # # print("clean")
    # result = prep.clean(text)
    # print("maryam clean", result)
    # result1= prep.normalizer(text)
    # print("normalize", result1)
    # result1= prep.correct_spacing(text)
    # print(result1)
    # t = prep.preprocess_text(text)
    # print(t)
    # token = prep.tokenizer(text)
    # print(token)
    # t = prep.stopword_removal(token)
    # print(t)

    # changed_text = prep.correct_spacing(text)
    # print(changed_text)
    # remove_dialte = prep.remove_diacritics(changed_text)
    # print(remove_dialte)
    # remove_specials_chars = prep.remove_specials_chars(remove_dialte)
    # print(remove_specials_chars)
    # separate_me = prep.seperate_mi(remove_specials_chars)
    # print(separate_me)
    # persian_style = prep.persian_style(separate_me)
    # print(persian_style)
    # persian_num = prep.persian_number(persian_style)
    # print(persian_num)
    # token = prep.tokenizer(persian_num)
    # print(token)

    # import requests

    # url = 'http://localhost:5000/addQuestion'
    # data = {
    #     'question': 'What is the capital of France?',
    #     'answer': 'Paris',
    #     'category': 'Geography'
    # }
    # response = requests.post(url, data=data)
    #
    # print(response.json())



    # test = TestQA()

if __name__ == '__main__':
    main()