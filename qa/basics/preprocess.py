import re
from hazm import *
from abc import ABC


class Preprocess(ABC):
    """
    class Preprocess has some classes for preprocessing of Persian texts
    the methods include tokenizing, stop word removal, stemming, lemmatizing, POS Tagging
    """
    def __init__(self):

        self.normalizer = Normalizer()
        self.letter_dict = {u"ۀ": u"ه", u"ة": u"ت", u"ؤ": u"و", u"إ": u"ا", u"ئ": u"ی", u"ﷲ": u"", u"ٱ": u"ا", u"ٲ": u"ا", u"أ": u"ا"}
        self.letter_pattern = re.compile(r"(" + "|".join(self.letter_dict.keys()) + r")")
        self.space_pattern = re.compile(r"[\xad\u200C\ufeff\u200e\u200d\u200b\x7f\u202a\u2003\xa0\u206e\u200c\x9d\u200C\u200c\u2005\u2009\u200a\u202f\t\u200c]+")
        self.deleted_pattern = re.compile(r"([^\w\s]|[\|\[]]|\"|'ٍ|¬|[a-zA-Z]|[؛“،,”‘۔’’‘–]|[|\.÷+\:\-\?»\=\{}\*«_…\؟!/ـ]|[\\u\\x]|[\(\)]|[۰'ٓ۫'ٔ]|[ٓٔ]|[ًٌٍْﹼ،َُِّ«ٰ»ٖء]|\[]|\[\])")
        self.deleted_numbers = re.compile(r"([0-9]|\d|)|[۲۹۱۷۸۵۶۴۴۳]")
        self.special_chars = re.compile('|'.join(
            ['²', '³', 'µ', 'À', 'Á', 'Â', 'Ç', 'È', 'É', 'Ê', 'Í', 'Ö', 'Û', 'Ü', 'ß', 'à', 'á', 'â', 'ã', 'ä',
             'ç', 'è', 'é', 'ê', 'í', 'î', 'ï', 'ñ', 'ó', 'ô', 'õ', 'ö', 'ø', 'ú', 'ü', 'ý', 'ā', 'č', 'ē', 'ę',
             'ğ', 'ī', 'ı', 'ř', 'ş', 'š', 'ū', 'ž', 'Ɛ', 'Α', 'Κ',
             'ά', 'ή', 'α', 'β', 'ε', 'ζ', 'η', 'ι', 'κ', 'ν', 'ο', 'π', 'ρ', 'ς', 'τ', 'υ', 'ό', 'Տ', 'ա', 'ե',
             'թ', 'ի', 'ն', 'վ', 'ւ', 'ք', 'ḥ', 'ṭ', 'ῥ']))
        self.new_line_pattern = re.compile(r'\n+')
        self.add_additional_stopwords = ['می‌نماید','چیست','جهت','آن‌ها','چنانچه','خودشان' ,'بموجب','مبنی','گردیده','توسط','نمایند','آنرا','صرفا','بله','خیر','هیچگونه','نماید','می‌دارد','نموده',
                                    'نزد','منجر','نشده', 'نموده_باشد','طریق','نگردد','عنوان','مقابل','میتوان','نمود','صورتیکه','علیرغم','خصوصا','عملا','مواردیکه','آید','می‌گردد','نباشد',
                                    'می‌نمایند','می‌آورند','درصورتیکه','کدامند','ننماید','می‌باشند','گردیده اند','هنگامیکه','بنماید','نماییم','نمی‌توانند','نمی‌باشد','کردیم','داده_می‌شود',
                                    'عهده','نشد','مجدداً','برود','رفتن','کرده‌ام','بکند','نشده است','نشود']
        self.remove_additional_stopwords = ['اول', 'دوم','سوم', 'چهارم','پنجم','ششم','هفتم','هشتم','نهم','دهم','یک','دو','سه','چهار','پنج','شش','هفت','هشت','نه','ده',
                                'اولین','دومین','سومین','چهارمین','پنجمین','ششمین','هفتمین','هشتمین','نهمین','دهمین' ,'ویژه', 'حداقل','شخصی','جاری','نظر','چند','جدید'
            ,'اثر','دسته','بروز','تغییر','رسید','حل','ترتیب','مدت']


    def common_clean(self, string):
        """
        clean the text; normalize text and remove punctuations and specia chars and ...
        """
        string = self.normalizer.normalize(string)
        string = self.space_pattern.sub(" ", string)
        string = self.deleted_pattern.sub("", string)
        string = self.special_chars.sub("", string)
        string = self.letter_pattern.sub(
            lambda x: self.letter_dict[x.group()], string)
        string = self.new_line_pattern.sub(r'\n', string)
        string = self.seperate_mi(string)
        string = self.correct_spacing(string)
        return string

    def clean_cat(self, string: str) -> str:
        """
        clean the text with removing numbers too
        """
        string = self.common_clean(string)
        string = self.deleted_numbers.sub("", string)
        return string

    def preprocess_text(self, string):
        """
        preprocess on text; cleaning , tokenizing, stopword removal
        """
        string = self.common_clean(string)
        string = self.tokenizer(string)
        string = self.token_spacing(string)
        string = self.stopword_removal(string)
        string = ' '.join(string)
        return string

    def correct_spacing(self, text):
        """
        Fix spaces in prefixes and suffixes
        """
        correct_space = self.normalizer.correct_spacing(text)
        return correct_space

    def remove_diacritics(self, text):
        """
        Remove diacritics from text
        """
        remove_diacritics = self.normalizer.remove_diacritics(text)
        return remove_diacritics

    def remove_specials_chars(self, text):
        """
        remove special meaningless words from text
        """
        remove_specials_chars = self.normalizer.remove_specials_chars(text)
        return remove_specials_chars

    def seperate_mi(self, text):
        """
        separate the prefix mi (می)
        """
        seperate_mi = self.normalizer.seperate_mi(text)
        return seperate_mi

    def persian_style(self, text):
        """
        Replace some letters and symbols with Persian letters and symbols.
        """
        persian_style = self.normalizer.persian_style(text)
        return persian_style

    def tokenizer(self, text):
        """
        Tokenize text; Split the text into individual words or tokens
        """
        tokens = word_tokenize(text)
        return tokens

    def token_spacing(self, tokenized_text):
        """
        Normalize tokens of text; Check space of each token
        """
        spaced_tokens = self.normalizer.token_spacing(tokenized_text)
        return spaced_tokens

    def stopword_removal(self, tokenized_text):
        """
        Remove stop word of the tokenized text; Eliminate commonly occurring words
        """
        stopwords = stopwords_list()
        stopwords_plus = [item for item in stopwords if item not in self.remove_additional_stopwords]
        stopwords_plus.extend(self.add_additional_stopwords)
        words_without_stopwords = [word for word in tokenized_text if word not in stopwords_plus]
        return words_without_stopwords

    def lemmatizer(self, tokenized_text):
        """
        Lemmatize each word in the sentence; Normalize words to their base or root form
        """
        lemmatizer = Lemmatizer()
        lemmatized_words = [lemmatizer.lemmatize(word) for word in tokenized_text]
        return lemmatized_words
