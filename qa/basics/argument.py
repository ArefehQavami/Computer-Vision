import argparse


def initial_argument():

    """
        this function defined for initializes all of imported argument.
    """
    parser = argparse.ArgumentParser(description='A question-answering chatbot that provide answers to user queries')

    parser.add_argument('--dataset_path', type=str, default=r"C:\Users\m.gheysari\Desktop\QA-All.xlsx",
                        help='path of csv file dataset')

    parser.add_argument('--mongodb_user', type=str, default="qa",
                        help='user name of mongodb database')

    parser.add_argument('--mongodb_pass', type=str, default="Aa1234",
                        help='mongodb database''s password')

    parser.add_argument('--mongodb_ip', type=str, default="10.187.105.206",
                        help='ip address of mongodb database')

    parser.add_argument('--mongo_port', type=str, default="27017",
                        help='port of mongodb database')

    parser.add_argument('--mongodb_db', type=str, default="qa",
                        help='name of mongodb database')

    parser.add_argument('--mongodb_coll', type=str, default="qa",
                        help='name of mongodb''s collection')

    return parser