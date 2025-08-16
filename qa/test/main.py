import yaml
import argparse
import pandas as pd
from bot.basics.argument import initial_argument
from bot.basics.database import MongoDb


if __name__ == '__main__':

    # call argument parser
    parser = initial_argument()
    args = parser.parse_args()

    # Load the configuration file
    with open(r'C:\Users\m.gheysari\PycharmProjects\chatbot\bot\basics\config.yml', 'r') as file:
        config = yaml.load(file, Loader=yaml.FullLoader)


    # Create an instance of the MongoDBDatabase class
    db = MongoDb(args, config, use_config=True)

    # Connect to the MongoDB database
    db.connect()

    # load csv file
    if not db.use_config:
        qa_df = pd.read_excel(args.dataset_path)
    else:
        qa_df = pd.read_excel(config['dataset_path'])


    # Save the DataFrame to the collection
    # db.save_dataframe(qa_df)
