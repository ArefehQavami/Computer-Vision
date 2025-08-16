"""
      The implemented class includes methods for working with the Mongo database,
      allowing data to be read from or written to it.
"""
import pymongo
import pandas as pd
from pymongo import MongoClient

class MongoDb:
    """
         This class represents a MongoDB custom class for handling user data.
         It provides functions for CRUD operations on the "users" collection.
    """
    def __init__(self, config, use_config=False):

        self.config = config
        self.use_config = use_config

        self.mongo_url = 'mongodb://%s:%s@%s:%s/' % (self.config['mongodb_user'],
                                                     self.config['mongodb_pass'],
                                                     self.config['mongodb_ip'],
                                                     self.config['mongo_port'])

        self.client = None
        self.db = None


    def connect(self, db_name):

        """
            Connects to the MongoDB database.
            :param db_name: The name of database  to be connected.

        """

        self.client = MongoClient(self.mongo_url)

        self.db = self.client[db_name]

    def save_dataframe(self, dataframe: pd.DataFrame):

        """
            Saves a DataFrame to a collection in the MongoDB database.
            :param dataframe: The DataFrame to be saved.
        """

        self.connect(self.config['mongodb_db'])

        if not isinstance(dataframe, pd.DataFrame):
            raise ValueError("The input must be a DataFrame.")

        records = dataframe.to_dict(orient='records')

        if not self.use_config:
            self.db[self.args.mongodb_coll].insert_many(records)
        else:
            self.db[self.config['mongodb_coll']].insert_many(records)

        print("Data saved successfully.")
