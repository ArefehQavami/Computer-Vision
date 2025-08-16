from basics.database import MongoDb
from logger.elasticlogger import ElasticsearchLogger

class Report:
    def __init__(self, config):

        obj_db = MongoDb(config, use_config=True)
        obj_db.connect(config['mongodb_db'])
        self.collection = obj_db.db[config['mongodb_coll']]
        self.logger = ElasticsearchLogger(config, config['elastic_index_name'])

    def qa_count(self):
        """
        calculate number of records of a collection
        """
        record_count = self.collection.count_documents({})
        return record_count

    def qa_count_category(self):
        """
        calculate number of records based on category
        """
        pipeline = [
            {"$group": {"_id": "$category_id", "count": {"$sum": 1}, "category_name": {"$first": "$category_name"}}}
        ]
        result = self.collection.aggregate(pipeline)
        catgory_count = []
        for category in result:
            category_id = category["_id"]
            count = category["count"]
            category_name = category["category_name"]
            entry = {
                "category_id": category_id,
                "category_name": category_name,
                "count": count
            }
            catgory_count.append(entry)

        return catgory_count


