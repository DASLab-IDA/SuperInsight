import pymysql
from dbutils.pooled_db import PooledDB


class DataSource:
    def __init__(self, host: str, user: str, password: str, database: str, num_thread=1):
        try:
            # self.db = pymysql.connect(
            #     host=host,
            #     user=user,
            #     password=password,
            #     database=database)
            self.db_poll = PooledDB(creator=pymysql,
                                    maxconnections=0,
                                    mincached=num_thread,
                                    maxcached=0,
                                    maxusage=None,
                                    blocking=True,
                                    host=host,
                                    user=user,
                                    password=password,
                                    database=database)
            # self.cursor = self.db.cursor(cursor=pymysql.cursors.DictCursor)
        except Exception as e:
            print("Exception : %s" % e)

    def execute_sql(self, sql):
        try:
            db = self.db_poll.connection()
            cursor = db.cursor(cursor=pymysql.cursors.DictCursor)
            cursor.execute(sql)
            v_result = cursor.fetchall()
            cursor.close()
            db.close()
            return v_result
        except Exception as e:
            print(e)

    def close(self):
        # self.cursor.close()
        self.db_poll.close()
