import configparser
import logging
import random
import sqlite3 as sql
from datetime import datetime

import pandas as pd
import praw
from tqdm import tqdm

logger = logging.getLogger()
logger.setLevel(logging.INFO)
c_handler = logging.StreamHandler()
f_handler = logging.FileHandler('crawl.log')
fformat = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
c_handler.setFormatter(fformat)
f_handler.setFormatter(fformat)
logger.addHandler(c_handler)
logger.addHandler(f_handler)
logging.basicConfig(level=logging.NOTSET)


class RedditCrawler:
    def __init__(self, start_point, database, config="reddit_api.ini"):
        conf = configparser.ConfigParser()
        conf.read(config)

        self.client = praw.Reddit(user_agent=conf["api"]["user_agent"],
                                  client_id=conf["api"]["client_id"],
                                  client_secret=conf["api"]["client_secret"])

        self.db = sql.connect(database)
        self.init_db()

        self.obligatory_users = [start_point] if isinstance(start_point, str) else start_point
        self.optionally_users = set()
        self.black_list = ["AutoModerator", "[deleted]"]
        self.counter = 0

    def _exec(self, cmd, attr=()):
        cursor = self.db.cursor()
        cursor.execute(cmd, attr)
        data = cursor.fetchall()
        cursor.close()
        return data

    def init_db(self):
        tbl_cmt = """CREATE TABLE comment(ID INTEGER PRIMARY KEY AUTOINCREMENT, 
        user text, 
        subreddit text,
        content text,
        num_upvotes integer,
        num_downvotes integer,
        num_comments integer, 
        created_time_utc integer,
        link_title text,
        link_id text );"""

        tbl_tbl = "SELECT name FROM sqlite_master WHERE type ='table' AND name NOT LIKE 'sqlite_%';"
        existing_tbl = [item for t in self._exec(tbl_tbl) for item in t]

        if "comment" not in existing_tbl:
            self._exec(tbl_cmt)

    def add_user(self, data):
        stmt = """ INSERT INTO comment (user, subreddit, content, num_upvotes, num_downvotes, num_comments, 
        created_time_utc, link_title, link_id)  VALUES (?,?,?,?,?,?,?,?,?)"""
        self.db.executemany(stmt, data)
        self.db.commit()
        self.counter += len(data)

    def reset_db(self):
        self._exec("drop table comment")
        self.db.commit()

    @staticmethod
    def wrapper(gen):
        while True:
            try:
                yield True, next(gen)
            except StopIteration:
                break
            except Exception as e:
                yield False, e

    def process_user(self, user_name, number_comments=10, new_users=1):
        data = list()
        possible_next_users = set()

        user = self.client.redditor(user_name)

        comments = user.comments.new(limit=number_comments)  # if none its 1024

        for success, comment in self.wrapper(comments):
            if success:
                if comment.subreddit_type != 'public':
                    logger.info(
                        f"Type of subreddit '{comment.subreddit.display_name}' is '{comment.subreddit_type}'. Skipping.")
                    continue

                try:
                    user_data = (user.name, comment.subreddit.display_name, comment.body, comment.ups, comment.downs,
                                 comment.num_comments, comment.created_utc, comment.link_title, comment.link_id)
                except AttributeError:
                    continue

                data.append(user_data)

                if not comment.link_author == user.name and comment.link_author not in self.black_list:
                    possible_next_users.add(comment.link_author)
            else:
                logger.warning(f"Comment of user '{user.name}' throws error: {comment} of type {type(comment)}")
                return False

        self.add_user(list(data))

        if len(possible_next_users) >= new_users:
            self.optionally_users.update(random.sample(possible_next_users, k=new_users))

        return True

    def start(self, limit=None, status=10000):
        logger.info(f"Starting at {datetime.now()}")
        l = 0
        while True:
            if limit is not None:
                if self.counter >= limit:
                    print("Limit reached!")
                    break
            if self.obligatory_users:
                user = random.choice(self.obligatory_users)
                self.obligatory_users.remove(user)
            elif self.optionally_users:
                user = random.sample(self.optionally_users, k=1)[0]
                self.optionally_users.remove(user)
            else:
                logger.error("No more Users found!")
                break

            self.process_user(user, number_comments=None, new_users=4)

            progress = self.counter // status
            if progress > l:
                l = progress
                logger.info(f"{self.counter} comments crawled.")
        logger.info(f"Crawling ended at {datetime.now()}. Found {self.counter} comments.")

    def sql_to_csv(self, table):
        df = pd.read_sql_query(f"SELECT * FROM {table}", self.db)
        df.to_csv("reddit_user_data_large.csv")


def subreddit_information():
    errors = 0
    failed_subreddits = []
    conf = configparser.ConfigParser()
    conf.read("data/reddit_api.ini")
    client = praw.Reddit(user_agent=conf["api"]["user_agent"],
                         client_id=conf["api"]["client_id"],
                         client_secret=conf["api"]["client_secret"])

    row_list = []

    db = sql.connect("data/RedditComments/reddit_user_data_with_content_new.db")
    iterator = db.execute("SELECT distinct subreddit from main.comment")
    size = db.execute("SELECT count(distinct subreddit) from main.comment").fetchone()[0]

    with tqdm(total=size) as progress_bar:
        for name, in iterator:
            try:
                subreddit = client.subreddit(name)
                row = dict(subreddit=name,
                           num_subscribers=subreddit.subscribers,
                           over18=subreddit.over18,
                           public_description=subreddit.public_description,
                           description_html=subreddit.description_html,
                           description_md=subreddit.description)
                row_list.append(row)
            except Exception as e:
                errors += 1
                failed_subreddits.append(name)
                print(e)
            progress_bar.update(1)

    df = pd.DataFrame(row_list,
                      columns=["subreddit", "num_subscribers", "over18", "public_description", "description_html",
                               "description_md"])

    print(f"Parsed {size} subreddit with {errors} error.")
    pd.DataFrame(dict(failed_subreddits=failed_subreddits)).to_csv("data/RedditComments/failed_subreddits.csv",
                                                                   index=False)
    df.to_csv("data/RedditComments/subreddit_info.csv", index=False)
    df.to_sql("subreddits", db, index=True, index_label="id", if_exists="replace")


if __name__ == '__main__':
    r = RedditCrawler(start_point=["blokesa1", "oystertoe", "FencerDoesStuff"],  # AksReddit, wallstreetbets, Python
                      database="reddit_user_data_with_content.db")

    r.start(limit=20e6, status=10000)
