import os
import csv
import json

class Initialize_Data():
    labels = []
    posts = []

    def initialize_twitter_posts(self, csv_file, data_dir):
        with open(csv_file, encoding='utf-8') as csvfile:
            metareader = csv.reader(csvfile, delimiter=',', quotechar='"')
            for line in metareader:
                label, post_id = line
                post_fn = os.path.join(data_dir, '%s.json' % post_id)
                try:
                    post = json.load(open(post_fn, "r"))
                except IOError:
                    print("Post '%s' not found. Skip."%post_fn)
                    continue

                if 'text' in post:
                    self.labels.append(label)
                    self.posts.append(post['text'])

    def initialize_facebook_posts(self, csv_file, data_dir):
        with open(csv_file, encoding='utf-8') as csvfile:
            metareader = csv.reader(csvfile, delimiter=',', quotechar='"')
            for line in metareader:
                label, post_id = line
                post_fn = os.path.join(data_dir, '%s.json' % post_id)
                try:
                    post = json.load(open(post_fn, "r"))
                except IOError:
                    print("Post '%s' not found. Skip."%post_fn)
                    continue

                if 'message' in post:
                    self.labels.append(label)
                    self.posts.append(post['message'])
