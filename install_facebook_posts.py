import sys
import csv
import json
import os
import time

import facebook
from config imoprt FACEBOOK_API_ACCESS_TOKEN


access_token = FACEBOOK_API_ACCESS_TOKEN
graph = facebook.GraphAPI(access_token=access_token)

MAX_REQUEST_TIME = 350
DATA_PATH = "data"

def get_user_params(DATA_PATH):

    user_params = {}

    # get user input params
    user_params['inList'] = os.path.join(DATA_PATH, 'facebook_corpus.csv')
    user_params['rawDir'] = os.path.join(DATA_PATH, 'facebook_rawdata/')

    # apply defaults
    if user_params['inList'] == '':
        user_params['inList'] = './facebook_corpus.csv'
    if user_params['rawDir'] == '':
        user_params['rawDir'] = './facebook_rawdata/'

    return user_params

def get_lists(in_filename):

    fp = open(in_filename, 'r')
    reader = csv.reader(fp, delimiter=',', quotechar='"')
    total_list = []

    for row in reader:
        total_list.append(row)

    return total_list

def purge_already_fetched(fetch_list, raw_dir):

    # list of tweet ids that still need downloading
    rem_list = []
    count_done = 0

    # check each tweet to see if we have it
    for item in fetch_list:

        # check if json file exists
        comments_file = os.path.join(raw_dir, item[1] + '.json')
        if os.path.exists(comments_file):

            # attempt to parse json file
            try:
                parse_to_json(comments_file)
                count_done += 1
            except RuntimeError:
                print ("Error parsing", item)
                rem_list.append(item)
        else:
            rem_list.append(item)

    print ("We have already downloaded %i comment." % count_done)

    return rem_list

def parse_to_json(filename):

    # read tweet
    fp = open(filename, 'rb')

    # parse json
    try:
        comment_json = json.load(fp)
    except ValueError:
        raise RuntimeError('error parsing json')

    # look for twitter api error msgs
    if 'error' in comment_json or 'errors' in comment_json:
        raise RuntimeError('error in downloaded tweet')

    # extract creation date and tweet text
    return [comment_json['created_time'], comment_json['message']]

def download_comments(fetch_list, raw_dir):

    # ensure raw data directory exists
    if not os.path.exists(raw_dir):
        os.mkdir(raw_dir)

    # stay within rate limits
    download_pause_sec = 3600 / MAX_REQUEST_TIME

    # download tweets
    for idx in range(0, len(fetch_list)):
        item = fetch_list[idx]
        print (item)

        # print status
        print ('--> downloading comment #%s (%d of %d)' % \
              (item[1], idx + 1, len(fetch_list)))
        try:
            print('item id', item[1])
            result = graph.get_object(id=item[1])
            json_data = json.dumps(result)
            print(json_data)
        except (facebook.GraphAPIError) as e:
            fatal = True

            if e.code == 803:
                print ("Comment missing: ", item)
                fatal = False
                break

            if fatal:
                raise
            else:
                continue

        with open(raw_dir + item[1] + '.json', "w") as f:
            f.write(json_data + '\n')

    return

def main():
    # get user parameters
    user_params = get_user_params(DATA_PATH)
    print (user_params)

    # get fetch list
    total_list = get_lists(user_params['inList'])

    # remove already fetched or missing posts
    fetch_list = purge_already_fetched(total_list, user_params['rawDir'])

    print ("Fetching %i facebook commments..." % len(fetch_list))

    if fetch_list:
        # start fetching data from twitter
        download_comments(fetch_list, user_params['rawDir'])

        # second pass for any failed downloads
        fetch_list = purge_already_fetched(total_list, user_params['rawDir'])
        if fetch_list:
            print ('\nStarting second pass to retry %i failed downloads...' % len(fetch_list))
            download_comments(fetch_list, user_params['rawDir'])
    else:
        print ("Nothing to fetch any more.")

if __name__ == '__main__':
    main()