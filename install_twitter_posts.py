import sys
import csv
import json
import os
import time

try:
    import twitter
except ImportError:
    print("""\
You need to install python-twitter
""")

    sys.exit(1)
    print("start")

from twitterauth import CONSUMER_KEY, CONSUMER_SECRET, ACCESS_TOKEN_KEY, ACCESS_TOKEN_SECRET
api = twitter.Api(consumer_key=CONSUMER_KEY, consumer_secret=CONSUMER_SECRET,
                  access_token_key=ACCESS_TOKEN_KEY, access_token_secret=ACCESS_TOKEN_SECRET)


MAX_TWEETS_PER_HR = 350

DATA_PATH = "data"

def get_user_params(DATA_PATH):

    user_params = {}

    # get user input params
    user_params['inList'] = os.path.join(DATA_PATH, 'twitter-corpus.csv')
    user_params['rawDir'] = os.path.join(DATA_PATH, 'rawdata/')

    # apply defaults
    if user_params['inList'] == '':
        user_params['inList'] = '/twitter-corpus.csv'
  
    return user_params


def read_total_list(in_filename):

    # read total fetch list csv
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
        tweet_file = os.path.join(raw_dir, item[1] + '.json')
        if os.path.exists(tweet_file):

            # attempt to parse json file
            try:
                parse_tweet_json(tweet_file)
                count_done += 1
            except RuntimeError:
                print ("Error parsing", item)
                rem_list.append(item)
        else:
            rem_list.append(item)

    print ("We have already downloaded %i tweets." % count_done)

    return rem_list


def download_tweets(fetch_list, raw_dir):

    if not os.path.exists(raw_dir):
        os.mkdir(raw_dir)

    # stay within rate limits
    download_pause_sec = 3600 / MAX_TWEETS_PER_HR

    # download tweets
    for idx in range(0, len(fetch_list)):
        # stay in Twitter API rate limits
        print ('Pausing %d sec to obey Twitter API rate limits' % \
              (download_pause_sec))
        time.sleep(download_pause_sec)

        # current item
        item = fetch_list[idx]
        print (item)

        # print status
        print ('--> downloading tweet #%s (%d of %d)' % \
              (item[1], idx + 1, len(fetch_list)))

        try:
            print('item id', item[1])
            result = api.GetStatus(item[1])
            json_data = result.AsJsonString()

        except (twitter.TwitterError) as e:
            fatal = True
            for m in e.message:
                if m['code'] == 34:
                    print ("Tweet missing: ", item)
                    fatal = False
                    break
                elif m['code'] == 63:
                    print ("User of tweet '%s' has been suspended." % item)
                    fatal = False
                    break
                elif m['code'] == 88:
                    print ("Rate limit exceeded. Please lower max_tweets_per_hr.")
                    fatal = True
                    break
                elif m['code'] == 179:
                    print ("Not authorized to view this tweet.")
                    fatal = False
                    break
                elif m['code'] == 144:
                    print ("Not found")
                    fatal = False
                    break

            if fatal:
                raise
            else:
                continue

        with open(raw_dir + item[1] + '.json', "w") as f:
            f.write(json_data + "\n")

    return


def parse_tweet_json(filename):

    # read tweet
    fp = open(filename, 'rb')

    # parse json
    try:
        tweet_json = json.load(fp)
    except ValueError:
        raise RuntimeError('error parsing json')

    # look for twitter api error msgs
    if 'error' in tweet_json or 'errors' in tweet_json:
        raise RuntimeError('error in downloaded tweet')

    # extract creation date and tweet text
    return [tweet_json['created_at'], tweet_json['text']]

def main():
    # get user parameters
    user_params = get_user_params(DATA_PATH)
    print (user_params)

    # get fetch list
    total_list = read_total_list(user_params['inList'])

    # remove already fetched or missing tweets
    fetch_list = purge_already_fetched(total_list, user_params['rawDir'])
    print ("Fetching %i tweets..." % len(fetch_list))

    if fetch_list:
        # start fetching data from twitter
        download_tweets(fetch_list, user_params['rawDir'])

        # second pass for any failed downloads
        fetch_list = purge_already_fetched(total_list, user_params['rawDir'])
        if fetch_list:
            print ('\nStarting second pass to retry %i failed downloads...' % len(fetch_list))
            download_tweets(fetch_list, user_params['rawDir'])
    else:
        print ("Nothing to fetch any more.")

if __name__ == '__main__':
    main()
