from collections import defaultdict

import io
import re

LABEL = "label"
TWEET = "tweet"
NORMALIZED_LABEL = "normalized label"
    
def reader(path, new_name="tweets.tsv"):
    # Use this function to filter out tweets from the tweet file produced by the java program

    tweets = []

    with io.open(path, "r", encoding="utf-8") as tweet_file:
        for line in tweet_file.readlines():
            id, tweet = line.split("\t")

            if tweet[:-1] == "" or tweet[:-1] == "Twitter / Account gesperr": # the -1 takes the \n out
                continue
            
            tweets.append("\t".join([id, tweet]))
            
    
    with io.open(new_name, "w", encoding="utf-8") as new_file:
        new_file.writelines(tweets)
 
def merge_tweets_tags(tweets_path, annotations_path, output_name="merged_tweets.tsv"):

    # use this function to merge the tweet file and the annotation file into one
    # each line has the following format([] for readability):
    # tweet-id[\t]Label[\t]Tweet-text\n
    #
    # Output is also in utf-8 format

    tweets = {}            

    with io.open(tweets_path, "r", encoding="utf-8") as twts:
        for line in twts.readlines():
            id, tweet = line.split("\t")
            if int(id) not in tweets:
                tweets[int(id)] = {}
            tweets[int(id)][TWEET] = tweet.strip()
            
    with io.open(annotations_path, "r", encoding="utf-8") as annotations:
        for line in annotations.readlines():
            id, userid, label = line.split()
            if int(id) in tweets: # have to test here because annotations are not filtered and contain all 6000 labels
                tweets[int(id)][LABEL] = label.strip()

    with io.open(output_name, "w", encoding="utf-8") as out:
        for id in tweets:
            tweet = tweets[id][TWEET]
            label = tweets[id][LABEL]
            out.write(u"{}\t{}\t{}\n".format(id, label, tweet))

def read_tab_sep(filename):

    # use this function to read the file produced by the "merge_tweets_tags" function.
    # this returns a list of triples (id, label, tweet text)

    all_tweets = []

    with io.open(filename, "r", encoding="utf-8") as tweets:
        for line in tweets.readlines():
            id, label, tweet = line.split("\t")
            all_tweets.append([id.strip(), label.strip(), tweet.strip()])

    return all_tweets
    

def process_tweets(tweet_list):

    # use this function to process list of (id, tweet) tuples

    UNRELATED = "Unrelated"
    NEG = "Neg"
    
    NEG_LABEL = "Negative"
    
    all = {TWEET: [], LABEL: [], NORMALIZED_LABEL: []}
    
    related = {TWEET: [], LABEL: [], NORMALIZED_LABEL: []}
    
    negative = {TWEET: [], LABEL: [], NORMALIZED_LABEL: []}

    for label_tweet in tweet_list:
        label = label_tweet[0]
        tweet = label_tweet[1]
        
        # add tweet to all dict
        all[TWEET].append(tweet)
        all[LABEL].append(label)
        all[NORMALIZED_LABEL].append(label if label == UNRELATED else "Related")
        
        # add tweet to related if applicable
        if label != UNRELATED:
            related[TWEET].append(tweet)
            related[LABEL].append(label)
            related[NORMALIZED_LABEL].append(label if NEG not in label else NEG_LABEL)
            
        # add tweet to negative if applicable
        if NEG in label:
            negative[TWEET].append(tweet)
            negative[LABEL].append(label)
            negative[NORMALIZED_LABEL].append(label)
            
    print len(all[LABEL]), len(related[LABEL]), len(negative[LABEL])
                   
    return all, related, negative

def parse_pos_tags(path, procesed_name="processed_tags.txt"):

    # Use this function to parse the postag file from the java program used to tag the tweets

    with open(path, "r") as f:
        f.readline() # read first line with headers
        
        lines = []

        for line in f.readlines():
            splits = line.replace('"['," ").replace(']"'," ").split(" ")
            tweet_id = splits[0].split(",")[1]
            tags = []
            for word_tag in splits[1:-1]:
                word, tag = word_tag.split("\\t")
                tags.append(tag.replace("',", "").replace("'", "").replace('"",', ""))

            lines.append("{} {}\n".format(tweet_id, " ".join(tags)))
            

    with open(procesed_name, "w") as final: 
        final.writelines(lines)
