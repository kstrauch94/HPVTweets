# HPVTweets

All the resources (such as: annotation file, parsed tweet file, the bing liu sentiment lexicons...) should be placed in a folder named `data` inside this project.
This is because it is the default value in the parser. It is annoying to add all those options!

In order to reproduce the results run:

    python classify_tweets.py --tweets-file ./data/dataset/tweet_for_dp.txt.predict --annotations ./data/dataset/TweetsAnnotation.txt -c lr_h
    --rm-url --stem --bingliusent --depsent --negwords --bigramsent --unigramsent --subjscore --sentnet --argscores --scale --class-weights --optim-single

## Dependencies

python >= 3x

pandas >= 0.22.0

nltk >= 3.2.5

hunspell >= 0.5.3

sklaern >= 0.19.1


