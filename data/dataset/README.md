## DATASET

Here you should place the downloaded tweet and annotations ( https://sbmi.uth.edu/ontology/files/TweetsAnnotationResults.zip ) .

## STANDARD

Use the script `generate_data_file.py`:

	$ python3 generate_data_file.py -r tweet_file -o output_path

to generate file to be then fed as input to `TweeboParser` ( https://github.com/ikekonglp/TweeboParser ) used to create annotated data set (PoS, Dependency Parse).
The intermediate step is done in order to get rid of missing tweets ( and tweets with expired accounts) that make the parser crash. 

## SPELL CHECKED

Once you created the the file with the dependency it is possible to apply spell checking. For complete reproducibility the files using for spell check are provided.
This step is performed now for the following reason:
- need for tokenization
- need for pos tags (provided by parsing) for avoiding parsing urls and emoticons
- spell check is expensive
