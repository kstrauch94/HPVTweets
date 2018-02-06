## DATASET

Here you should place the downloaded tweet and annotations ( https://sbmi.uth.edu/ontology/files/TweetsAnnotationResults.zip ) . 
Use the script `generate_data_file.py`:

	$ python3 generate_data_file.py -t tweet_file -o output_path

to generate file to be then fed as input to `TweeboParser` ( https://github.com/ikekonglp/TweeboParser ) used to create annotated data set (PoS, Dependency Parse).
The intermediate step is done in order to get rid of missing tweets ( and tweets with expired accounts) that make the parser crash. 
