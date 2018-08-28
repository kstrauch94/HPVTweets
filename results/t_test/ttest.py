import argparse
from scipy import stats


def parse_args():

	parser = argparse.ArgumentParser(description='Tweet classifier architecture')
	parser.add_argument('--res1',required = True, type = str, help = "File with model 1 results. One float per line!")
	parser.add_argument('--res2',required = True, type = str, help = "File with model 2 results. One float per line!")
	return parser.parse_args()


def get_results(infile):

	raw = open(infile).readlines()
	res = [float(x.replace('\n','')) for x in raw]
	return res

if __name__ == "__main__":

	args = parse_args()
	
	res_1 = get_results(args.res1)
	res_2 = get_results(args.res2)
	
	st,pvalue = stats.ttest_ind(res_1,res_2)
	
	if pvalue > 0.05:
		print('Samples are likely drawn from the same distributions (fail to reject H0) : {}'.format(pvalue))
	else:
		print('Samples are likely drawn from different distributions (reject H0) : {}'.format(pvalue))
		

