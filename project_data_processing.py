# IMPORT STATEMENTS
from collections import Counter
from nltk.tokenize import word_tokenize
import sys
import nltk
from nltk.util import ngrams
from nltk import pos_tag
from nltk.tag.simplify import simplify_wsj_tag
#nltk.download('tag')
from nltk.tag import pos_tag, map_tag


# Returns all files in directory and its subdirectories
def get_all_files(directory):
    f = PlaintextCorpusReader(directory, '.*')
    return f.fileids()
	
# Returns absolute paths of all files in directory and its subdirectories. SEEMS TO LIST PATHS IN ALPHABETICAL ORDER.
def get_all_filepaths(directory):
	filename_list = get_all_files(directory)
	return [directory + "/" + f for f in filename_list]

# os.path.abspath("")  ----------> /mnt/castor/seas_home/g/gjimmy/Documents/Homeworks/CL_HW4	
# os.path.abspath("/mnt/castor/seas_home/g/gjimmy/Documents/Homeworks/CL_HW4/") -------------> /mnt/castor/seas_home/g/gjimmy/Documents/Homeworks/CL_HW4
# NOTE : os.path.abspath doesn't work well for ~ paths
# Returns absolute paths of all IMMEDIATE SUBDIRECTORIES in directory. MANUALLY SORTING PATHS IN ALPHABETICAL ORDER.
def get_all_immediate_subdirectory_paths(directory):
	return sorted([ os.path.join(os.path.abspath(directory), x) for x in next(os.walk(os.path.abspath(directory)))[1] ])
	
# Returns absolute paths of all IMMEDIATE FILES in directory. MANUALLY SORTING PATHS IN ALPHABETICAL ORDER.
def get_all_immediate_file_paths(directory):
	return sorted([ os.path.join(os.path.abspath(directory), x) for x in next(os.walk(os.path.abspath(directory)))[2] ])
	
# Returns absolute paths of all IMMEDIATE FILES ending with EXTENSION in directory. SEEMS TO LIST PATHS IN ALPHABETICAL ORDER.
# TESTED ---> ['/home1/g/gjimmy/Documents/Homeworks/CL_HW4/dev_00.config', '/home1/g/gjimmy/Documents/Homeworks/CL_HW4/dev_01.config',  .......... OUTPUTS ALL 40 config files
def get_all_immediate_file_paths_with_extension(directory, extension):
	return [ x for x in get_all_immediate_file_paths(directory) if x.endswith("."+extension)]
	
# Get the lines of a file
def get_file_lines(file):
	f = open(file)
	file_lines = f.readlines()
	return file_lines
	
def vectorize_training_data():
	train_data_file = "/home1/c/cis530/project/data/project_articles_train"
	train_data_file_lines = get_file_lines(train_data_file)
	train_data_file_line_token_counter_label_tuple_list = []
	for train_data_file_line in train_data_file_lines:
		train_data_file_line_split_list = train_data_file_line.split("\t")
		train_data_file_line_token_counter = Counter(word_tokenize(train_data_file_line_split_list[0].decode('utf8').lower()))
		train_data_file_line_label = int(train_data_file_line_split_list[1].strip())
		train_data_file_line_token_counter_label_tuple_list.append((train_data_file_line_token_counter, train_data_file_line_label))
		
	train_data_token_counter = Counter()
		
	for train_data_file_line_token_counter_label_tuple in train_data_file_line_token_counter_label_tuple_list:
		train_data_token_counter = train_data_token_counter + train_data_file_line_token_counter_label_tuple[0]
	
	most_common_thousand_tokens_list = sorted([x[0] for x in train_data_token_counter.most_common(10000)])
	
	train_data_vector_list = list()
	
	for train_data_file_line_token_counter_label_tuple in train_data_file_line_token_counter_label_tuple_list:
		train_data_vector = list()
		for most_common_thousand_token in most_common_thousand_tokens_list:
			if train_data_file_line_token_counter_label_tuple[0][most_common_thousand_token]:
				train_data_vector.append(train_data_file_line_token_counter_label_tuple[0][most_common_thousand_token])
			else: 
				train_data_vector.append(0)
		train_data_vector_list.append(train_data_vector)
		
	outfile1 = "train_data_vector_10000_most_common"
	fo1 = open(outfile1, 'w')

	for train_data_vector in train_data_vector_list:
		for index, train_data_vector_element in enumerate(train_data_vector):
			fo1.write(str(train_data_vector_element))
			if(index != len(train_data_vector) - 1):
				fo1.write(" ")
		fo1.write("\n")
		
	# outfile2 = "train_data_vector_label"
	# fo2 = open(outfile2, 'w')

	# for train_data_file_line_token_counter_label_tuple in train_data_file_line_token_counter_label_tuple_list:
		# fo2.write(str(train_data_file_line_token_counter_label_tuple[1]))
		# fo2.write("\n")
		
		
def get_pos_bigrams():
	train_data_file_line_label_tuple_list = get_training_data()
	train_data_all_file_lines_universal_tag_bigram_counter = Counter()
	train_data_file_line_universal_tag_bigram_counter_list = []
	for train_data_file_line_label_tuple in train_data_file_line_label_tuple_list:
		train_data_file_line_tokens = word_tokenize(train_data_file_line_label_tuple[0].decode('utf8').lower())
		train_data_file_line_token_tag_tuple_list = pos_tag(train_data_file_line_tokens)
		#print train_data_file_line_label_tuple, "\n", [ (word, map_tag('en-ptb', 'universal', tag).encode('utf8')) for word, tag in train_data_file_line_token_tag_tuple_list]
		train_data_file_line_universal_tag_list = [ map_tag('en-ptb', 'universal', tag).encode('utf8') for word, tag in train_data_file_line_token_tag_tuple_list]
		train_data_file_line_universal_tag_bigram_list = list(ngrams(train_data_file_line_universal_tag_list,2))
		#print train_data_file_line_label_tuple, "\n", train_data_file_line_universal_tag_bigram_list
		train_data_file_line_universal_tag_bigram_counter = Counter(train_data_file_line_universal_tag_bigram_list)
		train_data_file_line_universal_tag_bigram_counter_list.append(train_data_file_line_universal_tag_bigram_counter)
		#print train_data_file_line_universal_tag_bigram_counter.most_common()
		train_data_all_file_lines_universal_tag_bigram_counter = train_data_all_file_lines_universal_tag_bigram_counter + train_data_file_line_universal_tag_bigram_counter
	
	#print len(train_data_all_file_lines_universal_tag_bigram_counter)
	
	all_pos_bigrams_list = sorted([x[0] for x in train_data_all_file_lines_universal_tag_bigram_counter.most_common()])
	
	train_data_pos_bigram_vector_list = list()
	
	for train_data_file_line_universal_tag_bigram_counter in train_data_file_line_universal_tag_bigram_counter_list:
		train_data_pos_bigram_vector = list()
		for pos_bigram in all_pos_bigrams_list:
			if train_data_file_line_universal_tag_bigram_counter[pos_bigram]:
				train_data_pos_bigram_vector.append(train_data_file_line_universal_tag_bigram_counter[pos_bigram])
			else: 
				train_data_pos_bigram_vector.append(0)
		train_data_pos_bigram_vector_list.append(train_data_pos_bigram_vector)
		
	outfile1 = "train_data_pos_bigram_vector_10000_most_common"
	fo1 = open(outfile1, 'w')

	for train_data_pos_bigram_vector in train_data_pos_bigram_vector_list:
		for index, train_data_pos_bigram_vector_element in enumerate(train_data_pos_bigram_vector):
			fo1.write(str(train_data_pos_bigram_vector_element))
			if(index != len(train_data_pos_bigram_vector) - 1):
				fo1.write(" ")
		fo1.write("\n")
		
################################# BETTER CODE #################################################################################
		
def get_training_data():
	train_data_file = "/home1/c/cis530/project/data/project_articles_train"
	train_data_file_lines = get_file_lines(train_data_file)
	train_data_file_line_label_tuple_list = []
	for train_data_file_line in train_data_file_lines:
		train_data_file_line_split_list = train_data_file_line.split("\t")
		train_data_file_line_final = train_data_file_line_split_list[0].strip()
		train_data_file_line_label = int(train_data_file_line_split_list[1].strip())
		train_data_file_line_label_tuple_list.append((train_data_file_line_final, train_data_file_line_label))
	return train_data_file_line_label_tuple_list
	
def get_testing_data():
	test_data_file = "/home1/c/cis530/project/data/project_articles_test"
	test_data_file_lines = get_file_lines(test_data_file)
	return [ test_data_file_line.strip() for test_data_file_line in test_data_file_lines]
	
def get_list_list_tokens_from_list_string(list_file_lines):
	list_list_file_tokens = []
	for file_line in list_file_lines:
		file_line_tokens = word_tokenize(file_line.decode('utf8').lower())
		file_line_tokens = [ file_line_token.encode('utf8') for file_line_token in file_line_tokens]
		list_list_file_tokens.append(file_line_tokens)
	return list_list_file_tokens
	

def get_list_list_bigrams_from_list_list_tokens(list_list_tokens):
	list_list_bigrams = []
	for list_tokens in list_list_tokens:
		list_bigrams = list(ngrams(list_tokens,2))
		list_list_bigrams.append(list_bigrams)
	return list_list_bigrams
		
def get_list_list_trigrams_from_list_list_tokens(list_list_tokens):
	list_list_trigrams = []
	for list_tokens in list_list_tokens:
		list_trigrams = list(ngrams(list_tokens,3))
		list_list_trigrams.append(list_trigrams)
	return list_list_trigrams
	
def get_list_list_quadgrams_from_list_list_tokens(list_list_tokens):
	list_list_quadgrams = []
	for list_tokens in list_list_tokens:
		list_quadgrams = list(ngrams(list_tokens,4))
		list_list_quadgrams.append(list_quadgrams)
	return list_list_quadgrams
	
def get_list_list_pos_from_list_list_tokens(list_list_tokens):
	list_list_pos = []
	for list_tokens in list_list_tokens:
		list_tokens_decoded = [ x.decode('utf8') for x in list_tokens] #pos tagger needs decoded tokens
		list_token_pos_tuple = pos_tag(list_tokens_decoded)
		list_universal_pos_tag = [ map_tag('en-ptb', 'universal', tag).encode('utf8') for word, tag in list_token_pos_tuple]
		list_list_pos.append(list_universal_pos_tag)
	return list_list_pos

def get_list_list_pos_bigrams_from_list_list_pos(list_list_pos):
	return get_list_list_bigrams_from_list_list_tokens(list_list_pos)
	
def get_list_list_pos_trigrams_from_list_list_pos(list_list_pos):
	return get_list_list_trigrams_from_list_list_tokens(list_list_pos)
	
def get_list_list_pos_quadgrams_from_list_list_pos(list_list_pos):
	return get_list_list_quadgrams_from_list_list_tokens(list_list_pos)

def get_list_counters_from_list_list_ngrams(list_list_ngrams):
	list_counters = []
	for list_ngrams in list_list_ngrams:
		cntr = Counter(list_ngrams)
		list_counters.append(cntr)
	return list_counters

def get_aggregated_counter_from_list_counters(list_counters):
	aggregated_counter = Counter()
	for cntr in list_counters:
		aggregated_counter.update(cntr)
	return aggregated_counter

def get_vector_from_list_feature_ngrams_list_counters(list_feature_ngrams, list_counters):
	vector_list = []
	for counter in list_counters:
		vector = list()
		for feature_ngram in list_feature_ngrams:
			if counter[feature_ngram]:
				vector.append(counter[feature_ngram])
			else: 
				vector.append(0)
		vector_list.append(vector)
	return vector_list
	
def write_vector_list_to_file(vector_list, file_path):
	fo = open(file_path, 'w')
	for vector in vector_list:
			for index, vector_element in enumerate(vector):
				fo.write(str(vector_element))
				if(index != len(vector) - 1):
					fo.write(" ")
			fo.write("\n")

def write_list_feature_ngrams_to_file(list_feature_ngrams, file_path):
	fo = open(file_path, 'w')
	for feature_ngram in list_feature_ngrams:
		fo.write(str(feature_ngram)+"\n")
		
def get_stopwords_list():
	stopwords_file = "/home1/c/cis530/hw4/stopwords.txt"
	stopwords_file_lines = get_file_lines(stopwords_file)
	return [stopwords_file_line.strip() for stopwords_file_line in stopwords_file_lines]
	
def generate_training_labels():
	train_data_file_line_label_tuple_list = get_training_data()
	outfile2 = "train_data_vector_label"
	fo2 = open(outfile2, 'w')

	for train_data_file_line_label_tuple in train_data_file_line_label_tuple_list:
		fo2.write(str(train_data_file_line_label_tuple[1]))
		fo2.write("\n")
	
def generate_unigram_files_train_test():
	train_file_data = get_training_data()
	train_file_lines = [x[0] for x in train_file_data]
	test_file_lines = get_testing_data()
	
	train_list_list_unigrams = get_list_list_tokens_from_list_string(train_file_lines)
	test_list_list_unigrams = get_list_list_tokens_from_list_string(test_file_lines)
	
	train_list_counters = get_list_counters_from_list_list_ngrams(train_list_list_unigrams)
	test_list_counters = get_list_counters_from_list_list_ngrams(test_list_list_unigrams)
	
	# Setting feature ngrams
	train_aggregated_counter = get_aggregated_counter_from_list_counters(train_list_counters)
	# Change the below line to set number of number_of_most_common_ngrams
	number_of_most_common_ngrams =  len(train_aggregated_counter) #len(train_aggregated_counter) for all
	list_feature_ngrams = [x[0] for x in train_aggregated_counter.most_common(number_of_most_common_ngrams)]
	outfile_feature_ngrams = "feature_ngrams_unigram" + "_" + str(number_of_most_common_ngrams) 
	write_list_feature_ngrams_to_file(list_feature_ngrams, outfile_feature_ngrams)
	
	train_vector_list = get_vector_from_list_feature_ngrams_list_counters(list_feature_ngrams, train_list_counters)
	test_vector_list = get_vector_from_list_feature_ngrams_list_counters(list_feature_ngrams, test_list_counters)
	
	outfile_train = "train_unigram" + "_" + str(number_of_most_common_ngrams) 
	outfile_test = "test_unigram" + "_" + str(number_of_most_common_ngrams) 
	
	write_vector_list_to_file(train_vector_list, outfile_train)
	write_vector_list_to_file(test_vector_list, outfile_test)
	
def generate_bigram_files_train_test():
	train_file_data = get_training_data()
	train_file_lines = [x[0] for x in train_file_data]
	test_file_lines = get_testing_data()
	
	train_list_list_bigrams = get_list_list_bigrams_from_list_list_tokens(get_list_list_tokens_from_list_string(train_file_lines))
	test_list_list_bigrams = get_list_list_bigrams_from_list_list_tokens(get_list_list_tokens_from_list_string(test_file_lines))
	
	train_list_counters = get_list_counters_from_list_list_ngrams(train_list_list_bigrams)
	test_list_counters = get_list_counters_from_list_list_ngrams(test_list_list_bigrams)
	
	# Setting feature ngrams
	train_aggregated_counter = get_aggregated_counter_from_list_counters(train_list_counters)
	# Change the below line to set number of number_of_most_common_ngrams
	number_of_most_common_ngrams =  40000 #len(train_aggregated_counter) for all
	list_feature_ngrams = [x[0] for x in train_aggregated_counter.most_common(number_of_most_common_ngrams)]
	outfile_feature_ngrams = "feature_ngrams_bigram" + "_" + str(number_of_most_common_ngrams) 
	write_list_feature_ngrams_to_file(list_feature_ngrams, outfile_feature_ngrams)
	
	train_vector_list = get_vector_from_list_feature_ngrams_list_counters(list_feature_ngrams, train_list_counters)
	test_vector_list = get_vector_from_list_feature_ngrams_list_counters(list_feature_ngrams, test_list_counters)
	
	outfile_train = "train_bigram" + "_" + str(number_of_most_common_ngrams) 
	outfile_test = "test_bigram" + "_" + str(number_of_most_common_ngrams) 
	
	write_vector_list_to_file(train_vector_list, outfile_train)
	write_vector_list_to_file(test_vector_list, outfile_test)
	
def generate_trigram_files_train_test():
	train_file_data = get_training_data()
	train_file_lines = [x[0] for x in train_file_data]
	test_file_lines = get_testing_data()
	
	train_list_list_trigrams = get_list_list_trigrams_from_list_list_tokens(get_list_list_tokens_from_list_string(train_file_lines))
	test_list_list_trigrams = get_list_list_trigrams_from_list_list_tokens(get_list_list_tokens_from_list_string(test_file_lines))
	
	train_list_counters = get_list_counters_from_list_list_ngrams(train_list_list_trigrams)
	test_list_counters = get_list_counters_from_list_list_ngrams(test_list_list_trigrams)
	
	# Setting feature ngrams
	train_aggregated_counter = get_aggregated_counter_from_list_counters(train_list_counters)
	# Change the below line to set number of number_of_most_common_ngrams
	number_of_most_common_ngrams =  40000 #len(train_aggregated_counter) for all
	list_feature_ngrams = [x[0] for x in train_aggregated_counter.most_common(number_of_most_common_ngrams)]
	outfile_feature_ngrams = "feature_ngrams_trigram" + "_" + str(number_of_most_common_ngrams) 
	write_list_feature_ngrams_to_file(list_feature_ngrams, outfile_feature_ngrams)
	
	train_vector_list = get_vector_from_list_feature_ngrams_list_counters(list_feature_ngrams, train_list_counters)
	test_vector_list = get_vector_from_list_feature_ngrams_list_counters(list_feature_ngrams, test_list_counters)
	
	outfile_train = "train_trigram" + "_" + str(number_of_most_common_ngrams) 
	outfile_test = "test_trigram" + "_" + str(number_of_most_common_ngrams) 
	
	write_vector_list_to_file(train_vector_list, outfile_train)
	write_vector_list_to_file(test_vector_list, outfile_test)
	
def generate_unigram_stopwords_files_train_test():
	train_file_data = get_training_data()
	train_file_lines = [x[0] for x in train_file_data]
	test_file_lines = get_testing_data()
	
	train_list_list_unigrams = get_list_list_tokens_from_list_string(train_file_lines)
	test_list_list_unigrams = get_list_list_tokens_from_list_string(test_file_lines)
	
	train_list_counters = get_list_counters_from_list_list_ngrams(train_list_list_unigrams)
	test_list_counters = get_list_counters_from_list_list_ngrams(test_list_list_unigrams)
	
	# Setting feature ngrams
	train_aggregated_counter = get_aggregated_counter_from_list_counters(train_list_counters)
	# Change the below line to set number of number_of_most_common_ngrams
	number_of_most_common_ngrams =  len(train_aggregated_counter) #len(train_aggregated_counter) for all
	list_feature_ngrams = [x[0] for x in train_aggregated_counter.most_common(number_of_most_common_ngrams)]
	stopword_list = get_stopwords_list()
	list_feature_ngrams_stopwords = [ x for x in list_feature_ngrams if x in stopword_list]
	outfile_feature_ngrams = "feature_ngrams_unigram_stopword" + "_" + str(len(list_feature_ngrams_stopwords)) 
	write_list_feature_ngrams_to_file(list_feature_ngrams_stopwords, outfile_feature_ngrams)
	
	train_vector_list = get_vector_from_list_feature_ngrams_list_counters(list_feature_ngrams_stopwords, train_list_counters)
	test_vector_list = get_vector_from_list_feature_ngrams_list_counters(list_feature_ngrams_stopwords, test_list_counters)
	
	outfile_train = "train_unigram_stopword" + "_" + str(len(list_feature_ngrams_stopwords)) 
	outfile_test = "test_unigram_stopword" + "_" + str(len(list_feature_ngrams_stopwords)) 
	
	write_vector_list_to_file(train_vector_list, outfile_train)
	write_vector_list_to_file(test_vector_list, outfile_test)
	
def generate_pos_unigram_files_train_test():
	train_file_data = get_training_data()
	train_file_lines = [x[0] for x in train_file_data]
	test_file_lines = get_testing_data()
	
	train_list_list_pos_unigrams = get_list_list_pos_from_list_list_tokens(get_list_list_tokens_from_list_string(train_file_lines))
	test_list_list_pos_unigrams = get_list_list_pos_from_list_list_tokens(get_list_list_tokens_from_list_string(test_file_lines))
	
	train_list_counters = get_list_counters_from_list_list_ngrams(train_list_list_pos_unigrams)
	test_list_counters = get_list_counters_from_list_list_ngrams(test_list_list_pos_unigrams)
	
	# Setting feature ngrams
	train_aggregated_counter = get_aggregated_counter_from_list_counters(train_list_counters)
	# Change the below line to set number of number_of_most_common_ngrams
	number_of_most_common_ngrams =  len(train_aggregated_counter) #len(train_aggregated_counter) for all
	list_feature_ngrams = [x[0] for x in train_aggregated_counter.most_common(number_of_most_common_ngrams)]
	outfile_feature_ngrams = "feature_ngrams_pos_unigram" + "_" + str(number_of_most_common_ngrams) 
	write_list_feature_ngrams_to_file(list_feature_ngrams, outfile_feature_ngrams)
	
	train_vector_list = get_vector_from_list_feature_ngrams_list_counters(list_feature_ngrams, train_list_counters)
	test_vector_list = get_vector_from_list_feature_ngrams_list_counters(list_feature_ngrams, test_list_counters)
	
	outfile_train = "train_pos_unigram" + "_" + str(number_of_most_common_ngrams) 
	outfile_test = "test_pos_unigram" + "_" + str(number_of_most_common_ngrams) 
	
	write_vector_list_to_file(train_vector_list, outfile_train)
	write_vector_list_to_file(test_vector_list, outfile_test)
	
def generate_pos_bigram_files_train_test():
	train_file_data = get_training_data()
	train_file_lines = [x[0] for x in train_file_data]
	test_file_lines = get_testing_data()
	
	train_list_list_pos_bigrams = get_list_list_pos_bigrams_from_list_list_pos(get_list_list_pos_from_list_list_tokens(get_list_list_tokens_from_list_string(train_file_lines)))
	test_list_list_pos_bigrams = get_list_list_pos_bigrams_from_list_list_pos(get_list_list_pos_from_list_list_tokens(get_list_list_tokens_from_list_string(test_file_lines)))
	
	train_list_counters = get_list_counters_from_list_list_ngrams(train_list_list_pos_bigrams)
	test_list_counters = get_list_counters_from_list_list_ngrams(test_list_list_pos_bigrams)
	
	# Setting feature ngrams
	train_aggregated_counter = get_aggregated_counter_from_list_counters(train_list_counters)
	# Change the below line to set number of number_of_most_common_ngrams
	number_of_most_common_ngrams =  len(train_aggregated_counter) #len(train_aggregated_counter) for all
	list_feature_ngrams = [x[0] for x in train_aggregated_counter.most_common(number_of_most_common_ngrams)]
	outfile_feature_ngrams = "feature_ngrams_pos_bigram" + "_" + str(number_of_most_common_ngrams) 
	write_list_feature_ngrams_to_file(list_feature_ngrams, outfile_feature_ngrams)
	
	train_vector_list = get_vector_from_list_feature_ngrams_list_counters(list_feature_ngrams, train_list_counters)
	test_vector_list = get_vector_from_list_feature_ngrams_list_counters(list_feature_ngrams, test_list_counters)
	
	outfile_train = "train_pos_bigram" + "_" + str(number_of_most_common_ngrams) 
	outfile_test = "test_pos_bigram" + "_" + str(number_of_most_common_ngrams) 
	
	write_vector_list_to_file(train_vector_list, outfile_train)
	write_vector_list_to_file(test_vector_list, outfile_test)
	
def generate_pos_trigram_files_train_test():
	train_file_data = get_training_data()
	train_file_lines = [x[0] for x in train_file_data]
	test_file_lines = get_testing_data()
	
	train_list_list_pos_trigrams = get_list_list_pos_trigrams_from_list_list_pos(get_list_list_pos_from_list_list_tokens(get_list_list_tokens_from_list_string(train_file_lines)))
	test_list_list_pos_trigrams = get_list_list_pos_trigrams_from_list_list_pos(get_list_list_pos_from_list_list_tokens(get_list_list_tokens_from_list_string(test_file_lines)))
	
	train_list_counters = get_list_counters_from_list_list_ngrams(train_list_list_pos_trigrams)
	test_list_counters = get_list_counters_from_list_list_ngrams(test_list_list_pos_trigrams)
	
	# Setting feature ngrams
	train_aggregated_counter = get_aggregated_counter_from_list_counters(train_list_counters)
	# Change the below line to set number of number_of_most_common_ngrams
	number_of_most_common_ngrams =  len(train_aggregated_counter) #len(train_aggregated_counter) for all
	list_feature_ngrams = [x[0] for x in train_aggregated_counter.most_common(number_of_most_common_ngrams)]
	outfile_feature_ngrams = "feature_ngrams_pos_trigram" + "_" + str(number_of_most_common_ngrams) 
	write_list_feature_ngrams_to_file(list_feature_ngrams, outfile_feature_ngrams)
	
	train_vector_list = get_vector_from_list_feature_ngrams_list_counters(list_feature_ngrams, train_list_counters)
	test_vector_list = get_vector_from_list_feature_ngrams_list_counters(list_feature_ngrams, test_list_counters)
	
	outfile_train = "train_pos_trigram" + "_" + str(number_of_most_common_ngrams) 
	outfile_test = "test_pos_trigram" + "_" + str(number_of_most_common_ngrams) 
	
	write_vector_list_to_file(train_vector_list, outfile_train)
	write_vector_list_to_file(test_vector_list, outfile_test)
	
def generate_pos_quadgram_files_train_test():
	train_file_data = get_training_data()
	train_file_lines = [x[0] for x in train_file_data]
	test_file_lines = get_testing_data()
	
	train_list_list_pos_quadgrams = get_list_list_pos_quadgrams_from_list_list_pos(get_list_list_pos_from_list_list_tokens(get_list_list_tokens_from_list_string(train_file_lines)))
	test_list_list_pos_quadgrams = get_list_list_pos_quadgrams_from_list_list_pos(get_list_list_pos_from_list_list_tokens(get_list_list_tokens_from_list_string(test_file_lines)))
	
	train_list_counters = get_list_counters_from_list_list_ngrams(train_list_list_pos_quadgrams)
	test_list_counters = get_list_counters_from_list_list_ngrams(test_list_list_pos_quadgrams)
	
	# Setting feature ngrams
	train_aggregated_counter = get_aggregated_counter_from_list_counters(train_list_counters)
	# Change the below line to set number of number_of_most_common_ngrams
	number_of_most_common_ngrams =  len(train_aggregated_counter) #len(train_aggregated_counter) for all
	list_feature_ngrams = [x[0] for x in train_aggregated_counter.most_common(number_of_most_common_ngrams)]
	outfile_feature_ngrams = "feature_ngrams_pos_quadgram" + "_" + str(number_of_most_common_ngrams) 
	write_list_feature_ngrams_to_file(list_feature_ngrams, outfile_feature_ngrams)
	
	train_vector_list = get_vector_from_list_feature_ngrams_list_counters(list_feature_ngrams, train_list_counters)
	test_vector_list = get_vector_from_list_feature_ngrams_list_counters(list_feature_ngrams, test_list_counters)
	
	outfile_train = "train_pos_quadgram" + "_" + str(number_of_most_common_ngrams) 
	outfile_test = "test_pos_quadgram" + "_" + str(number_of_most_common_ngrams) 
	
	write_vector_list_to_file(train_vector_list, outfile_train)
	write_vector_list_to_file(test_vector_list, outfile_test)
	

def test():
	generate_training_labels()
	