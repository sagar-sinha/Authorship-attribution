Authorship atribution is a supervised binary classifier which predicts a piece of text to be the writings of a particular author among a group of authors. For a particular task to identify Gina Kolata's writing among her peers, the system achieved an accuracy of 94.27% on a test set of 1346 excerpts from New York Times articles.

These intructions will help you get started.
1) Run the following Python functions:

	generate_training_labels() to generate train_data_vector_label  [training Labels]
	generate_unigram_files_train_test() to generate the files train_unigram_40001 and test_unigram_40001  [features]
	generate_bigram_files_train_test() to generate the files train_bigram_40000 and test_bigram_40000	  [features]
	generate_trigram_files_train_test() to generate the files train_trigram_40000 and test_trigram_40000  [features]
	
2) Note that all the matlab scripts mentioned below assume that the files generated in the previous step reside in the same directory as them.

3) Open matlab and run the script 'load_training_and_testing.m' to load 
train_data_vector_label, train_unigram_40001, test_unigram_40001, train_bigram_40000, test_bigram_40000, train_trigram_40000 and test_trigram_40000 into the workspace   [training data]

4) To generate the predictions for Logistic Regression run the file - 'logistic_train_and_test.m'. This will also generate a test prediction file 'test_logistic_40kuni_40kbi_40ktri.txt'.   [Required by the ensemble model]

5) To generate the predictions for SVM run the file - 'svm_train_and_test.m'. This will also generate a test prediction file 'test_svm_40kuni_40kni_40ktri.txt'.							 [Required by the ensemble model]

6) To generate the predictions for Adaboost run the file - 'adaboost.m'. This will also generate a test prediction file 'test_adaboost_10kuni_20kbi.txt'. This will take a long time.		 [Required by the ensemble model] 

7) To generate final predictions for the  model: an ensemble of Logistic Regression, SVM and Adaboost run 'test_ensemble_of_svm_logistic_tree.m'. This will also generate a test prediction file 'test_leaderboard_ada_svm_log.txt'.
	Note: This script requires that the files generated in step 4, 5 and 6 are present in the working directory.
