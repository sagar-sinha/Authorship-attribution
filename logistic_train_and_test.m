addpath('./liblinear');
trainingPoints = [train_unigram_40001 train_bigram_40000 train_trigram_40000];
testingPoints = [test_unigram_40001 test_bigram_40000 test_trigram_40000];
trainingLabels = train_data_vector_label;
trainingPoints_sparse = sparse(trainingPoints);
[predictions_logistic, ~] = logistic_predict( trainingPoints_sparse, trainingLabels, testingPoints );
dlmwrite('test_logistic_40kuni_40kbi_40ktri.txt', predictions_logistic);