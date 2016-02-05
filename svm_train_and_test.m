trainingPoints = [train_unigram_40001 train_bigram_40000 train_trigram_40000];
testingPoints = [test_unigram_40001 test_bigram_40000 test_trigram_40000];
trainingLabels = train_data_vector_label;
Xtrain = sparse(trainingPoints);
Xtest = sparse(testingPoints);
Y_test_dummy = zeros(size(Xtest,1),1);
kernel = @(a,b)kernel_intersection(a,b);
predictions_svm = kernel_libsvm(Xtrain, trainingLabels, Xtest, Y_test_dummy, kernel);
dlmwrite('test_svm_40kuni_40kni_40ktri.txt', predictions_svm); 