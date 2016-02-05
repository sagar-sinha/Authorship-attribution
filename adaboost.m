trainingPoints = [train_unigram_40001(:,1:10000) train_bigram_40000(:,1:20000)];
trainingLabels = train_data_vector_label;

testingPoints = [test_unigram_40001(:,1:10000) test_bigram_40000(:,1:20000)];

ClassTreeEns = fitensemble(trainingPoints,trainingLabels,'AdaBoostM1',2000,'Tree');

[predictions_ada, ~] = predict(ClassTreeEns,testingPoints);
dlmwrite('test_adaboost_10kuni_20kbi.txt', predictions_ada); 