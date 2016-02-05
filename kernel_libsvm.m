function [yhat] = kernel_libsvm(X, Y, Xtest, Ytest, kernel)

addpath('./libsvm');
K = kernel(X, X);
Ktest = kernel(X, Xtest);

model = svmtrain(Y, [(1:size(K,1))' K], sprintf('-t 4 -c %g', 0.01));
[yhat, ~] = svmpredict(Ytest, [(1:size(Ktest,1))' Ktest], model);
