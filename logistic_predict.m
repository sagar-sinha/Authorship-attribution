function [ predicted_label, score ] = logistic_predict( train_x, train_y, test_x )
    addpath('./liblinear');
     model = train(train_y, train_x, ['-s 0', 'col']);
    [predicted_label, score] = predict(ones(size(test_x,1),1), sparse(test_x), model, ['-q', 'col']);
end