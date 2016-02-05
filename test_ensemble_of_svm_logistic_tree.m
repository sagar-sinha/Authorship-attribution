predictions_ada = load('test_adaboost_10kuni_20kbi.txt');
predictions_svm = load('test_svm_40kuni_40kni_40ktri.txt');
predictions_logistic = load('test_logistic_40kuni_40kbi_40ktri.txt');

predictions_final = (predictions_ada + predictions_svm + predictions_logistic)/3;
predictions_final(predictions_final > 0.5) = 1;
predictions_final(predictions_final < 0.5) = 0;
dlmwrite('test_leaderboard_ada_svm_log.txt', predictions_final);