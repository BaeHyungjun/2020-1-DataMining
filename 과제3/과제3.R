# 20152410 배형준 데이터마이닝 과제3

#The response is a categorical variable with 10 classes coded from 0 to 9. All predictors are quantitative.

#1. Construct a random forest classifier, report the test classification error and make the confusion matrix. Note "ranger" is faster than "randomForest".

#2. Construct a boosting classifier, report the test classification error and make the confusion matrix. Note "xgboost" is faster than "gbm".




##### load dataset

student = 20152410

mnist_train = read.csv('./MNIST_train_small.csv', header=TRUE)
mnist_test = read.csv('./MNIST_test_small.csv', header=TRUE)

train_data = mnist_train[, 2:785]
train_label = as.factor(mnist_train$y)

test_data = mnist_test[, 2:785]
test_label = as.factor(mnist_test$y)




##### 1. random forest classifier

library(caret)
library(ranger)

set.seed(student)
ranger_model = ranger(x = train_data,
                      y = train_label)
ranger_model

ranger_pred = predict(ranger_model, data=test_data,
                      num.trees=ranger_model$num.trees)

ranger_clf_error = mean(ranger_pred$predictions != test_label)
cat('Test error of ranger classifier : ', 100*ranger_clf_error, '%')

ranger_table = table(ranger_pred$predictions, test_label)
ranger_cfm = confusionMatrix(ranger_table, mode='everything')
ranger_cfm




##### 2. boosting classifier

library(xgboost)

train_dmatrix = xgb.DMatrix(data=as.matrix(train_data), label=as.integer(train_label)-1)
test_dmatrix = xgb.DMatrix(data=as.matrix(test_data), label=as.integer(test_label)-1)

xgb_params = list(eta=0.2,
                  num_class=length(levels(train_label)),
                  objective='multi:softmax',
                  eval_metric='mlogloss')

set.seed(student)
xgb_model = xgb.train(data=train_dmatrix,
                      params=xgb_params,
                      nrounds=500,
                      early_stopping_rounds=20,
                      watchlist=list(val1=train_dmatrix, val2=test_dmatrix),
                      verbose=1)
xgb_model

xgb_pred = as.factor(predict(xgb_model, newdata=test_dmatrix))

xgb_clf_error = mean(xgb_pred != test_label)
cat('Test error of xgboost classifier : ', 100*xgb_clf_error, '%')

xgb_table = table(xgb_pred, test_label)
xgb_cfm = confusionMatrix(xgb_table, mode='everything')
xgb_cfm
