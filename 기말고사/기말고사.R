# 20152410 배형준 데이터마이닝 기말고사




##### Sequence of data analysis
# 1. load dataset, split dataset as train and validation using stratifed partition function
# 2. data modeling using caret to tune hyper parameter of candidate models
# 3. evaluate the models
# 4. choose models to use or to make ensemble
# 5. confime final model, predict class and posterior probability of Xtest




##### library package to use

library(caret)
library(dplyr)
library(ROCR) # for auc, acc, cutoff
library(glmnet)
library(kernlab) # for support vector machine
library(ranger) # for random forest as ranger
library(xgboost) # for xgboost
library(writexl) # for exporting answer
library(ggplot2)
library(corrplot)




##### 1. load dataset, split dataset as train and validation using stratifed partition based on target variable

trainset = read.csv('./Train.csv', header=TRUE)
testset = read.csv('./Xtest.csv', header=TRUE)

train_X = trainset[, 2:51]
train_Y = as.factor(trainset[, 52])
test_X = testset[, 2:51]




student = 20152410
train_size = 0.75
set.seed(student)
train_index = createDataPartition(train_Y, p=train_size)

X_train = train_X[train_index$Resample1, ]
Y_train = train_Y[train_index$Resample1]
X_val = train_X[-train_index$Resample1, ]
Y_val = train_Y[-train_index$Resample1]

factor_Y_train = ifelse(Y_train == '1', 'yes', 'no')

# check wheter data is stratifed based on target

cat(' ratio of trainset target :', summary(train_Y)[1] / summary(train_Y)[2],
    '\n ratio of train target :', summary(Y_train)[1] / summary(Y_train)[2],
    '\n ratio of validation target :', summary(Y_val)[1] / summary(Y_val)[2])




##### 2. data modeling using caret to tune hyper parameter of candidate models
### candidate : glmnet, svmradial, random forest, xgboost

### 2-1. glmnet

glmnet_tune_length = 10
glmnet_fold_number = 4
glmnet_train_control = trainControl(method='cv',
                                    number=glmnet_fold_number,
                                    search='random',
                                    classProbs = TRUE,
                                    summaryFunction=twoClassSummary)

glmnet_start = Sys.time()

set.seed(student)
glmnet_model = train(X_train,
                     factor_Y_train,
                     method='glmnet',
                     trControl=glmnet_train_control,
                     metric='ROC',
                     tuneLength=glmnet_tune_length,
                     preProcess = c('center', 'scale'))

glmnet_train_time = Sys.time() - glmnet_start
cat('Train time of glmnet model : '); glmnet_train_time

glmnet_model

best_alpha = as.numeric(glmnet_model$bestTune[1])
best_lambda = as.numeric(glmnet_model$bestTune[2])

glmnet_pred = predict(glmnet_model, newdata=X_val, type='prob')

glmnet_best_tune = glmnet_model$results %>% 
  arrange(desc(ROC)) %>% 
  head(1)
glmnet_best_tune




### 2-2. support vector machine with rbf kernel

svm_tune_length = 10
svm_fold_number = 4
svm_train_control = trainControl(method='cv',
                                 number=svm_fold_number,
                                 search='random',
                                 classProbs = TRUE,
                                 summaryFunction=twoClassSummary)

svm_start = Sys.time()

set.seed(student)
svm_model = train(X_train,
                  factor_Y_train,
                  method='svmRadial',
                  trControl=svm_train_control,
                  metric='ROC',
                  tuneLength=svm_tune_length,
                  preProcess = c('center', 'scale'))

svm_train_time = Sys.time() - svm_start
cat('Train time of svm model : '); svm_train_time

svm_model

best_sigma = as.numeric(svm_model$bestTune[1])
best_C = as.numeric(svm_model$bestTune[2])

svm_pred = predict(svm_model, newdata=X_val, type='prob')

svm_best_tune = svm_model$results %>% 
  arrange(desc(ROC)) %>% 
  head(1)
svm_best_tune




### 2-3. random forest

rf_tune_length = 10
rf_fold_number = 4
rf_train_control = trainControl(method='cv',
                                number=rf_fold_number,
                                search='random',
                                classProbs=TRUE,
                                summaryFunction=twoClassSummary)

rf_start = Sys.time()

set.seed(student)
rf_model = train(X_train,
                 factor_Y_train,
                 method='ranger',
                 trControl=rf_train_control,
                 metric='ROC',
                 tuneLength=rf_tune_length)

rf_train_time = Sys.time() - rf_start
cat('Train time of random forest model : '); rf_train_time

rf_model

best_mtry = as.numeric(rf_model$bestTune[1])
best_splitrule = as.character(rf_model$bestTune[2][1, 1])
best_min.node.size = as.numeric(rf_model$bestTune[3])

rf_pred = predict(rf_model, newdata=X_val, type='prob')

rf_best_tune = rf_model$results %>% 
  arrange(desc(ROC)) %>% 
  head(1)
rf_best_tune




### 2-4. xgboost

xgb_tune_length = 10
xgb_fold_number = 4
xgb_train_control = trainControl(method='cv',
                                number=xgb_fold_number,
                                search='random',
                                classProbs=TRUE,
                                summaryFunction=twoClassSummary)

xgb_start = Sys.time()

set.seed(student)
xgb_model = train(X_train,
                  factor_Y_train,
                  method='xgbTree',
                  trControl=xgb_train_control,
                  metric='ROC',
                  tuneLength=xgb_tune_length)

xgb_train_time = Sys.time() - xgb_start
cat('Train time of xgboost model : '); xgb_train_time

xgb_model

best_nrounds = as.numeric(xgb_model$bestTune[1])
best_max_depth = as.numeric(xgb_model$bestTune[2])
best_eta = as.numeric(xgb_model$bestTune[3])
best_gamma = as.numeric(xgb_model$bestTune[4])
best_colsample_bytree = as.numeric(xgb_model$bestTune[5])
best_min_child_weight = as.numeric(xgb_model$bestTune[6])
best_subsample = as.numeric(xgb_model$bestTune[7])

xgb_pred = predict(xgb_model, newdata=X_val, type='prob')

xgb_best_tune = xgb_model$results %>% 
  arrange(desc(ROC)) %>% 
  head(1)
xgb_best_tune




##### 3. evaluate the models

### 3-1. glmnet

glmnet_prediction = prediction(glmnet_pred['yes'], Y_val)
glmnet_performance_auc = performance(glmnet_prediction, 'auc', 'cutoff')
glmnet_performance_acc = performance(glmnet_prediction, 'acc', 'cutoff')
plot(glmnet_performance_acc, main='Cutoff vs Accuracy of glmnet')

glmnet_auc = glmnet_performance_auc@y.values[[1]]
glmnet_acc = max(glmnet_performance_acc@y.values[[1]])
glmnet_cutoff = glmnet_performance_acc@x.values[[1]][which.max(glmnet_performance_acc@y.values[[1]])]

cat(' AUC of glmnet :', glmnet_auc,
    '\n Max Accuracy of glmnet :', glmnet_acc,
    '\n Cutoff of maximum accuracy of glmnet :', glmnet_cutoff)




### 3-2. support vector machine with rbf kernel

svm_prediction = prediction(svm_pred['yes'], Y_val)
svm_performance_auc = performance(svm_prediction, 'auc', 'cutoff')
svm_performance_acc = performance(svm_prediction, 'acc', 'cutoff')
plot(svm_performance_acc, main='Cutoff vs Accuracy of svm')

svm_auc = svm_performance_auc@y.values[[1]]
svm_acc = max(svm_performance_acc@y.values[[1]])
svm_cutoff = svm_performance_acc@x.values[[1]][which.max(svm_performance_acc@y.values[[1]])]

cat(' AUC of svm :', svm_auc,
    '\n Max Accuracy of svm :', svm_acc,
    '\n Cutoff of maximum accuracy of svm :', svm_cutoff)




### 3-3. random forest

rf_prediction = prediction(rf_pred['yes'], Y_val)
rf_performance_auc = performance(rf_prediction, 'auc', 'cutoff')
rf_performance_acc = performance(rf_prediction, 'acc', 'cutoff')
plot(rf_performance_acc, main='Cutoff vs Accuracy of random forest')

rf_auc = rf_performance_auc@y.values[[1]]
rf_acc = max(rf_performance_acc@y.values[[1]])
rf_cutoff = rf_performance_acc@x.values[[1]][which.max(rf_performance_acc@y.values[[1]])]

cat(' AUC of rf :', rf_auc,
    '\n Max Accuracy of rf :', rf_acc,
    '\n Cutoff of maximum accuracy of rf :', rf_cutoff)




### 3-4. xgboost

xgb_prediction = prediction(xgb_pred['yes'], Y_val)
xgb_performance_auc = performance(xgb_prediction, 'auc', 'cutoff')
xgb_performance_acc = performance(xgb_prediction, 'acc', 'cutoff')
plot(xgb_performance_acc, main='Cutoff vs Accuracy of xgboost')

xgb_auc = xgb_performance_auc@y.values[[1]]
xgb_acc = max(xgb_performance_acc@y.values[[1]])
xgb_cutoff = xgb_performance_acc@x.values[[1]][which.max(xgb_performance_acc@y.values[[1]])]

cat(' AUC of xgb :', xgb_auc,
    '\n Max Accuracy of xgb :', xgb_acc,
    '\n Cutoff of maximum accuracy of xgb :', xgb_cutoff)




##### 4. choose models to use or to make ensemble

validation_pred = data.frame(glmnet_pred['yes'],
                             svm_pred['yes'],
                             rf_pred['yes'],
                             xgb_pred['yes'])
colnames(validation_pred) = c('glmnet', 'svm', 'rf', 'xgb')

corr_matrix = cor(validation_pred)
corrplot(corr_matrix, method='number', type='lower')

validation_pred = validation_pred %>% 
  mutate(ensemble = (glmnet + svm + rf + xgb) / 4)

ensemble_prediction = prediction(validation_pred$ensemble, Y_val)
ensemble_performance_auc = performance(ensemble_prediction, 'auc', 'cutoff')
ensemble_performance_acc = performance(ensemble_prediction, 'acc', 'cutoff')
plot(ensemble_performance_acc, main='Cutoff vs Accuracy of ensemble')

ensemble_auc = ensemble_performance_auc@y.values[[1]]
ensemble_acc = max(ensemble_performance_acc@y.values[[1]])
ensemble_cutoff = ensemble_performance_acc@x.values[[1]][which.max(ensemble_performance_acc@y.values[[1]])]

cat(' AUC of ensemble :', ensemble_auc,
    '\n Max Accuracy of ensemble :', ensemble_acc,
    '\n Cutoff of maximum accuracy of ensemble :', ensemble_cutoff)

modeling_result = data.frame(AUC = c(glmnet_auc, svm_auc, rf_auc, xgb_auc, ensemble_auc),
                             Accuracy = c(glmnet_acc, svm_acc, rf_acc, xgb_acc, ensemble_acc),
                             Cutoff = c(glmnet_cutoff, svm_cutoff, rf_cutoff, xgb_cutoff, ensemble_cutoff))
rownames(modeling_result) = c('glmnet', 'svm', 'rf', 'xgb', 'ensemble')
modeling_result




##### 5. confime final model and predict class and posterior probability of Xtest

### 5-1. glmnet

final_glmnet_model = glmnet(x=as.matrix(train_X),
                            y=train_Y,
                            family='binomial',
                            alpha=best_alpha,
                            lambda=best_lambda,
                            standardize = TRUE)

final_glmnet_pred = predict(final_glmnet_model, newx=as.matrix(test_X), type='response')[, 1]

final_glmnet_model




### 5-2. support vector machine with rbf kernel

final_svm_model = ksvm(train_Y ~ .,
                       data=cbind(train_X, train_Y),
                       scaled=TRUE,
                       type='C-svc',
                       kernel='rbfdot',
                       kpar=list(sigma=best_sigma),
                       C=best_C,
                       prob.model=TRUE)

final_svm_pred = predict(final_svm_model, newdata=test_X, type="probabilities")[, 2]

final_svm_model




### 5-3. random forest

final_rf_model = ranger(train_Y ~ .,
                        data=cbind(train_X, train_Y),
                        mtry=best_mtry,
                        splitrule=best_splitrule,
                        min.node.size=best_min.node.size,
                        probability=TRUE)

final_rf_pred_ = predict(final_rf_model, data=test_X, type="response",
                         num.trees=final_rf_model$num.trees)
final_rf_pred = final_rf_pred_$predictions[, 2]

final_rf_model




### 5-4. xgboost

temp_train_Y = ifelse(train_Y == '1', 1, 0)
final_xgb_model = xgboost(data=as.matrix(train_X),
                          label=temp_train_Y,
                          objective='binary:logistic',
                          nrounds=best_nrounds,
                          max_depth=best_max_depth,
                          eta=best_eta,
                          gamma=best_gamma,
                          colsample_bytree=best_colsample_bytree,
                          min_child_weight=best_min_child_weight,
                          subsample=best_subsample,
                          verbose=FALSE)

final_xgb_pred = predict(final_xgb_model, newdata=as.matrix(test_X), type="prob")

final_xgb_model




### 5-5. ensemble

final_ensemble_pred = (final_glmnet_pred + final_svm_pred + final_rf_pred + final_xgb_pred) / 4
final_ensemble_class = ifelse(final_ensemble_pred >= ensemble_cutoff, '1', '0')

final_ensemble_ = data.frame(final_ensemble_class, final_ensemble_pred)
final_ensemble = cbind(rownames(final_ensemble_), final_ensemble_)
colnames(final_ensemble) = c('ID', 'yhat', 'prob')

head(final_ensemble)

write_xlsx(final_ensemble, path='./final_answer.xlsx')



