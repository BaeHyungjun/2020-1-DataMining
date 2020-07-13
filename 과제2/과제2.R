# 20152410 배형준 데이터마이닝 과제2

# Use "Default" data in "ISLR" package. Split it into 50% training set and 50% test set using your CAU ID number.
# 1. Construct (a) a naive Bayes classifier, (b) a classification tree classifier, and (c) a logistic regression classifier.
# 2. For each classifier, make an ROC curve, calculate the AUC, and compare the three classifiers.
# 3. For each classifier, find the optimal cutoff value to maximize the accuracy. Compare the three classifiers.
# 4. For each classifier, find the optimal cutoff value to maximize the F1 score. Compare the three classifiers.
# 5. Write your conclusions and discussion.




###### 1. Construct (a) a naive Bayes classifier, (b) a classification tree classifier, and (c) a logistic regression classifier.

library(ISLR)

data = ISLR::Default

n = dim(data)[1]
train_size = 0.5
student = 20152410

set.seed(student)
train_index = sample(1:n, n*train_size, replace=FALSE)
train = data[train_index, ]
test = data[-train_index, ]




### (a) a naive Bayes classifier

library(e1071)

model_nb = naiveBayes(default ~ ., data=train)
pred_nb = predict(model_nb, newdata=test, type='raw')
model_nb




### (b) a classification tree classifier

library(tree)

model_tree = tree(default ~ ., data=train, split='deviance')
pred_tree = predict(model_tree, newdata=test)
plot(model_tree)
text(model_tree)




### (c) a logistic regression classifier

model_logit = glm(default ~ ., data=train, family='binomial')
pred_logit = predict(model_logit, newdata=test, type='response')
summary(model_logit)




##### 2. For each classifier, make an ROC curve, calculate the AUC, and compare the three classifiers.

library(ROCR)

### ROC curve of 3 classifiers
prediction_nb = prediction(pred_nb[, 'Yes'], test$default)
performance_nb = performance(prediction_nb, 'tpr', 'fpr')
plot(performance_nb, main='ROC curve of naive bayes', colorize=TRUE) # colorize=TRUE의 색은 cutoff를 의미

prediction_tree = prediction(pred_tree[, 'Yes'], test$default)
performance_tree = performance(prediction_tree, 'tpr', 'fpr')
plot(performance_tree, main='ROC curve of classification tree', colorize=TRUE)

prediction_logit = prediction(pred_logit, test$default)
performance_logit = performance(prediction_logit, 'tpr', 'fpr')
plot(performance_logit, main='ROC curve of logistic regression', colorize=TRUE)


### AUC of 3 classifiers
auc_nb = performance(prediction_nb, 'auc')
auc_tree = performance(prediction_tree, 'auc')
auc_logit = performance(prediction_logit, 'auc')

cat('AUC of naive bayes : ', auc_nb@y.values[[1]])
cat('AUC of classification tree : ', auc_tree@y.values[[1]])
cat('AUC of logistic regression : ', auc_logit@y.values[[1]])




##### 3. For each classifier, find the optimal cutoff value to maximize the accuracy. Compare the three classifiers.

acc_nb = performance(prediction_nb, 'acc', 'cutoff')
plot(acc_nb, main='Accuracy of naive bayes for every cutoff')

acc_tree = performance(prediction_tree, 'acc', 'cutoff')
plot(acc_tree, main='Accuracy of classification tree for every cutoff')

acc_logit = performance(prediction_logit, 'acc', 'cutoff')
plot(acc_logit, main='Accuracy of logistic regression for every cutoff')


cat('Optimal cutoff of naive bayes : ', acc_nb@x.values[[1]][which.max(acc_nb@y.values[[1]])])
cat('Maximum accuracy of naive bayes : ', max(acc_nb@y.values[[1]]))

cat('Optimal cutoff of classification tree : ', acc_tree@x.values[[1]][which.max(acc_tree@y.values[[1]])])
cat('Maximum accuracy of classification tree : ', max(acc_tree@y.values[[1]]))

cat('Optimal cutoff of logistic regression : ', acc_logit@x.values[[1]][which.max(acc_logit@y.values[[1]])])
cat('Maximum accuracy of logistic regression : ', max(acc_logit@y.values[[1]]))




##### 4. For each classifier, find the optimal cutoff value to maximize the F1 score. Compare the three classifiers.

f1_nb = performance(prediction_nb, 'f', 'cutoff')
plot(f1_nb, main='F1 score of naive bayes for every cutoff')

f1_tree = performance(prediction_tree, 'f', 'cutoff')
plot(f1_tree, main='F1 score of classification tree for every cutoff')

f1_logit = performance(prediction_logit, 'f', 'cutoff')
plot(f1_logit, main='F1 score of logistic regression for every cutoff')


cat('Optimal cutoff of naive bayes : ', f1_nb@x.values[[1]][which.max(f1_nb@y.values[[1]])])
cat('Maximum F1 score of naive bayes : ', max(f1_nb@y.values[[1]], na.rm=TRUE))

cat('Optimal cutoff of classfication tree : ', f1_tree@x.values[[1]][which.max(f1_tree@y.values[[1]])])
cat('Maximum F1 score of classfication tree : ', max(f1_tree@y.values[[1]], na.rm=TRUE))

cat('Optimal cutoff of logistic regression : ', f1_logit@x.values[[1]][which.max(f1_logit@y.values[[1]])])
cat('Maximum F1 score of logistic regression : ', max(f1_logit@y.values[[1]], na.rm=TRUE))




##### 5. Write your conclusions and discussion.

