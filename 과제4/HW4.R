# 20152410 배형준 데이터마이닝 HW4

#Construct support vector machine classifiers for MNIST_small data.
#Report the best tuning parameters and what kernel works the best.
#Make a confusion matrix and provide your conclusions and discussion.




##### load dataset

mnist_train = read.csv('./MNIST_train_small.csv', header=TRUE)
mnist_test = read.csv('./MNIST_test_small.csv', header=TRUE)

x_train = mnist_train[, 2:785]
y_train = as.factor(mnist_train[, 1])
x_test = mnist_test[, 2:785]
y_test = as.factor(mnist_test[, 1])

student = 20152410




##### construct support vector machine classifier

library(e1071)
library(caret)
library(kernlab)

method_list = c('svmLinear', 'svmPoly', 'svmRadial')
fold_number = 4
tune_length = 20
train_control = trainControl(method='cv',
                             number=fold_number,
                             search='random',
                             verboseIter = TRUE)




### linear svm

set.seed(student)
linear_model = train(x_train,
                     y_train,
                     method=method_list[1],
                     trControl=train_control,
                     metric='accuracy',
                     tuneLength=tune_length)

linear_model

linear_pred = predict(linear_model, newdata=x_test)
linear_table = table(linear_pred, y_test)
linear_cm = confusionMatrix(linear_table, mode='everything')
linear_cm




### polynomial svm

set.seed(student)
ploy_model = train(x_train,
                   y_train,
                   method=method_list[2],
                   trControl=train_control,
                   metric='accuracy',
                   tuneLength=tune_length)

ploy_model

ploy_pred = predict(ploy_model, newdata=x_test)
ploy_table = table(ploy_pred, y_test)
ploy_cm = confusionMatrix(ploy_table, mode='everything')
ploy_cm




### radial svm

set.seed(student)
radial_model = train(x_train,
                     y_train,
                     method=method_list[3],
                     trControl=train_control,
                     metric='accuracy',
                     tuneLength=tune_length)

radial_model

radial_pred = predict(radial_model, newdata=x_test)
radial_table = table(radial_pred, y_test)
radial_cm = confusionMatrix(radial_table, mode='everything')
radial_cm

