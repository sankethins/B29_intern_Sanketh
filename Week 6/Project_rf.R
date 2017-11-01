rm(list = ls(all = TRUE))

#library(doParallel)
# install.packages("caret")
# install.packages("e1071")
# install.packages("Rcpp")
# install.packages("randomForest")
# install.packages("ipred")
# install.packages("C50")

library(doMC)
library(e1071)
library(kernlab)
library(caret)
library(DMwR)
library(glmnet)
library(foreign)
library(dplyr)
library(ggplot2)
library(pROC)
library(e1071)
library(C50)
library(rpart)
library(randomForest)
#library(mlr)
library(rpart)
library(data.table)
library(factoextra)
library(ROCR)
library(vegan)
library(class)
library(corrplot)
library(ROSE)
library(MASS)
#library(ModelMetrics)

setwd("D:/Sanketh/INSOFE/Internship/Day 1/b29-interns-data-sankethins-master/Data")
getwd()

# Reading data
census_data<-read.table("adult1.data", header = F, sep = ",", na.strings = c(""," "," ?","NA",NA))
sum(is.na(census_data))
census_data$V15<-ifelse(census_data$V15 %in% c(" >50K"), 0, 1)
census_data$V15<-as.factor(census_data$V15)
nearZeroVar(census_data[, -15], saveMetrics = TRUE)
census_data1<-census_data

census_data$V5<-NULL
census_data$V11<-NULL
census_data$V12<-NULL
#census_data$V13<-NULL

str(census_data)
summary(census_data)
sum(is.na(census_data))

# Missing value imputation
colMeans(is.na(census_data))
census_data[,which(colMeans(is.na(census_data)) > 0.1)]
census_data<-centralImputation(census_data)
sum(is.na(census_data))

sum(is.na(census_data1))
census_data1<-centralImputation(census_data1)
sum(is.na(census_data))



#registerDoParallel(8)



# Correlation plot and scatter plots
numeric_data<-census_data[,names(census_data) %in% c("V1","V3","V13")]
corrplot(cor(numeric_data), method = "number")
plot(census_data$V2,census_data$V15)



# Outlier analysis
numeric_data1<-numeric_data
replace_outlier_with_missing <- function(x, na.rm = TRUE, ...) {
  qnt <- quantile(x, probs=c(.25, .75), na.rm = na.rm, ...)  # get %iles
  H <- 1.5 * IQR(x, na.rm = na.rm)  # outlier limit threshold
  y <- x
  y[x < (qnt[1] - H)] <- NA  # replace values below lower bounds
  y[x > (qnt[2] + H)] <- NA  # replace values above higher bound
  y  # returns treated variable
}

# numeric_data1 <- as.data.frame (sapply(numeric_data1, replace_outlier_with_missing))
# sum(is.na(numeric_data1$V13)) #9008
# sum(is.na(numeric_data1$V1)) #143
# sum(is.na(numeric_data1$V3)) #992

# Splitting data into test and train

set.seed(555)
train_rows<-sample(x = 1:nrow(census_data), size = 0.7*nrow(census_data))
train_data <- census_data[train_rows, ]
test_data <- census_data[-train_rows, ]

train_data1<-census_data1[train_rows, ]
test_data1 <- census_data1[-train_rows, ]

# Standardization
library(caret)
train_datanew=subset(train_data,select=-c(V15))
test_datanew=subset(test_data,select=-c(V15))

std_method <- preProcess(train_datanew,method = c("center","scale"))

train_datanew <- predict(std_method, train_datanew)
train_datanew<-as.data.frame(train_datanew)

test_datanew <- predict(std_method, test_datanew)
test_datanew<-as.data.frame(test_datanew)

plot(std_method, type = "l")
summary(std_method)

V15<-train_data$V15
train_datanew1<-as.data.frame(cbind(train_datanew,V15))

V15<-test_data$V15
test_datanew1<-as.data.frame(cbind(test_datanew,V15))

###############
library(caret)

train_datanew3=subset(train_data1,select=-c(V15))
test_datanew3=subset(test_data1,select=-c(V15))

std_method <- preProcess(train_datanew,method = c("center","scale"))

train_datanew3 <- predict(std_method, train_datanew3)
train_datanew3<-as.data.frame(train_datanew3)

test_datanew3 <- predict(std_method, test_datanew3)
test_datanew3<-as.data.frame(test_datanew3)

plot(std_method, type = "l")
summary(std_method)

V15<-train_data1$V15
train_datanew3<-as.data.frame(cbind(train_datanew3,V15))

V15<-test_data1$V15
test_datanew3<-as.data.frame(cbind(test_datanew3,V15))

#Handling Class imbalance
table(train_datanew1$V15)
train_datanewROSE<-ROSE(V15~., data = train_datanew1, seed = 1)$data
table(train_datanewROSE$V15)

table(train_datanew1$V15)
train_datanewSMOTE<-SMOTE(V15 ~.,train_datanew1,perc.over =200 ,k=5,perc.under = 150,learner = NULL)
table(train_datanewSMOTE$V15)

# Random Forest
library(randomForest)
model_rf <- randomForest(V15 ~ . , train_datanew1,ntrees=1500)
x<-data.frame(importance(model_rf))
attach(x)
for(i in 1:nrow(x))
{
  if(x[i,1]>25)
  {
    print(rownames(x[i,0], do.NULL = TRUE, prefix = "row"))
  }
}
varImp(model_rf)
varImpPlot(model_rf)


preds_rf <- predict(model_rf, test_datanew1)
preds_rftrain <- predict(model_rf, train_datanew1,type="prob")
preds_rftrain2<-ifelse(preds_rftrain[,1]>preds_rftrain[,2],preds_rftrain[,1],preds_rftrain[,2])
preds_rftest <- data.frame(predict(model_rf, test_datanew1,type="prob"))
preds_rftest2<-ifelse(preds_rftest[,1]>preds_rftest[,2],preds_rftest[,1],preds_rftest[,2])
preds_trainrf<-predict(model_rf, train_datanew1)
confusionMatrix(preds_rf, test_datanew1$V15)
str(preds_rf)
str(test_datanew1$V15)

# Feature selection using random forest
train_datanew2<-train_datanew1[,names(train_datanew1) %in% c("V13","V6","V7","V8","V4","V1","V3","V15")]
test_datanew2<-test_datanew1[,names(train_datanew1) %in% c("V13","V6","V7","V8","V4","V1","V3","V15")]

str(train_datanew2)
summary(train_datanew2)

# V4_education<-train_datanew2[,names(train_datanew2) %in% c("V4")]
# levels(V4_education)
# v4_education<-gsub(" 10th"," Secondary education",v4_education)
# v4_education<-gsub(" 11th"," Secondary education",v4_education)
# v4_education<-gsub(" 12th"," Secondary education",v4_education)
# v4_education<-gsub(" 9th"," Secondary education",v4_education)
# v4_education<-gsub(" 7th-8th"," Secondary education",v4_education)
# v4_education<-gsub(" 1st-4th"," Elementary or primary school",v4_education)
# v4_education<-gsub(" 5th-6th"," Elementary or primary school",v4_education)
# v4_education<-gsub(" Preschool"," Elementary or primary school",v4_education)
# as.data.frame(V4_education)

train_datanew2$V4<-as.character(train_datanew2$V4)
train_datanew2$V4 <- replace(train_datanew2$V4, train_datanew2$V4==" 10th", " Secondary education")
train_datanew2$V4 <- replace(train_datanew2$V4, train_datanew2$V4==" 11th", " Secondary education")
train_datanew2$V4 <- replace(train_datanew2$V4, train_datanew2$V4==" 12th", " Secondary education")
train_datanew2$V4 <- replace(train_datanew2$V4, train_datanew2$V4==" 9th", " Secondary education")
train_datanew2$V4 <- replace(train_datanew2$V4, train_datanew2$V4==" 7th-8th", " Secondary education")
train_datanew2$V4 <- replace(train_datanew2$V4, train_datanew2$V4==" 1st-4th", " Elementary or primary school")
train_datanew2$V4 <- replace(train_datanew2$V4, train_datanew2$V4==" 5th-6th", " Elementary or primary school")
train_datanew2$V4 <- replace(train_datanew2$V4, train_datanew2$V4==" Preschool", " Elementary or primary school")
train_datanew2$V4 <- as.factor(as.character(train_datanew2$V4))
levels(train_datanew2$V4)

test_datanew2$V4<-as.character(test_datanew2$V4)
test_datanew2$V4 <- replace(test_datanew2$V4, test_datanew2$V4==" 10th", " Secondary education")
test_datanew2$V4 <- replace(test_datanew2$V4, test_datanew2$V4==" 11th", " Secondary education")
test_datanew2$V4 <- replace(test_datanew2$V4, test_datanew2$V4==" 12th", " Secondary education")
test_datanew2$V4 <- replace(test_datanew2$V4, test_datanew2$V4==" 9th", " Secondary education")
test_datanew2$V4 <- replace(test_datanew2$V4, test_datanew2$V4==" 7th-8th", " Secondary education")
test_datanew2$V4 <- replace(test_datanew2$V4, test_datanew2$V4==" 1st-4th", " Elementary or primary school")
test_datanew2$V4 <- replace(test_datanew2$V4, test_datanew2$V4==" 5th-6th", " Elementary or primary school")
test_datanew2$V4 <- replace(test_datanew2$V4, test_datanew2$V4==" Preschool", " Elementary or primary school")
test_datanew2$V4 <- as.factor(as.character(test_datanew2$V4))
levels(test_datanew2$V4)


registerDoParallel(8)

#BEST FIT MODEL FOR CROSS VALIDATION
fitControl <- trainControl(method = "cv",number = 5,savePredictions = 'final',classProbs = F)

# Neural Network
cran <- getOption("repos")
cran["dmlc"] <- "https://s3.amazonaws.com/mxnet-r/"
options(repos = cran)
install.packages("mxnet")
install.packages("brew")
library(mxnet)

# neural network on selected features
train.x = data.matrix(train_datanew2[,-8])
train.y = as.numeric(as.character(train_datanew2[,8]))
test.x = data.matrix(test_datanew2[,-8])
test.y = as.numeric(as.character(test_datanew2[,8]))
mx.set.seed(98)
Sys.time() -> start
model_mlp <- mx.mlp(train.x, train.y, hidden_node=c(20), out_node=2,activation="relu", out_activation="softmax",num.round=20, array.batch.size=50, learning.rate=0.05, momentum=0.5,eval.metric=mx.metric.accuracy)
Sys.time() -> end
paste(end - start)

preds = predict(model_mlp, test.x)
preds=t(preds)
pred.label1<-ifelse(preds[,2]>0.65,1,0)
confusionMatrix(as.factor(pred.label1),as.factor(test.y),positive = "0")


# Neural network all variables
train.x = data.matrix(train_datanew1[,-12])
train.y = as.numeric(as.character(train_datanew1[,12]))
test.x = data.matrix(test_datanew1[,-12])
test.y = as.numeric(as.character(test_datanew1[,12]))
mx.set.seed(98)
Sys.time() -> start
model_mlp <- mx.mlp(train.x, train.y, hidden_node=c(20), out_node=2,activation="relu", out_activation="softmax",num.round=20, array.batch.size=50, learning.rate=0.05, momentum=0.5,eval.metric=mx.metric.accuracy)
Sys.time() -> end
paste(end - start)

preds = predict(model_mlp, test.x)
preds=t(preds)
pred.label1<-ifelse(preds[,2]>0.65,1,0)
confusionMatrix(as.factor(pred.label1),as.factor(test.y),positive = "0")

# Neural network all variables after removing class imbalance (ROSE)
train.x = data.matrix(train_datanewROSE[,-12])
train.y = as.numeric(as.character(train_datanewROSE[,12]))
test.x = data.matrix(test_datanew1[,-12])
test.y = as.numeric(as.character(test_datanew1[,12]))
mx.set.seed(98)
Sys.time() -> start
model_mlp <- mx.mlp(train.x, train.y, hidden_node=c(20), out_node=2,activation="relu", out_activation="softmax",num.round=20, array.batch.size=50, learning.rate=0.05, momentum=0.5,eval.metric=mx.metric.accuracy)
Sys.time() -> end
paste(end - start)

preds = predict(model_mlp, test.x)
preds=t(preds)
pred.label1<-ifelse(preds[,2]>0.65,1,0)
preds_ann_en<-as.factor(pred.label1)
confusionMatrix(as.factor(pred.label1),as.factor(test.y),positive = "0")

# Neural network all variables after removing class imbalance (SMOTE)
train.x = data.matrix(train_datanewSMOTE[,-12])
train.y = as.numeric(as.character(train_datanewSMOTE[,12]))
test.x = data.matrix(test_datanew1[,-12])
test.y = as.numeric(as.character(test_datanew1[,12]))
mx.set.seed(98)
Sys.time() -> start
model_mlp <- mx.mlp(train.x, train.y, hidden_node=c(20), out_node=2,activation="relu", out_activation="softmax",num.round=20, array.batch.size=50, learning.rate=0.05, momentum=0.5,eval.metric=mx.metric.accuracy)
Sys.time() -> end
paste(end - start)

preds = predict(model_mlp, test.x)
preds=t(preds)
pred.label1<-ifelse(preds[,2]>0.65,1,0)
confusionMatrix(as.factor(pred.label1),as.factor(test.y),positive = "0")

# Neural network basic
train.x = data.matrix(train_datanew3[,-15])
train.y = as.numeric(as.character(train_datanew3[,15]))
test.x = data.matrix(test_datanew3[,-15])
test.y = as.numeric(as.character(test_datanew3[,15]))
mx.set.seed(98)
Sys.time() -> start
model_mlp <- mx.mlp(train.x, train.y, hidden_node=c(20), out_node=2,activation="relu", out_activation="softmax",num.round=20, array.batch.size=50, learning.rate=0.05, momentum=0.5,eval.metric=mx.metric.accuracy)
Sys.time() -> end
paste(end - start)

preds = predict(model_mlp, test.x)
preds=t(preds)
pred.label1<-ifelse(preds[,2]>0.65,1,0)
confusionMatrix(as.factor(pred.label1),as.factor(test.y),positive = "0")
##############SVM

library(e1071)


######SVM linear on original data
model_svm <- svm(V15 ~ . , train_datanew1, kernel = "linear")
summary(model_svm)
preds_svmtrain<-predict(model_svm, train_datanew1)
preds_svm <- predict(model_svm, test_datanew1)
test_lab<-test_datanew1$V15
confusionMatrix(preds_svm, test_lab,positive = "0")


#####SVM linear on new data
model_svm <- svm(V15 ~ . , train_datanew2, kernel = "linear")
summary(model_svm)
preds_svmtrain<-predict(model_svm, train_datanew2)
preds_svm <- predict(model_svm, test_datanew2)
test_lab<-test_datanew2$V15
confusionMatrix(preds_svm, test_lab,positive = "0")

#####SVM linear on after removing class imbalance - ROSE
model_svm <- svm(V15 ~ . , train_datanewROSE, kernel = "linear")
summary(model_svm)
preds_svmtrain<-predict(model_svm, train_datanewROSE)
preds_svm <- predict(model_svm, test_datanew1)
test_lab<-test_datanew1$V15
confusionMatrix(preds_svm, test_lab,positive = "0")

#####SVM linear on after removing class imbalance - SMOTE
model_svm <- svm(V15 ~ . , train_datanewSMOTE, kernel = "linear")
summary(model_svm)
preds_svmtrain<-predict(model_svm, train_datanewSMOTE)
preds_svm <- predict(model_svm, test_datanew1)
test_lab<-test_datanew1$V15
confusionMatrix(preds_svm, test_lab,positive = "0")

#####SVM linear basic
model_svm <- svm(V15 ~ . , train_datanew3, kernel = "linear")
summary(model_svm)
preds_svmtrain<-predict(model_svm, train_datanew3)
preds_svm <- predict(model_svm, test_datanew3)
test_lab<-test_datanew3$V15
confusionMatrix(preds_svm, test_lab,positive = "0")


## SVM using cv on original data
ctrl <- trainControl(method="cv",number = 3)
rpart.grid <- expand.grid(C=50)
model_svmval<-train(V15~.,data=train_datanew1,method='svmLinear',trControl=ctrl,tuneLength=2,tuneGrid=rpart.grid)
preds_svm_val <- predict(model_svmval, test_datanew1)
confusionMatrix(preds_svm_val, test_datanew1$V15,positive = "0")

## SVM using cv on new data
ctrl <- trainControl(method="cv",number = 3)
rpart.grid <- expand.grid(C=10)
model_svmval<-train(V15~.,data=train_datanew2,method='svmLinear',trControl=ctrl,tuneLength=2,tuneGrid=rpart.grid)
preds_svm_val <- predict(model_svmval, test_datanew2)
confusionMatrix(preds_svm_val, test_datanew2$V15,positive = "0")

## SVM using cv after removing class imbalance - ROSE
ctrl <- trainControl(method="cv",number = 2)
rpart.grid <- expand.grid(C=50)
model_svmval<-train(V15~.,data=train_datanewROSE,method='svmLinear',trControl=ctrl,tuneLength=2,tuneGrid=rpart.grid)
preds_svm_val <- predict(model_svmval, test_datanew1)
confusionMatrix(preds_svm_val, test_datanew1$V15,positive = "0")

## SVM using cv after removing class imbalance - SMOTE
ctrl <- trainControl(method="cv",number = 2)
rpart.grid <- expand.grid(C=50)
model_svmval<-train(V15~.,data=train_datanewSMOTE,method='svmLinear',trControl=ctrl,tuneLength=2,tuneGrid=rpart.grid)
preds_svm_val <- predict(model_svmval, test_datanew1)
confusionMatrix(preds_svm_val, test_datanew1$V15,positive = "0")

######SVM on original data with class weights 4:1
model_svm <- svm(V15 ~ . , train_datanew1, kernel = "linear",class.weights= c("0" = 4, "1" = 1))
summary(model_svm)
preds_svmtrain<-predict(model_svm, train_datanew1)
preds_svm <- predict(model_svm, test_datanew1)
test_lab<-test_datanew1$V15
confusionMatrix(preds_svm, test_lab,positive = "0")


#####SVM on new data with class weights 4:1
model_svm <- svm(V15 ~ . , train_datanew2, kernel = "linear",class.weights= c("0" = 4, "1" = 1))
summary(model_svm)
preds_svmtrain<-predict(model_svm, train_datanew2)
preds_svm <- predict(model_svm, test_datanew2)
test_lab<-test_datanew2$V15
confusionMatrix(preds_svm, test_lab,positive = "0")

#####SVM on basic with class weights 4:1
model_svm <- svm(V15 ~ . , train_datanew3, kernel = "linear",class.weights= c("0" = 4, "1" = 1))
summary(model_svm)
preds_svmtrain<-predict(model_svm, train_datanew3)
preds_svm <- predict(model_svm, test_datanew3)
test_lab<-test_datanew3$V15
confusionMatrix(preds_svm, test_lab,positive = "0")

######SVM on original data with class weights 3:1
model_svm <- svm(V15 ~ . , train_datanew1, kernel = "linear",class.weights= c("0" = 3, "1" = 1),cost=10)
summary(model_svm)
preds_svmtrain<-predict(model_svm, train_datanew1)
preds_svm <- predict(model_svm, test_datanew1)
test_lab<-test_datanew1$V15
confusionMatrix(preds_svm, test_lab,positive = "0")


#####SVM on new data with class weights 3:1
model_svm <- svm(V15 ~ . , train_datanew2, kernel = "linear",class.weights= c("0" = 3, "1" = 1),cost=10)
summary(model_svm)
preds_svmtrain<-predict(model_svm, train_datanew2)
preds_svm <- predict(model_svm, test_datanew2)
test_lab<-test_datanew2$V15
confusionMatrix(preds_svm, test_lab,positive = "0")

#####SVM on basic with class weights 3:1
model_svm <- svm(V15 ~ . , train_datanew3, kernel = "linear",class.weights= c("0" = 3, "1" = 1),cost=10)
summary(model_svm)
preds_svmtrain<-predict(model_svm, train_datanew3)
preds_svm <- predict(model_svm, test_datanew3)
test_lab<-test_datanew3$V15
confusionMatrix(preds_svm, test_lab,positive = "0")

#####SVM on new data with class weights 2:1
model_svm <- svm(V15 ~ . , train_datanew3, kernel = "linear",class.weights= c("0" = 2, "1" = 1),cost=50)
summary(model_svm)
preds_svmtrain<-predict(model_svm, train_datanew3)
preds_svm <- predict(model_svm, test_datanew3)
preds_svm_en<-preds_svm
test_lab<-test_datanew3$V15
confusionMatrix(preds_svm, test_lab,positive = "0")

## SVM tanh on original data 
model_svm_th <- ksvm(V15 ~ . ,train_datanew1, kernel = "tanhdot")
preds_svm_th <- predict(model_svm_th, test_datanew1)
confusionMatrix(preds_svm_th, test_datanew1$V15)
svmtanh_aucplot<-plot.roc(as.numeric(test_datanew1$V15),as.numeric(preds_svm_th),lwd=2,type="b",print.auc=T,col="blue",main="svmtanh")
preds_train_svm_th <- predict(model_svm_th)

## SVM tanh on new data
model_svm_th <- ksvm(V15 ~ . ,train_datanew2, kernel = "tanhdot")
preds_svm_th <- predict(model_svm_th, test_datanew2)
confusionMatrix(preds_svm_th, test_datanew2$V15)
svmtanh_aucplot<-plot.roc(as.numeric(test_datanew2$V15),as.numeric(preds_svm_th),lwd=2,type="b",print.auc=T,col="blue",main="svmtanh")
preds_train_svm_th <- predict(model_svm_th)

## SVM tanh after removing class imbalance - ROSE
model_svm_th <- ksvm(V15 ~ . ,train_datanewROSE, kernel = "tanhdot")
preds_svm_th <- predict(model_svm_th, test_datanew1)
confusionMatrix(preds_svm_th, test_datanew1$V15)
svmtanh_aucplot<-plot.roc(as.numeric(test_datanew1$V15),as.numeric(preds_svm_th),lwd=2,type="b",print.auc=T,col="blue",main="svmtanh")
preds_train_svm_th <- predict(model_svm_th)

## SVM tanh after removing class imbalance - ROSE
model_svm_th <- ksvm(V15 ~ . ,train_datanewSMOTE, kernel = "tanhdot")
preds_svm_th <- predict(model_svm_th, test_datanew1)
confusionMatrix(preds_svm_th, test_datanew1$V15)
svmtanh_aucplot<-plot.roc(as.numeric(test_datanew1$V15),as.numeric(preds_svm_th),lwd=2,type="b",print.auc=T,col="blue",main="svmtanh")
preds_train_svm_th <- predict(model_svm_th)

## SVM tanh basic
model_svm_th <- ksvm(V15 ~ . ,train_datanew3, kernel = "tanhdot")
preds_svm_th <- predict(model_svm_th, test_datanew3)
confusionMatrix(preds_svm_th, test_datanew3$V15)
svmtanh_aucplot<-plot.roc(as.numeric(test_datanew3$V15),as.numeric(preds_svm_th),lwd=2,type="b",print.auc=T,col="blue",main="svmtanh")
preds_train_svm_th <- predict(model_svm_th)

## SVM rbfdot on original data 
cat("\014")  
model_svm_rbf <- ksvm(V15 ~ . ,train_datanew1, kernel = "rbfdot",class.weights= c("0" = 3, "1" = 1),cost=0.1,gamma=0.1)
preds_svm_rbf <- predict(model_svm_rbf, test_datanew1)
confusionMatrix(preds_svm_rbf, test_datanew1$V15)
svmtanh_aucplot<-plot.roc(as.numeric(test_datanew1$V15),as.numeric(preds_svm_rbf),lwd=2,type="b",print.auc=T,col="blue",main="svmrbf")
preds_train_svm_rbf <- predict(model_svm_rbf)

model_svm_rbf <- ksvm(V15 ~ . ,train_datanew1, kernel = "rbfdot",class.weights= c("0" = 3, "1" = 1),cost=0.1,gamma=0.2)
preds_svm_rbf <- predict(model_svm_rbf, test_datanew1)
confusionMatrix(preds_svm_rbf, test_datanew1$V15)
svmtanh_aucplot<-plot.roc(as.numeric(test_datanew1$V15),as.numeric(preds_svm_rbf),lwd=2,type="b",print.auc=T,col="blue",main="svmrbf")
preds_train_svm_rbf <- predict(model_svm_rbf)

model_svm_rbf <- ksvm(V15 ~ . ,train_datanew1, kernel = "rbfdot",class.weights= c("0" = 3, "1" = 1),cost=0.1,gamma=0.3)
preds_svm_rbf <- predict(model_svm_rbf, test_datanew1)
confusionMatrix(preds_svm_rbf, test_datanew1$V15)
svmtanh_aucplot<-plot.roc(as.numeric(test_datanew1$V15),as.numeric(preds_svm_rbf),lwd=2,type="b",print.auc=T,col="blue",main="svmrbf")
preds_train_svm_rbf <- predict(model_svm_rbf)

model_svm_rbf <- ksvm(V15 ~ . ,train_datanew1, kernel = "rbfdot",class.weights= c("0" = 3, "1" = 1),cost=0.1,gamma=0.4)
preds_svm_rbf <- predict(model_svm_rbf, test_datanew1)
confusionMatrix(preds_svm_rbf, test_datanew1$V15)
svmtanh_aucplot<-plot.roc(as.numeric(test_datanew1$V15),as.numeric(preds_svm_rbf),lwd=2,type="b",print.auc=T,col="blue",main="svmrbf")
preds_train_svm_rbf <- predict(model_svm_rbf)

model_svm_rbf <- ksvm(V15 ~ . ,train_datanew1, kernel = "rbfdot",class.weights= c("0" = 3, "1" = 1),cost=0.1,gamma=0.5)
preds_svm_rbf <- predict(model_svm_rbf, test_datanew1)
confusionMatrix(preds_svm_rbf, test_datanew1$V15)
svmtanh_aucplot<-plot.roc(as.numeric(test_datanew1$V15),as.numeric(preds_svm_rbf),lwd=2,type="b",print.auc=T,col="blue",main="svmrbf")
preds_train_svm_rbf <- predict(model_svm_rbf)

model_svm_rbf <- ksvm(V15 ~ . ,train_datanew1, kernel = "rbfdot",class.weights= c("0" = 3, "1" = 1),cost=0.1,gamma=0.6)
preds_svm_rbf <- predict(model_svm_rbf, test_datanew1)
confusionMatrix(preds_svm_rbf, test_datanew1$V15)
svmtanh_aucplot<-plot.roc(as.numeric(test_datanew1$V15),as.numeric(preds_svm_rbf),lwd=2,type="b",print.auc=T,col="blue",main="svmrbf")
preds_train_svm_rbf <- predict(model_svm_rbf)

model_svm_rbf <- ksvm(V15 ~ . ,train_datanew1, kernel = "rbfdot",class.weights= c("0" = 3, "1" = 1),cost=0.1,gamma=0.7)
preds_svm_rbf <- predict(model_svm_rbf, test_datanew1)
confusionMatrix(preds_svm_rbf, test_datanew1$V15)
svmtanh_aucplot<-plot.roc(as.numeric(test_datanew1$V15),as.numeric(preds_svm_rbf),lwd=2,type="b",print.auc=T,col="blue",main="svmrbf")
preds_train_svm_rbf <- predict(model_svm_rbf)

model_svm_rbf <- ksvm(V15 ~ . ,train_datanew1, kernel = "rbfdot",class.weights= c("0" = 3, "1" = 1),cost=0.1,gamma=0.8)
preds_svm_rbf <- predict(model_svm_rbf, test_datanew1)
confusionMatrix(preds_svm_rbf, test_datanew1$V15)
svmtanh_aucplot<-plot.roc(as.numeric(test_datanew1$V15),as.numeric(preds_svm_rbf),lwd=2,type="b",print.auc=T,col="blue",main="svmrbf")
preds_train_svm_rbf <- predict(model_svm_rbf)

model_svm_rbf <- ksvm(V15 ~ . ,train_datanew1, kernel = "rbfdot",class.weights= c("0" = 3, "1" = 1),cost=0.1,gamma=0.9)
preds_svm_rbf <- predict(model_svm_rbf, test_datanew1)
confusionMatrix(preds_svm_rbf, test_datanew1$V15)
svmtanh_aucplot<-plot.roc(as.numeric(test_datanew1$V15),as.numeric(preds_svm_rbf),lwd=2,type="b",print.auc=T,col="blue",main="svmrbf")
preds_train_svm_rbf <- predict(model_svm_rbf)



model_svm_rbf <- ksvm(V15 ~ . ,train_datanew1, kernel = "rbfdot",class.weights= c("0" = 3, "1" = 1),cost=10,gamma=0.1)
preds_svm_rbf <- predict(model_svm_rbf, test_datanew1)
confusionMatrix(preds_svm_rbf, test_datanew1$V15)
svmtanh_aucplot<-plot.roc(as.numeric(test_datanew1$V15),as.numeric(preds_svm_rbf),lwd=2,type="b",print.auc=T,col="blue",main="svmrbf")
preds_train_svm_rbf <- predict(model_svm_rbf)

model_svm_rbf <- ksvm(V15 ~ . ,train_datanew1, kernel = "rbfdot",class.weights= c("0" = 3, "1" = 1),cost=10,gamma=0.2)
preds_svm_rbf <- predict(model_svm_rbf, test_datanew1)
confusionMatrix(preds_svm_rbf, test_datanew1$V15)
svmtanh_aucplot<-plot.roc(as.numeric(test_datanew1$V15),as.numeric(preds_svm_rbf),lwd=2,type="b",print.auc=T,col="blue",main="svmrbf")
preds_train_svm_rbf <- predict(model_svm_rbf)

model_svm_rbf <- ksvm(V15 ~ . ,train_datanew1, kernel = "rbfdot",class.weights= c("0" = 3, "1" = 1),cost=10,gamma=0.3)
preds_svm_rbf <- predict(model_svm_rbf, test_datanew1)
confusionMatrix(preds_svm_rbf, test_datanew1$V15)
svmtanh_aucplot<-plot.roc(as.numeric(test_datanew1$V15),as.numeric(preds_svm_rbf),lwd=2,type="b",print.auc=T,col="blue",main="svmrbf")
preds_train_svm_rbf <- predict(model_svm_rbf)

model_svm_rbf <- ksvm(V15 ~ . ,train_datanew1, kernel = "rbfdot",class.weights= c("0" = 3, "1" = 1),cost=10,gamma=0.4)
preds_svm_rbf <- predict(model_svm_rbf, test_datanew1)
confusionMatrix(preds_svm_rbf, test_datanew1$V15)
svmtanh_aucplot<-plot.roc(as.numeric(test_datanew1$V15),as.numeric(preds_svm_rbf),lwd=2,type="b",print.auc=T,col="blue",main="svmrbf")
preds_train_svm_rbf <- predict(model_svm_rbf)

model_svm_rbf <- ksvm(V15 ~ . ,train_datanew1, kernel = "rbfdot",class.weights= c("0" = 3, "1" = 1),cost=10,gamma=0.5)
preds_svm_rbf <- predict(model_svm_rbf, test_datanew1)
confusionMatrix(preds_svm_rbf, test_datanew1$V15)
svmtanh_aucplot<-plot.roc(as.numeric(test_datanew1$V15),as.numeric(preds_svm_rbf),lwd=2,type="b",print.auc=T,col="blue",main="svmrbf")
preds_train_svm_rbf <- predict(model_svm_rbf)

model_svm_rbf <- ksvm(V15 ~ . ,train_datanew1, kernel = "rbfdot",class.weights= c("0" = 3, "1" = 1),cost=10,gamma=0.6)
preds_svm_rbf <- predict(model_svm_rbf, test_datanew1)
confusionMatrix(preds_svm_rbf, test_datanew1$V15)
svmtanh_aucplot<-plot.roc(as.numeric(test_datanew1$V15),as.numeric(preds_svm_rbf),lwd=2,type="b",print.auc=T,col="blue",main="svmrbf")
preds_train_svm_rbf <- predict(model_svm_rbf)

model_svm_rbf <- ksvm(V15 ~ . ,train_datanew1, kernel = "rbfdot",class.weights= c("0" = 3, "1" = 1),cost=10,gamma=0.7)
preds_svm_rbf <- predict(model_svm_rbf, test_datanew1)
confusionMatrix(preds_svm_rbf, test_datanew1$V15)
svmtanh_aucplot<-plot.roc(as.numeric(test_datanew1$V15),as.numeric(preds_svm_rbf),lwd=2,type="b",print.auc=T,col="blue",main="svmrbf")
preds_train_svm_rbf <- predict(model_svm_rbf)

model_svm_rbf <- ksvm(V15 ~ . ,train_datanew1, kernel = "rbfdot",class.weights= c("0" = 3, "1" = 1),cost=10,gamma=0.8)
preds_svm_rbf <- predict(model_svm_rbf, test_datanew1)
confusionMatrix(preds_svm_rbf, test_datanew1$V15)
svmtanh_aucplot<-plot.roc(as.numeric(test_datanew1$V15),as.numeric(preds_svm_rbf),lwd=2,type="b",print.auc=T,col="blue",main="svmrbf")
preds_train_svm_rbf <- predict(model_svm_rbf)

model_svm_rbf <- ksvm(V15 ~ . ,train_datanew1, kernel = "rbfdot",class.weights= c("0" = 3, "1" = 1),cost=10,gamma=0.9)
preds_svm_rbf <- predict(model_svm_rbf, test_datanew1)
confusionMatrix(preds_svm_rbf, test_datanew1$V15)
svmtanh_aucplot<-plot.roc(as.numeric(test_datanew1$V15),as.numeric(preds_svm_rbf),lwd=2,type="b",print.auc=T,col="blue",main="svmrbf")
preds_train_svm_rbf <- predict(model_svm_rbf)

model_svm_rbf <- ksvm(V15 ~ . ,train_datanew1, kernel = "rbfdot",class.weights= c("0" = 3, "1" = 1),cost=50,gamma=0.1)
preds_svm_rbf <- predict(model_svm_rbf, test_datanew1)
confusionMatrix(preds_svm_rbf, test_datanew1$V15)
svmtanh_aucplot<-plot.roc(as.numeric(test_datanew1$V15),as.numeric(preds_svm_rbf),lwd=2,type="b",print.auc=T,col="blue",main="svmrbf")
preds_train_svm_rbf <- predict(model_svm_rbf)

model_svm_rbf <- ksvm(V15 ~ . ,train_datanew1, kernel = "rbfdot",class.weights= c("0" = 3, "1" = 1),cost=50,gamma=0.2)
preds_svm_rbf <- predict(model_svm_rbf, test_datanew1)
confusionMatrix(preds_svm_rbf, test_datanew1$V15)
svmtanh_aucplot<-plot.roc(as.numeric(test_datanew1$V15),as.numeric(preds_svm_rbf),lwd=2,type="b",print.auc=T,col="blue",main="svmrbf")
preds_train_svm_rbf <- predict(model_svm_rbf)

model_svm_rbf <- ksvm(V15 ~ . ,train_datanew1, kernel = "rbfdot",class.weights= c("0" = 3, "1" = 1),cost=50,gamma=0.3)
preds_svm_rbf <- predict(model_svm_rbf, test_datanew1)
confusionMatrix(preds_svm_rbf, test_datanew1$V15)
svmtanh_aucplot<-plot.roc(as.numeric(test_datanew1$V15),as.numeric(preds_svm_rbf),lwd=2,type="b",print.auc=T,col="blue",main="svmrbf")
preds_train_svm_rbf <- predict(model_svm_rbf)

model_svm_rbf <- ksvm(V15 ~ . ,train_datanew1, kernel = "rbfdot",class.weights= c("0" = 3, "1" = 1),cost=50,gamma=0.4)
preds_svm_rbf <- predict(model_svm_rbf, test_datanew1)
confusionMatrix(preds_svm_rbf, test_datanew1$V15)
svmtanh_aucplot<-plot.roc(as.numeric(test_datanew1$V15),as.numeric(preds_svm_rbf),lwd=2,type="b",print.auc=T,col="blue",main="svmrbf")
preds_train_svm_rbf <- predict(model_svm_rbf)

model_svm_rbf <- ksvm(V15 ~ . ,train_datanew1, kernel = "rbfdot",class.weights= c("0" = 3, "1" = 1),cost=50,gamma=0.5)
preds_svm_rbf <- predict(model_svm_rbf, test_datanew1)
confusionMatrix(preds_svm_rbf, test_datanew1$V15)
svmtanh_aucplot<-plot.roc(as.numeric(test_datanew1$V15),as.numeric(preds_svm_rbf),lwd=2,type="b",print.auc=T,col="blue",main="svmrbf")
preds_train_svm_rbf <- predict(model_svm_rbf)

model_svm_rbf <- ksvm(V15 ~ . ,train_datanew1, kernel = "rbfdot",class.weights= c("0" = 3, "1" = 1),cost=50,gamma=0.6)
preds_svm_rbf <- predict(model_svm_rbf, test_datanew1)
confusionMatrix(preds_svm_rbf, test_datanew1$V15)
svmtanh_aucplot<-plot.roc(as.numeric(test_datanew1$V15),as.numeric(preds_svm_rbf),lwd=2,type="b",print.auc=T,col="blue",main="svmrbf")
preds_train_svm_rbf <- predict(model_svm_rbf)

model_svm_rbf <- ksvm(V15 ~ . ,train_datanew1, kernel = "rbfdot",class.weights= c("0" = 3, "1" = 1),cost=50,gamma=0.7)
preds_svm_rbf <- predict(model_svm_rbf, test_datanew1)
confusionMatrix(preds_svm_rbf, test_datanew1$V15)
svmtanh_aucplot<-plot.roc(as.numeric(test_datanew1$V15),as.numeric(preds_svm_rbf),lwd=2,type="b",print.auc=T,col="blue",main="svmrbf")
preds_train_svm_rbf <- predict(model_svm_rbf)

model_svm_rbf <- ksvm(V15 ~ . ,train_datanew1, kernel = "rbfdot",class.weights= c("0" = 3, "1" = 1),cost=50,gamma=0.8)
preds_svm_rbf <- predict(model_svm_rbf, test_datanew1)
confusionMatrix(preds_svm_rbf, test_datanew1$V15)
svmtanh_aucplot<-plot.roc(as.numeric(test_datanew1$V15),as.numeric(preds_svm_rbf),lwd=2,type="b",print.auc=T,col="blue",main="svmrbf")
preds_train_svm_rbf <- predict(model_svm_rbf)

model_svm_rbf <- ksvm(V15 ~ . ,train_datanew1, kernel = "rbfdot",class.weights= c("0" = 3, "1" = 1),cost=250,gamma=10)
preds_svm_rbf <- predict(model_svm_rbf, test_datanew1)
confusionMatrix(preds_svm_rbf, test_datanew1$V15)
svmtanh_aucplot<-plot.roc(as.numeric(test_datanew1$V15),as.numeric(preds_svm_rbf),lwd=2,type="b",print.auc=T,col="blue",main="svmrbf")
preds_train_svm_rbf <- predict(model_svm_rbf)


## SVM rbfdot on new data
model_svm_th <- ksvm(V15 ~ . ,train_datanew2, kernel = "rbfdot",class.weights= c("0" = 3, "1" = 1),cost=10)
preds_svm_th <- predict(model_svm_th, test_datanew2)
confusionMatrix(preds_svm_th, test_datanew1$V15)
svmtanh_aucplot<-plot.roc(as.numeric(test_datanew2$V15),as.numeric(preds_svm_th),lwd=2,type="b",print.auc=T,col="blue",main="svmrbf")
preds_train_svm_th <- predict(model_svm_th)

## SVM rbfdot after removing class imbalance - ROSE
model_svm_th <- ksvm(V15 ~ . ,train_datanewROSE, kernel = "rbfdot",cost=10)
preds_svm_th <- predict(model_svm_th, test_datanew1)
confusionMatrix(preds_svm_th, test_datanew1$V15)
svmtanh_aucplot<-plot.roc(as.numeric(test_datanew1$V15),as.numeric(preds_svm_th),lwd=2,type="b",print.auc=T,col="blue",main="svmrbf")
preds_train_svm_th <- predict(model_svm_th)

## SVM rbfdot after removing class imbalance - SMOTE
model_svm_th <- ksvm(V15 ~ . ,train_datanewROSE, kernel = "rbfdot",cost=10)
preds_svm_th <- predict(model_svm_th, test_datanew1)
confusionMatrix(preds_svm_th, test_datanew1$V15)
svmtanh_aucplot<-plot.roc(as.numeric(test_datanew1$V15),as.numeric(preds_svm_th),lwd=2,type="b",print.auc=T,col="blue",main="svmrbf")
preds_train_svm_th <- predict(model_svm_th)

## SVM rbfdot basic
model_svm_th <- ksvm(V15 ~ . ,train_datanew3, kernel = "rbfdot",cost=10,class.weights= c("0" = 2.6, "1" = 1))
preds_svm_th <- predict(model_svm_th, test_datanew3)
confusionMatrix(preds_svm_th, test_datanew3$V15)
svmtanh_aucplot<-plot.roc(as.numeric(test_datanew3$V15),as.numeric(preds_svm_th),lwd=2,type="b",print.auc=T,col="blue",main="svmrbf")
preds_train_svm_th <- predict(model_svm_th)

# Logistic regression
auc_scores<-NULL
logistic_model<-glm(V15~., data = train_datanew1, family = binomial)
summary(logistic_model)
probabilities<-predict(logistic_model, type = "response")
predictions <- prediction(probabilities, train_datanew1$V15)
perf <- performance(predictions, measure="tpr", x.measure="fpr")
plot(perf, col=rainbow(10), colorize=T, print.cutoffs.at=seq(0,1,0.05))
perf_auc <- performance(predictions, measure="auc")
auc_logistic <- perf_auc@y.values[[1]]
auc_logistic #0.8898914
probabilities_val<-predict(logistic_model,test_datanew1, type = "response")
predictions_val<-ifelse(probabilities_val > 0.75,1,0)
confusion_matrix_logistic<-confusionMatrix(predictions_val, test_datanew1$V15)
confusion_matrix_logistic

auc_scores<-NULL
logistic_model<-glm(V15~., data = train_datanew2, family = binomial)
summary(logistic_model)
probabilities<-predict(logistic_model, type = "response")
predictions <- prediction(probabilities, train_datanew2$V15)
perf <- performance(predictions, measure="tpr", x.measure="fpr")
plot(perf, col=rainbow(10), colorize=T, print.cutoffs.at=seq(0,1,0.05))
perf_auc <- performance(predictions, measure="auc")
auc_logistic <- perf_auc@y.values[[1]]
auc_logistic #0.8898914
probabilities_val<-predict(logistic_model,test_datanew2, type = "response")
predictions_val<-ifelse(probabilities_val > 0.75,1,0)
confusion_matrix_logistic<-confusionMatrix(predictions_val, test_datanew2$V15)
confusion_matrix_logistic

auc_scores<-NULL
logistic_model<-glm(V15~., data = train_datanewROSE, family = binomial)
summary(logistic_model)
probabilities<-predict(logistic_model, type = "response")
predictions <- prediction(probabilities, train_datanewROSE$V15)
perf <- performance(predictions, measure="tpr", x.measure="fpr")
plot(perf, col=rainbow(10), colorize=T, print.cutoffs.at=seq(0,1,0.05))
perf_auc <- performance(predictions, measure="auc")
auc_logistic <- perf_auc@y.values[[1]]
auc_logistic #0.8898914
probabilities_val<-predict(logistic_model,test_datanew1, type = "response")
predictions_val<-ifelse(probabilities_val > 0.75,1,0)
confusion_matrix_logistic<-confusionMatrix(predictions_val, test_datanew1$V15)
confusion_matrix_logistic

auc_scores<-NULL
logistic_model<-glm(V15~., data = train_datanewSMOTE, family = binomial)
summary(logistic_model)
probabilities<-predict(logistic_model, type = "response")
predictions <- prediction(probabilities, train_datanewSMOTE$V15)
perf <- performance(predictions, measure="tpr", x.measure="fpr")
plot(perf, col=rainbow(10), colorize=T, print.cutoffs.at=seq(0,1,0.05))
perf_auc <- performance(predictions, measure="auc")
auc_logistic <- perf_auc@y.values[[1]]
auc_logistic #0.8898914
probabilities_val<-predict(logistic_model,test_datanew1, type = "response")
predictions_val<-ifelse(probabilities_val > 0.55,1,0)
confusion_matrix_logistic<-confusionMatrix(predictions_val, test_datanew1$V15)
confusion_matrix_logistic

auc_scores<-NULL
logistic_model<-glm(V15~., data = train_datanew3, family = binomial)
summary(logistic_model)
probabilities<-predict(logistic_model, type = "response")
predictions <- prediction(probabilities, train_datanew3$V15)
perf <- performance(predictions, measure="tpr", x.measure="fpr")
plot(perf, col=rainbow(10), colorize=T, print.cutoffs.at=seq(0,1,0.05))
perf_auc <- performance(predictions, measure="auc")
auc_logistic <- perf_auc@y.values[[1]]
auc_logistic #0.8898914
probabilities_val<-predict(logistic_model,test_datanew3, type = "response")
predictions_val<-ifelse(probabilities_val > 0.7,1,0)
preds_logistic<-predictions_val
confusion_matrix_logistic<-confusionMatrix(predictions_val, test_datanew3$V15)
confusion_matrix_logistic

names(getModelInfo())

###########RANDOM FOREST##############

customRF <- list(type = "Classification", library = "randomForest", loop = NULL)
customRF$parameters <- data.frame(parameter = c("mtry", "ntree"), class = rep("numeric", 2), label = c("mtry", "ntree"))
customRF$grid <- function(x, y, len = NULL, search = "grid") {}
customRF$fit <- function(x, y, wts, param, lev, last, weights, classProbs, ...) {
  randomForest(x, y, mtry = param$mtry, ntree=param$ntree, ...)
}
customRF$predict <- function(modelFit, newdata, preProc = NULL, submodels = NULL)
  predict(modelFit, newdata)
customRF$prob <- function(modelFit, newdata, preProc = NULL, submodels = NULL)
  predict(modelFit, newdata, type = "prob")
customRF$sort <- function(x) x[order(x[,1]),]
customRF$levels <- function(x) x$classes


# train model
model_rf <- randomForest(V15 ~ . , train_datanew1,ntrees=1500,mtry=3)
preds_rf <- predict(model_rf, test_datanew1)
confusionMatrix(preds_rf, test_datanew1$V15)

model_rf <- randomForest(V15 ~ . , train_datanew1,ntrees=2000,mtry=3)
preds_rf <- predict(model_rf, test_datanew1)
confusionMatrix(preds_rf, test_datanew1$V15)

model_rf <- randomForest(V15 ~ . , train_datanew1,ntrees=3000,mtry=3)
preds_rf <- predict(model_rf, test_datanew1)
confusionMatrix(preds_rf, test_datanew1$V15)
plot(model_rf)

model_rf <- randomForest(V15 ~ . , train_datanew1,ntrees=1500,mtry=2)
preds_rf <- predict(model_rf, test_datanew1)
confusionMatrix(preds_rf, test_datanew1$V15)

model_rf <- randomForest(V15 ~ . , train_datanew1,ntrees=2000,mtry=2)
preds_rf <- predict(model_rf, test_datanew1)
confusionMatrix(preds_rf, test_datanew1$V15)

model_rf <- randomForest(V15 ~ . , train_datanew1,ntrees=3000,mtry=2)
preds_rf <- predict(model_rf, test_datanew1)
confusionMatrix(preds_rf, test_datanew1$V15)

model_rf <- randomForest(V15 ~ . , train_datanew1,ntrees=1500,mtry=4)
preds_rf <- predict(model_rf, test_datanew1)
confusionMatrix(preds_rf, test_datanew1$V15)

model_rf <- randomForest(V15 ~ . , train_datanew1,ntrees=2000,mtry=4)
preds_rf <- predict(model_rf, test_datanew1)
confusionMatrix(preds_rf, test_datanew1$V15)

model_rf <- randomForest(V15 ~ . , train_datanew1,ntrees=3000,mtry=4)
preds_rf <- predict(model_rf, test_datanew1)
confusionMatrix(preds_rf, test_datanew1$V15)

model_rf <- randomForest(V15 ~ . , train_datanewROSE,ntrees=2000,mtry=2)
preds_rf <- predict(model_rf, test_datanew1)
confusionMatrix(preds_rf, test_datanew1$V15)

model_rf <- randomForest(V15 ~ . , train_datanewSMOTE,ntrees=2000,mtry=2)
preds_rf <- predict(model_rf, test_datanew1)
confusionMatrix(preds_rf, test_datanew1$V15)

model_rf <- randomForest(V15 ~ . , train_datanew3,ntrees=2000,mtry=2)
preds_rf <- predict(model_rf, test_datanew3)
confusionMatrix(preds_rf, test_datanew3$V15)


###############KNN#################
model_knncross<-train(V15~.,data=train_datanew1,method='knn',  tuneGrid=expand.grid(.k=1:9),trControl=fitControl,tuneLength=3)
plot(model_knncross)

model_knn <- knn3(V15 ~ . , train_datanew1, k =7)
preds_train_knn<-predict(model_knn,train_datanew1)
preds_k <- predict(model_knn, test_datanew1)
preds_k2<-ifelse(preds_k[, 1] > preds_k[, 2], preds_k[, 1], preds_k[, 2])
preds_knn <- ifelse(preds_k[, 1] > preds_k[, 2], 0, 1)
confusionMatrix(preds_knn, test_datanew1$V15)

model_knn <- knn3(V15 ~ . , train_datanew2, k =7)
preds_train_knn<-predict(model_knn,train_datanew2)
preds_k <- predict(model_knn, test_datanew2)
preds_k2<-ifelse(preds_k[, 1] > preds_k[, 2], preds_k[, 1], preds_k[, 2])
preds_knn <- ifelse(preds_k[, 1] > preds_k[, 2], 0, 1)
confusionMatrix(preds_knn, test_datanew2$V15)

model_knn <- knn3(V15 ~ . , train_datanewROSE, k =7)
preds_train_knn<-predict(model_knn,train_datanewROSE)
preds_k <- predict(model_knn, test_datanew1)
preds_k2<-ifelse(preds_k[, 1] > preds_k[, 2], preds_k[, 1], preds_k[, 2])
preds_knn <- ifelse(preds_k[, 1] > preds_k[, 2], 0, 1)
confusionMatrix(preds_knn, test_datanew1$V15)

model_knn <- knn3(V15 ~ . , train_datanewSMOTE, k =7)
preds_train_knn<-predict(model_knn,train_datanewSMOTE)
preds_k <- predict(model_knn, test_datanew1)
preds_k2<-ifelse(preds_k[, 1] > preds_k[, 2], preds_k[, 1], preds_k[, 2])
preds_knn <- ifelse(preds_k[, 1] > preds_k[, 2], 0, 1)
confusionMatrix(preds_knn, test_datanew1$V15)

model_knn <- knn3(V15 ~ . , train_datanew3, k =7)
preds_train_knn<-predict(model_knn,train_datanew3)
preds_k <- predict(model_knn, test_datanew3)
preds_k2<-ifelse(preds_k[, 1] > preds_k[, 2], preds_k[, 1], preds_k[, 2])
preds_knn <- ifelse(preds_k[, 1] > preds_k[, 2], 0, 1)
confusionMatrix(preds_knn, test_datanew3$V15)


#####Decision tree cart

ctrl <- trainControl(method="repeatedcv",repeats = 10)
rpart.grid <- expand.grid(.cp=seq(0.01,.2,.01))
model_rpart<-train(V15~.,data=train_datanew1,method='rpart',trControl=ctrl,tuneLength=3,tuneGrid=rpart.grid)
model_rpart
preds_dt <- predict(model_rpart, test_datanew1)
confusionMatrix(preds_dt, test_datanew1$V15)

ctrl <- trainControl(method="cv",number = 10)
rpart.grid <- expand.grid(.cp=0.001)
model_rpart<-train(V15~.,data=train_datanew1,method='rpart',trControl=ctrl,tuneLength=3,tuneGrid=rpart.grid)
model_rpart
preds_dt <- predict(model_rpart, test_datanew1)

cart_model<-rpart(V15~.,train_datanew1)
predictions_cart<-predict(cart_model,test_datanew1,type="class")
confusionMatrix(predictions_cart,test_datanew1$V15)
confusionMatrix(preds_dt, test_datanew1$V15)

ctrl <- trainControl(method="repeatedcv",repeats = 10)
rpart.grid <- expand.grid(.cp=seq(0.01,.2,.01))
model_rpart<-train(V15~.,data=train_datanew2,method='rpart',trControl=ctrl,tuneLength=3,tuneGrid=rpart.grid)
model_rpart
preds_dt <- predict(model_rpart, test_datanew2)
confusionMatrix(preds_dt, test_datanew2$V15)

ctrl <- trainControl(method="cv",number = 10)
rpart.grid <- expand.grid(.cp=0.001)
model_rpart<-train(V15~.,data=train_datanew2,method='rpart',trControl=ctrl,tuneLength=3,tuneGrid=rpart.grid)
model_rpart
preds_dt <- predict(model_rpart, test_datanew1)

cart_model<-rpart(V15~.,train_datanew2)
predictions_cart<-predict(cart_model,test_datanew2,type="class")
confusionMatrix(predictions_cart,test_datanew2$V15)
confusionMatrix(preds_dt, test_datanew2$V15)

ctrl <- trainControl(method="repeatedcv",repeats = 10)
rpart.grid <- expand.grid(.cp=seq(0.01,.2,.01))
model_rpart<-train(V15~.,data=train_datanewROSE,method='rpart',trControl=ctrl,tuneLength=3,tuneGrid=rpart.grid)
model_rpart
preds_dt <- predict(model_rpart, test_datanew1)
confusionMatrix(preds_dt, test_datanew1$V15)

ctrl <- trainControl(method="cv",number = 10)
rpart.grid <- expand.grid(.cp=0.001)
model_rpart<-train(V15~.,data=train_datanewROSE,method='rpart',trControl=ctrl,tuneLength=3,tuneGrid=rpart.grid)
model_rpart
preds_dt <- predict(model_rpart, test_datanew1)
confusionMatrix(preds_dt, test_datanew1$V15)


cart_model<-rpart(V15~.,train_datanewROSE)
predictions_cart<-predict(cart_model,test_datanew1,type="class")
confusionMatrix(predictions_cart,test_datanew1$V15)

cart_model<-rpart(V15~.,train_datanewSMOTE)
predictions_cart<-predict(cart_model,test_datanew1,type="class")
confusionMatrix(predictions_cart,test_datanew1$V15)

cart_model<-rpart(V15~.,train_datanew3)
predictions_cart<-predict(cart_model,test_datanew3,type="class")
preds_dt_en<-predictions_cart
confusionMatrix(predictions_cart,test_datanew3$V15)

##############GBM################
train_datanew1$V15 <- as.numeric(as.character(train_datanew1$V15))
str(train_datanew1)
test_datanew1$V15 <- as.numeric(as.charater(test_datanew1$V15))
library(gbm)
# gbmGrid <-  expand.grid(interaction.depth = c(1, 3, 6, 9, 10),
#                         n.trees = 1500,
#                         shrinkage = seq(.0005, .05,.005),
#                         n.minobsinnode = 10)
# model_gbm1<-train(V15~.,data=train_datanew1,method='gbm',  tuneGrid=gbmGrid,trControl=fitControl)
# model_gbm1
model_gbm <-gbm(V15 ~ . , cv.folds = 20, interaction.depth = 2,shrinkage =0.005,distribution='bernoulli',data=train_datanew1, n.trees = 1500)
model_gbm2 <-gbm(V15 ~ . , cv.folds = 20, interaction.depth = 10,shrinkage =0.0005,distribution='bernoulli',data=train_datanew1, n.trees = 2500)
preds_train_g2 <- predict(model_gbm2, train_datanew1)
preds_test_g2 <- predict(model_gbm2, test_datanew1)

#preds_gbm <- ifelse(preds_test_g >= cutoff_gbm, 1, 0)

plot(model_gbm1)
gbm.perf(model_gbm)
gbm.perf(model_gbm2)

preds_g <- predict(model_gbm, type = 'response')
preds_g2 <- predict(model_gbm2,type = 'response')
confusionMatrix(preds_g1, test_datanew1$V15)

#install.packages("pROC")
library(pROC)
#install.packages("caret")
library(caret)
gbm_roc <- roc(train_datanew1$V15, preds_g)
gbm_roc1 <- roc(train_datanew1$V15, preds_g2)

cutoff_gbm <- coords(gbm_roc, "best", ret = "threshold")
cutoff_gbm1 <- coords(gbm_roc1, "best", ret = "threshold")

preds_train_gbm <- ifelse(preds_g >= cutoff_gbm, 1, 0)
preds_train_gbm1 <- ifelse(preds_g2 >= cutoff_gbm1, 1, 0)

preds_test_g <- predict(model_gbm, test_datanew1, type = 'response')
preds_test_g1 <- predict(model_gbm2, test_datanew1, type = 'response')
preds_gbm <- ifelse(preds_test_g >= cutoff_gbm, 1, 0)
preds_gbm1 <- ifelse(preds_test_g1 >= cutoff_gbm1, 1, 0)
preds_gbm_en<-preds_gbm1
preds_test_g 

confusionMatrix(preds_gbm, test_datanew1$V15)
confusionMatrix(preds_gbm1, test_datanew1$V15)

train_datanew1$V15 <- as.factor(as.character(train_datanew1$V15))
test_datanew1$V15 <- as.factor(as.character(test_datanew1$V15))

####GBM on data after removing class imbalance
train_datanew1$V15 <- as.numeric(as.character(train_datanewROSE$V15))
str(train_datanewROSE)
test_datanew1$V15 <- as.numeric(as.charater(test_datanew1$V15))
library(gbm)
# gbmGrid <-  expand.grid(interaction.depth = c(1, 3, 6, 9, 10),
#                         n.trees = 1500,
#                         shrinkage = seq(.0005, .05,.005),
#                         n.minobsinnode = 10)
# model_gbm1<-train(V15~.,data=train_datanew1,method='gbm',  tuneGrid=gbmGrid,trControl=fitControl)
# model_gbm1
model_gbm <-gbm(V15 ~ . , cv.folds = 20, interaction.depth = 2,shrinkage =0.005,distribution='bernoulli',data=train_datanewROSE, n.trees = 1500)
model_gbm2 <-gbm(V15 ~ . , cv.folds = 20, interaction.depth = 10,shrinkage =0.0005,distribution='bernoulli',data=train_datanewROSE, n.trees = 2500)
preds_train_g2 <- predict(model_gbm2, train_datanewROSE)
preds_test_g2 <- predict(model_gbm2, test_datanew1)

#preds_gbm <- ifelse(preds_test_g >= cutoff_gbm, 1, 0)

plot(model_gbm1)
gbm.perf(model_gbm)
gbm.perf(model_gbm2)

preds_g <- predict(model_gbm, type = 'response')
preds_g2 <- predict(model_gbm2,type = 'response')
confusionMatrix(preds_g1, test_datanew1$V15)


#install.packages("pROC")
library(pROC)
#install.packages("caret")
library(caret)
gbm_roc <- roc(train_datanewROSE$V15, preds_g)
gbm_roc1 <- roc(train_datanewROSE$V15, preds_g2)

cutoff_gbm <- coords(gbm_roc, "best", ret = "threshold")
cutoff_gbm1 <- coords(gbm_roc1, "best", ret = "threshold")

preds_train_gbm <- ifelse(preds_g >= cutoff_gbm, 1, 0)
preds_train_gbm1 <- ifelse(preds_g2 >= cutoff_gbm1, 1, 0)

preds_test_g <- predict(model_gbm, test_datanew1, type = 'response')
preds_test_g1 <- predict(model_gbm2, test_datanew1, type = 'response')
preds_gbm <- ifelse(preds_test_g >= cutoff_gbm, 1, 0)
preds_gbm1 <- ifelse(preds_test_g1 >= cutoff_gbm1, 1, 0)

preds_test_g 

confusionMatrix(preds_gbm, test_datanew1$V15)
confusionMatrix(preds_gbm1, test_datanew1$V15)

train_datanew1$V15 <- as.factor(as.character(train_datanewROSE$V15))
test_datanew1$V15 <- as.factor(as.character(test_datanew1$V15))

############GBM on new data
train_datanew2$V15 <- as.numeric(as.character(train_datanew2$V15))
str(train_datanew1)
test_datanew2$V15 <- as.numeric(as.charater(test_datanew2$V15))
library(gbm)
# gbmGrid <-  expand.grid(interaction.depth = c(1, 3, 6, 9, 10),
#                         n.trees = 1500,
#                         shrinkage = seq(.0005, .05,.005),
#                         n.minobsinnode = 10)
# model_gbm1<-train(V15~.,data=train_datanew1,method='gbm',  tuneGrid=gbmGrid,trControl=fitControl)
# model_gbm1
model_gbm <-gbm(V15 ~ . , cv.folds = 20, interaction.depth = 2,shrinkage =0.005,distribution='bernoulli',data=train_datanew2, n.trees = 1500)
model_gbm2 <-gbm(V15 ~ . , cv.folds = 20, interaction.depth = 10,shrinkage =0.0005,distribution='bernoulli',data=train_datanew2, n.trees = 2500)
preds_train_g2 <- predict(model_gbm2, train_datanew2)
preds_test_g2 <- predict(model_gbm2, test_datanew2)

#preds_gbm <- ifelse(preds_test_g >= cutoff_gbm, 1, 0)

plot(model_gbm1)
gbm.perf(model_gbm)
gbm.perf(model_gbm2)

preds_g <- predict(model_gbm, type = 'response')
preds_g2 <- predict(model_gbm2,type = 'response')
confusionMatrix(preds_g1, test_datanew2$V15)

#install.packages("pROC")
library(pROC)
#install.packages("caret")
library(caret)
gbm_roc <- roc(train_datanew2$V15, preds_g)
gbm_roc1 <- roc(train_datanew2$V15, preds_g2)

cutoff_gbm <- coords(gbm_roc, "best", ret = "threshold")
cutoff_gbm1 <- coords(gbm_roc1, "best", ret = "threshold")

preds_train_gbm <- ifelse(preds_g >= cutoff_gbm, 1, 0)
preds_train_gbm1 <- ifelse(preds_g2 >= cutoff_gbm1, 1, 0)

preds_test_g <- predict(model_gbm, test_datanew2, type = 'response')
preds_test_g1 <- predict(model_gbm2, test_datanew2, type = 'response')
preds_gbm <- ifelse(preds_test_g >= cutoff_gbm, 1, 0)
preds_gbm1 <- ifelse(preds_test_g1 >= cutoff_gbm1, 1, 0)

preds_test_g 

confusionMatrix(preds_gbm, test_datanew2$V15)
confusionMatrix(preds_gbm1, test_datanew2$V15)

train_datanew1$V15 <- as.factor(as.character(train_datanew2$V15))
test_datanew1$V15 <- as.factor(as.character(test_datanew2$V15))

##########GBM after smote

train_datanew1$V15 <- as.numeric(as.character(train_datanewSMOTE$V15))
str(train_datanewROSE)
test_datanew1$V15 <- as.numeric(as.charater(test_datanew1$V15))
library(gbm)
# gbmGrid <-  expand.grid(interaction.depth = c(1, 3, 6, 9, 10),
#                         n.trees = 1500,
#                         shrinkage = seq(.0005, .05,.005),
#                         n.minobsinnode = 10)
# model_gbm1<-train(V15~.,data=train_datanew1,method='gbm',  tuneGrid=gbmGrid,trControl=fitControl)
# model_gbm1
model_gbm <-gbm(V15 ~ . , cv.folds = 20, interaction.depth = 2,shrinkage =0.005,distribution='bernoulli',data=train_datanewSMOTE, n.trees = 1500)
model_gbm2 <-gbm(V15 ~ . , cv.folds = 20, interaction.depth = 10,shrinkage =0.0005,distribution='bernoulli',data=train_datanewSMOTE, n.trees = 2500)
preds_train_g2 <- predict(model_gbm2, train_datanewROSE)
preds_test_g2 <- predict(model_gbm2, test_datanew1)

#preds_gbm <- ifelse(preds_test_g >= cutoff_gbm, 1, 0)

plot(model_gbm1)
gbm.perf(model_gbm)
gbm.perf(model_gbm2)

preds_g <- predict(model_gbm, type = 'response')
preds_g2 <- predict(model_gbm2,type = 'response')
confusionMatrix(preds_g1, test_datanew1$V15)


#install.packages("pROC")
library(pROC)
#install.packages("caret")
library(caret)
gbm_roc <- roc(train_datanewSMOTE$V15, preds_g)
gbm_roc1 <- roc(train_datanewSMOTE$V15, preds_g2)

cutoff_gbm <- coords(gbm_roc, "best", ret = "threshold")
cutoff_gbm1 <- coords(gbm_roc1, "best", ret = "threshold")

preds_train_gbm <- ifelse(preds_g >= cutoff_gbm, 1, 0)
preds_train_gbm1 <- ifelse(preds_g2 >= cutoff_gbm1, 1, 0)

preds_test_g <- predict(model_gbm, test_datanew1, type = 'response')
preds_test_g1 <- predict(model_gbm2, test_datanew1, type = 'response')
preds_gbm <- ifelse(preds_test_g >= cutoff_gbm, 1, 0)
preds_gbm1 <- ifelse(preds_test_g1 >= cutoff_gbm1, 1, 0)

preds_test_g 

confusionMatrix(preds_gbm, test_datanew1$V15)
confusionMatrix(preds_gbm1, test_datanew1$V15)

train_datanew1$V15 <- as.factor(as.character(train_datanewSMOTE$V15))
test_datanew1$V15 <- as.factor(as.character(test_datanew1$V15))

###GBM basic

set.seed (3232)
trCtrl = trainControl (method = "cv", number = 10,tunelength = 3)
gbmGrid <-  expand.grid(interaction.depth = c(1, 5, 9), 
                        n.trees = c(1000,2000,3000), 
                        shrinkage = 0.1,
                        n.minobsinnode = 20)
boostFit = train (V15 ~ ., method = "gbm", data = train_datanew3, verbose = FALSE)

preds_gbm <- predict(boostFit, test_datanew3)
confusionMatrix(preds_gbm, test_datanew3$V15,positive = "0")

train_datanew1$V15 <- as.numeric(as.character(train_datanew3$V15))
str(train_datanew1)
test_datanew1$V15 <- as.numeric(as.charater(test_datanew3$V15))
library(gbm)
# gbmGrid <-  expand.grid(interaction.depth = c(1, 3, 6, 9, 10),
#                         n.trees = 1500,
#                         shrinkage = seq(.0005, .05,.005),
#                         n.minobsinnode = 10)
# model_gbm1<-train(V15~.,data=train_datanew1,method='gbm',  tuneGrid=gbmGrid,trControl=fitControl)
# model_gbm1
model_gbm <-gbm(V15 ~ . , cv.folds = 20, interaction.depth = 2,shrinkage =0.005,distribution='bernoulli',data=train_datanew3, n.trees = 1500)
model_gbm2 <-gbm(V15 ~ . , cv.folds = 20, interaction.depth = 10,shrinkage =0.0005,distribution='bernoulli',data=train_datanew3, n.trees = 2500)
preds_train_g2 <- predict(model_gbm2, train_datanew1)
preds_test_g2 <- predict(model_gbm2, test_datanew1)

#preds_gbm <- ifelse(preds_test_g >= cutoff_gbm, 1, 0)

plot(model_gbm1)
gbm.perf(model_gbm)
gbm.perf(model_gbm2)

preds_g <- predict(model_gbm, type = 'response')
preds_g2 <- predict(model_gbm2,type = 'response')
confusionMatrix(preds_g1, test_datanew3$V15)

#install.packages("pROC")
library(pROC)
#install.packages("caret")
library(caret)
gbm_roc <- roc(train_datanew3$V15, preds_g)
gbm_roc1 <- roc(train_datanew3$V15, preds_g2)

cutoff_gbm <- coords(gbm_roc, "best", ret = "threshold")
cutoff_gbm1 <- coords(gbm_roc1, "best", ret = "threshold")

preds_train_gbm <- ifelse(preds_g >= cutoff_gbm, 1, 0)
preds_train_gbm1 <- ifelse(preds_g2 >= cutoff_gbm1, 1, 0)

preds_test_g <- predict(model_gbm, test_datanew3, type = 'response')
preds_test_g1 <- predict(model_gbm2, test_datanew3, type = 'response')
preds_gbm <- ifelse(preds_test_g >= cutoff_gbm, 1, 0)
preds_gbm1 <- ifelse(preds_test_g1 >= cutoff_gbm1, 1, 0)

preds_test_g 

confusionMatrix(preds_gbm, test_datanew3$V15)
confusionMatrix(preds_gbm1, test_datanew3$V15)

train_datanew1$V15 <- as.factor(as.character(train_datanew3$V15))
test_datanew1$V15 <- as.factor(as.character(test_datanew3$V15))

#### BAGGING

library(ipred)
set.seed(1234)

model_tree_bag <- bagging(V15 ~ . , data=train_datanew3,nbagg=25, control = rpart.control(cp = .0005, xval = 10))
preds_tree_bag <- predict(model_tree_bag, test_datanew3)
confusionMatrix(preds_tree_bag,test_datanew3$V15)
preds_train_tree_bag <- predict(model_tree_bag)

model_tree_bag <- bagging(V15 ~ . , data=train_datanewROSE,nbagg=25, control = rpart.control(cp = .0005, xval = 10))
preds_tree_bag <- predict(model_tree_bag, test_datanew1)
confusionMatrix(preds_tree_bag,test_datanew1$V15)
preds_bag_en<-preds_tree_bag
preds_train_tree_bag <- predict(model_tree_bag)

model_tree_bag <- bagging(V15 ~ . , data=train_datanewSMOTE,nbagg=25, control = rpart.control(cp = .0005, xval = 10))
preds_tree_bag <- predict(model_tree_bag, test_datanew1)
confusionMatrix(preds_tree_bag,test_datanew1$V15)
preds_train_tree_bag <- predict(model_tree_bag)

model_tree_bag <- bagging(V15 ~ . , data=train_datanew2,nbagg=25, control = rpart.control(cp = .0005, xval = 10))
preds_tree_bag <- predict(model_tree_bag, test_datanew2)
confusionMatrix(preds_tree_bag,test_datanew2$V15)
preds_train_tree_bag <- predict(model_tree_bag)

###### Taking majority voting of predictions ######

test_datanew_majority<-ifelse(preds_logistic=='1' & preds_svm_en=='1','1',ifelse(preds_logistic=='1' & preds_gbm_en=='1','1','0'))
confusionMatrix(test_datanew_majority,test_datanew1$V15)

test_datanew_majority1<-ifelse(preds_ann_en=='1' & preds_bag_en=='1','1',ifelse(preds_ann_en=='1' & preds_dt_en=='1','1','0'))
confusionMatrix(test_datanew_majority1,test_datanew1$V15)

test_datanew_majority2<-ifelse(preds_logistic=='1' & test_datanew_majority=='1','1',ifelse(preds_logistic=='1' & test_datanew_majority1=='1','1','0'))
confusionMatrix(test_datanew_majority2,test_datanew1$V15)

####### Stacking with GBM as top layer

train_preds_df <- data.frame(svm = preds_svmtrain,rf =preds_train_rf,knn = preds_train_knn,tree = preds_train_rpart, tree_bag = preds_train_tree_bag,gbm = preds_train_g2, V15 = train_datanew1$V15)
test_preds_df<- data.frame(svm=preds_svm,rf = preds_rfvaltest,knn = preds_k,tree =preds_train_dec1, tree_bag = preds_tree_bag,gbm=preds_test_g2,V15 = test_datanew1$V15)
stack_df <- rbind(train_preds_df)
stack_df$V15 <- as.factor(stack_df$V15)
numeric_st_df <- sapply(stack_df[, !(names(stack_df) %in% "V15")],function(x) as.numeric(as.character(x)))
pca_stack <- prcomp(numeric_st_df, scale = F)
predicted_stack <- as.data.frame(predict(pca_stack, numeric_st_df))[1:7]
stacked_df <- data.frame(predicted_stack, V15 = stack_df[, (names(stack_df) %in% "V15")])
stacked_model<-gbm(V15 ~ . , cv.folds = 20, interaction.depth = 10,shrinkage =0.0205,distribution='bernoulli',data=stacked_df,n.trees = 1500)
test_preds_df$V15 <- as.factor(test_preds_df$V15)
numeric_st_df_test <- sapply(test_preds_df[, !(names(test_preds_df) %in% "V15")],function(x) as.numeric(as.character(x)))
predicted_stack_test <- as.data.frame(predict(pca_stack, numeric_st_df_test))[1:7]
stacked_df_test <- data.frame(predicted_stack_test, V15 = test_preds_df[, (names(test_preds_df) %in% "V15")])
preds_st_test <-  predict(stacked_model, stacked_df_test)
confusionMatrix(preds_st_test, test_datanew1$V15)

####### Stacking with GLM at top layer
train_preds_df <- data.frame(svm = preds_svmtrain,rf =preds_train_rf,knn = preds_train_knn,tree = preds_train_rpart, tree_bag = preds_train_tree_bag,gbm = preds_train_g2, V15 = train_datanew1$V15)
# Getting all the predictions from the validation data into a dataframe.
test_preds_df<- data.frame(svm=preds_svm,rf = preds_rfvaltest,knn = preds_k,tree =preds_train_dec1, tree_bag = preds_tree_bag,gbm=preds_test_g2,V15 = test_datanew1$V15)
stack_df <- rbind(train_preds_df)

stack_df$V15 <- as.factor(stack_df$V15)
numeric_st_df <- sapply(stack_df[, !(names(stack_df) %in% "V15")],function(x) as.numeric(as.character(x)))
pca_stack <- prcomp(numeric_st_df, scale = F)
predicted_stack <- as.data.frame(predict(pca_stack, numeric_st_df))[1:7]
stacked_df <- data.frame(predicted_stack, V15 = stack_df[, (names(stack_df) %in% "V15")])
stacked_model <- glm(V15~., data = stacked_df, family = binomial,control = list(maxit=100))
test_preds_df$V15 <- as.factor(test_preds_df$V15)
numeric_st_df_test <- sapply(test_preds_df[, !(names(test_preds_df) %in% "V15")],function(x) as.numeric(as.character(x)))
predicted_stack_test <- as.data.frame(predict(pca_stack, numeric_st_df_test))[1:7]
stacked_df_test <- data.frame(predicted_stack_test, V15 = test_preds_df[, (names(test_preds_df) %in% "V15")])
preds_st_test <-  predict(stacked_model, stacked_df_test,type="response")
preds_st_test <- ifelse(preds_st_test > 0.450, "1", "0")
confusionMatrix(preds_st_test, test_datanew1$V15)