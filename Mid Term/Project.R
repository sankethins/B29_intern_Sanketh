rm(list = ls(all = TRUE))

library(caret)
library(DMwR)
library(glmnet)
library(foreign)
library(dplyr)
library(ggplot2)
library(pROC)
require(e1071)
library(C50)
library(rpart)
library(MLmetrics)
library(kernlab)
library(randomForest)
library(mlr)
library(rpart)
library(data.table)
library(h2o)
library(factoextra)
library(ROCR)
library(vegan)
library(class)
library(corrplot)
library(ROSE)
library(MASS)

setwd("D:/Sanketh/INSOFE/Internship/Day 1/b29-interns-data-sankethins-master/Data")
getwd()

# Reading data
census_data<-read.table("adult1.data", header = F, sep = ",", na.strings = c(""," "," ?","NA",NA))
sum(is.na(census_data))
census_data$V15<-ifelse(census_data$V15 %in% c(" >50K"), 0, 1)
census_data$V15<-as.factor(census_data$V15)
nearZeroVar(census_data[, -15], saveMetrics = TRUE)

census_data$V5<-NULL
census_data$V11<-NULL
census_data$V12<-NULL

str(census_data)
summary(census_data)
sum(is.na(census_data))

# Missing value imputation
colMeans(is.na(census_data))
census_data[,which(colMeans(is.na(census_data)) > 0.1)]
census_data<-knnImputation(census_data)
sum(is.na(census_data))

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

numeric_data1 <- as.data.frame (sapply(numeric_data1, replace_outlier_with_missing))
sum(is.na(numeric_data1$V13)) #9008
sum(is.na(numeric_data1$V1)) #143
sum(is.na(numeric_data1$V3)) #992

# Splitting the data into train test and validation
set.seed(123)
train_rows<-sample(x = 1:nrow(census_data), size = 0.7*nrow(census_data))
train_data<-census_data[train_rows,]
test_data<-census_data[-train_rows,]

validation_rows<-sample(x = 1:nrow(train_data), size = 0.7*nrow(train_data))
train_data<-train_data[validation_rows,]
validation_data<-train_data[-validation_rows,]

# Standardizing the data
preprocstep<-preProcess(train_data,method = c("center","scale"))
preproc_train_data<-predict(preprocstep, train_data)
preproc_validation_data<-predict(preprocstep, validation_data)
preproc_test_data<-predict(preprocstep, test_data)

# Handling Class Imbalance
table(preproc_train_data$V15)
preproc_train_data_rose<-ROSE(V15~., data = preproc_train_data, seed = 1)$data
table(preproc_train_data_rose$V15)

# Logistic regression
auc_scores<-NULL
logistic_model<-glm(V15~., data = preproc_train_data, family = binomial)
summary(logistic_model)
logistic_model_aic<-stepAIC(logistic_model, direction = "both")
summary(logistic_model_aic)
logistic_model_rose<-glm(V15~., data = preproc_train_data_rose, family = binomial)
summary(logistic_model_rose)
logistic_model_aic_rose<-stepAIC(logistic_model_rose, direction = "both")
summary(logistic_model_aic_rose)

probabilities<-predict(logistic_model, type = "response")
predictions <- prediction(probabilities, preproc_train_data$V15)
perf <- performance(predictions, measure="tpr", x.measure="fpr")
plot(perf, col=rainbow(10), colorize=T, print.cutoffs.at=seq(0,1,0.05))
perf_auc <- performance(predictions, measure="auc")
auc_logistic <- perf_auc@y.values[[1]]
auc_logistic #0.8898914
auc_scores[i]<-data.frame(Model = c("Logistic"),AUC = auc_logistic)
auc_scores[i]

probabilities<-predict(logistic_model_aic, type = "response")
predictions <- prediction(probabilities, preproc_train_data$V15)
perf <- performance(predictions, measure="tpr", x.measure="fpr")
plot(perf, col=rainbow(10), colorize=T, print.cutoffs.at=seq(0,1,0.05))
perf_auc <- performance(predictions, measure="auc")
auc_logistic <- perf_auc@y.values[[1]]
auc_logistic #0.8887511


probabilities<-predict(logistic_model_rose, type = "response")
predictions <- prediction(probabilities, preproc_train_data$V15)
perf <- performance(predictions, measure="tpr", x.measure="fpr")
plot(perf, col=rainbow(10), colorize=T, print.cutoffs.at=seq(0,1,0.05))
perf_auc <- performance(predictions, measure="auc")
auc_logistic <- perf_auc@y.values[[1]]
auc_logistic #0.4954455

probabilities<-predict(logistic_model_aic_rose, type = "response")
predictions <- prediction(probabilities, preproc_train_data$V15)
perf <- performance(predictions, measure="tpr", x.measure="fpr")
plot(perf, col=rainbow(10), colorize=T, print.cutoffs.at=seq(0,1,0.05))
perf_auc <- performance(predictions, measure="auc")
auc_logistic <- perf_auc@y.values[[1]]
auc_logistic #0.4954455

probabilities_val<-predict(logistic_model,preproc_validation_data, type = "response")
predictions_val<-ifelse(probabilities_val > 0.75, 1, 0)
confusion_matrix_logistic<-confusionMatrix(predictions_val, preproc_validation_data$V15)
confusion_matrix_logistic

##Decision Trees C5.0
c50_model<-C5.0(V15~.,data=preproc_train_data)
predictions_c50<-predict(c50_model,preproc_validation_data,type = "class")
confusion_matrix_c50<-confusionMatrix(predictions_c50,preproc_validation_data$V15)
confusion_matrix_c50 #0.8529,0.5628

c50_model_rose<-C5.0(V15~.,data=preproc_train_data_rose)
predictions_c50<-predict(c50_model_rose,preproc_validation_data,type="class")
confusion_matrix_c50<-confusionMatrix(predictions_c50,preproc_validation_data$V15)
confusion_matrix_c50 #0.8184,0.5779

cart_model<-rpart(V15~.,preproc_train_data)
predictions_cart<-predict(cart_model,preproc_validation_data,type="class")
confusion_matrix_cart<-confusionMatrix(predictions_cart,preproc_validation_data$V15)
confusion_matrix_cart #0.8218,0.4069

cart_model_rose<-rpart(V15~.,preproc_train_data_rose)
predictions_cart<-predict(cart_model_rose,preproc_validation_data,type="class")
confusion_matrix_cart<-confusionMatrix(predictions_cart,preproc_validation_data$V15)
confusion_matrix_cart #0.7543,0.4537

#KNN
k=c(1:30)
knnAuc1=c()
knnAuc2=c()
for(i in k)
{
  knn_model=knn3(V15 ~ . , preproc_train_data, k = i)
  preds_k <- predict(knnmodel1, Valpcafinal)
  predsknn <- ifelse(preds_k[, 1] > preds_k[, 2], 0, 1)
  roc_obj <- roc(Valpcafinal$TARGET, predsknn)
  knnAuc1[i]=auc(roc_obj)
}
knnAuc1


