library(dplyr)
library(e1071)
library(caret)
library(ROCR)

### Loading the data

data_redwine <- read.csv("~/Desktop/YaleNUS/Y3S1/IntrotoDS/Assignments/DSProj_NaiveBayes/winequality-red.csv")
data_whitewine <- read.csv("~/Desktop/YaleNUS/Y3S1/IntrotoDS/Assignments/DSProj_NaiveBayes/winequality-white.csv", sep=";")

### Check for NAs 
sum(is.na(data_redwine))
sum(is.na(data_whitewine))

### Assign labels 
data_red_wine_labelled <- mutate(data_redwine,
       label = case_when(
         quality > 6 ~ "good", 
         quality <= 6 ~ "bad"
       ))

data_white_wine_labelled <- mutate(data_whitewine,
                                label = case_when(
                                  quality > 6 ~ "good", 
                                  quality <= 6 ~ "bad"
                                ))

### Red Wine NB Analysis
indxTrain <- createDataPartition(y = data_red_wine_labelled$label ,p = 0.75,list = FALSE) # uses random sampling
traindata <- data_red_wine_labelled[indxTrain, !(names(data_red_wine_labelled) %in% c('quality'))]
testdata <- data_red_wine_labelled[-indxTrain, !(names(data_red_wine_labelled) %in% c('quality'))]

nb_model <- naiveBayes(as.factor(label) ~ . , traindata, laplace=.01)

nb_prediction <- predict(nb_model,
                         # remove column "label"
                         testdata[,!(names(data_red_wine_labelled) %in% c('label'))],
                         type='raw') 

score <- nb_prediction[, c("good")]

actual_class <- testdata$label == 'good' 
pred <- prediction(score, actual_class)

perf <- performance(pred, "tpr", "fpr")
plot(perf, lwd=2, xlab="False Positive Rate (FPR)", ylab="True Positive Rate (TPR)")
abline(a=0, b=1, col="gray50", lty=3)


auc <- performance(pred, "auc")
auc <- unlist(slot(auc, "y.values"))
auc


### White Wine NB Analysis
indxTrain_white <- createDataPartition(y = data_white_wine_labelled$label ,p = 0.75,list = FALSE)
traindata_white <- data_white_wine_labelled[indxTrain, !(names(data_white_wine_labelled) %in% c('quality'))]
testdata_white <- data_white_wine_labelled[-indxTrain, !(names(data_white_wine_labelled) %in% c('quality'))]

nb_model_white <- naiveBayes(as.factor(label) ~ . , traindata_white, laplace=.01)

nb_prediction_white <- predict(nb_model_white,
                         # remove column "label"
                         testdata_white[,!(names(data_white_wine_labelled) %in% c('label'))],
                         type='raw') 

score_white <- nb_prediction_white[, c("good")]

actual_class_white <- testdata_white$label == 'good' 
pred_white <- prediction(score_white, actual_class_white)

perf_white <- performance(pred_white, "tpr", "fpr")

plot(perf_white, lwd=2, xlab="False Positive Rate (FPR)", ylab="True Positive Rate (TPR)")
abline(a=0, b=1, col="gray50", lty=3)

auc_white <- performance(pred_white, "auc")
auc_white <- unlist(slot(auc_white, "y.values"))
auc_white


### Side-by-side Plot 
par(mfrow=c(1,2)) 
plot(perf, lwd=2, main="Red Wine Quality Label NB Model Prediction Perf", xlab="False Positive Rate (FPR)", ylab="True Positive Rate (TPR)")
abline(a=0, b=1, col="gray50", lty=3)
plot(perf_white, lwd=2, main="White Wine Quality Label NB Model Prediction Perf", xlab="False Positive Rate (FPR)", ylab="True Positive Rate (TPR)")
abline(a=0, b=1, col="gray50", lty=3)


print(paste0("AUC of NB Model (Red Wine):", auc))
print(paste0("AUC of NB Model (White Wine):", auc_white))
