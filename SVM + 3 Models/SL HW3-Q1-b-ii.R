# Q1-b-ii : Analyze predictors to observe accuracy, precision and recall changes.

# Import Library
library(lubridate) # Extract Date to Year, Month, Day
library(caret)
library(party)     # Random Forest
library(relaimpo)  # Relative Importance
library(earth)     # MARS
library(dplyr)     # Sort Result
library(MASS)      # LDA
library(e1071)
library(class)     # KNN
library(gridExtra)
library(ggplot2)


# Read in dataset kc_weather
KCWeather = read.csv("kc_weather.csv", header=TRUE)
head(KCWeather)

# Seperate Date into Year, Month, Day
KCWeather$Year <- year(KCWeather$Date)
KCWeather$Month <- month(KCWeather$Date)
KCWeather$Day <- day(KCWeather$Date)

# Numerical Labels for Events
KCWeather$EventLabel <- ifelse(KCWeather$Events == "Rain",0 , ifelse(KCWeather$Events == "Snow", 1, 2))

# -------------------------------------------------- Random Forest Model --------------------------------------------------
# Random Forest Model
RandomForest <- cforest(EventLabel ~ Temp.F + Dew_Point.F + Humidity.percentage + Sea_Level_Press.in + Visibility.mi + Wind.mph + Precip.in + Year + Month + Day , data = KCWeather, control=cforest_unbiased(mtry=2,ntree=50))

# Variable of importance by mean decrease in accuracy
VarImp_mean <- varimp(RandomForest)

# Adject correlations between predictors if the condition is TRUE
AjectCorr <- varimp(RandomForest, conditional=TRUE) 

# Robust class imbalance
RobustClassBal <- data.frame(varimpAUC(RandomForest))
evaludateRC <- tibble::rownames_to_column(RobustClassBal, "Variables")

# Sort 
sortRC <- arrange(evaludateRC, desc(evaludateRC$varimpAUC.RandomForest.), row.names(evaludateRC))


# -------------------------------------------------- Relative Importance --------------------------------------------------
# Linear Regression Model
LR <- lm(EventLabel ~ Temp.F + Dew_Point.F + Humidity.percentage + Sea_Level_Press.in + Visibility.mi + Wind.mph + Precip.in + Year + Month + Day , data = KCWeather)

# Calculate relative of importance scaled to 100
relativeImportScore <- calc.relimp(LR, type = "lmg", rela = TRUE)

# Relative of importance in decreasing order
RelativeImportResult <- data.frame(relativeImportScore$lmg)
evaluateRI <- tibble::rownames_to_column(RelativeImportResult, "Variables")

# Sort
sortevaluateRI <- arrange(evaluateRI, desc(evaluateRI$relativeImportScore.lmg, row.names(evaluateRI)))


# -------------------------------------------------- MARS --------------------------------------------------
# MARS model
marsModel <- earth(EventLabel ~ Temp.F + Dew_Point.F + Humidity.percentage + Sea_Level_Press.in + Visibility.mi + Wind.mph + Precip.in + Year + Month + Day, data = KCWeather)

# Estimate Variable Importance
EstVar <- evimp(marsModel)


# --------------------------------------- SUM PREDICTORS CHOICE ---------------------------------------
merge_Robust_RI <- merge(RobustClassBal, RelativeImportResult["relativeImportScore.lmg"], by="row.names", all.x = FALSE)
sort_merge_RF <- arrange(merge_Robust_RI, desc(merge_Robust_RI$varimpAUC.RandomForest.))
sort_merge_RI <- arrange(merge_Robust_RI, desc(merge_Robust_RI$relativeImportScore.lmg))

# Random Forest Graph
RandForestGraph <- ggplot(merge_Robust_RI, aes(x=reorder(merge_Robust_RI$Row.names, -merge_Robust_RI$varimpAUC.RandomForest.), merge_Robust_RI$varimpAUC.RandomForest.), color=merge_Robust_RI$varimpAUC.RandomForest.) + geom_bar(stat="identity", aes(fill = merge_Robust_RI$varimpAUC.RandomForest.)) + geom_text(aes(label = merge_Robust_RI$Row.names), hjust = 1.05, srt=90, size = 3.5, color = "white", fontface= "bold") + theme(axis.text.x=element_blank())
mashRandForest <- RandForestGraph + labs(title = "Random Forest's Picks", x = "Model", y = "9 Categories", fill = "Score") + scale_fill_gradient(low="skyblue2",high="tomato")

# Relative Important Score Graph
RelativeImpScoreGraph <- ggplot(merge_Robust_RI, aes(x=reorder(merge_Robust_RI$Row.names, -merge_Robust_RI$relativeImportScore.lmg), merge_Robust_RI$relativeImportScore.lmg), color=merge_Robust_RI$relativeImportScore.lmg) + geom_bar(stat="identity", aes(fill = merge_Robust_RI$relativeImportScore.lmg)) + geom_text(aes(label = merge_Robust_RI$Row.names), hjust = 1.05, srt=90, size = 3.5, color = "white", fontface= "bold") + theme(axis.text.x=element_blank())
mashRelativeImpScore <- RelativeImpScoreGraph + labs(title = "Relative Important Score's Picks", x = "Model", y = "9 Categories", fill = "Score") + scale_fill_gradient(low="seagreen4",high="darkorange")

# Put 2 graphs in one
grid.arrange(mashRandForest, mashRelativeImpScore, ncol = 2)

# ------------------------------------------------------- Build Models using selected Predictors -------------------------------------------------------
# Parse data randomly into 80% Training and 20% Testing
all <- (1:366)
TrainRandom<- sample(1:366, 290, replace=F) # Take 290 data randomly as Training
TestRandom <- setdiff(all, TrainRandom)     # Rest of the data as Testing

# Train and Test Dataset
TrainSet <- KCWeather[TrainRandom,]
TestSet <- KCWeather[TestRandom,]

# Labels for Training and Testing 
TestEvents <- KCWeather$Events[TestRandom]
TrainEvents <- KCWeather$Events[TrainRandom]

# --------------------------------------------------------- LDA ---------------------------------------------------------
# Train LDA Model
#LDA_Model <- lda(Events ~ Temp.F + Dew_Point.F + Humidity.percentage + Sea_Level_Press.in + Precip.in, data = TrainSet)
LDA_Model <- train(Events ~ Temp.F + Dew_Point.F + Humidity.percentage + Sea_Level_Press.in + Precip.in, data = TrainSet, method = 'lda', repeats=100)

# Test LDA Model
LDA_Predict <- predict(LDA_Model, TestSet)

# Confusion Matrix
#LDA_Class <- LDA_Predict$class  # For lda() command
table(LDA_Predict, TestEvents)

# Test Error Rate 
mean(LDA_Predict == TestEvents)

# Confusion Matrix ( Accuracy, Precision, Recall )
LDA_Confusion <- confusionMatrix(data = LDA_Predict, reference = TestEvents, mode = "prec_recall")

# Accuracy + Precision + Recall
LDA_Accuracy <- LDA_Confusion$overall["Accuracy"]
LDA_Precision <- LDA_Confusion$byClass[, "Precision"]
LDA_Recall <- LDA_Confusion$byClass[, "Recall"]

# --------------------------------------------------------- QDA ---------------------------------------------------------
# Train QDA Model
#QDA_Model <- qda(Events ~ Temp.F + Dew_Point.F + Humidity.percentage + Sea_Level_Press.in + Precip.in, data = TrainSet)
QDA_Model <- qda(Events ~ Temp.F + Dew_Point.F + Humidity.percentage + Sea_Level_Press.in + Precip.in, data = TrainSet, method = 'qda', repeats=100)

# Test QDA Model
QDA_Predict <- predict(QDA_Model, TestSet)

# Confusion Matrix
#QDA_Class <- QDA_Predict$class
table(QDA_Predict, TestEvents)

# Test Error Rate 
mean(QDA_Predict == TestEvents)

# Confusion Matrix ( Accuracy, Precision, Recall )
QDA_Confusion <- confusionMatrix(data = QDA_Predict, reference = TestEvents, mode = "prec_recall")

# Accuracy + Precision + Recall
QDA_Accuracy <- QDA_Confusion$overall["Accuracy"]
QDA_Precision <- QDA_Confusion$byClass[, "Precision"]
QDA_Recall <- QDA_Confusion$byClass[, "Recall"]
# --------------------------------------------------------- SVM ---------------------------------------------------------
# Train SVM Model (L2 Regularized Linear Support Vector Machines with Class Weights)
SVM_Model <- train(Events ~ Temp.F + Dew_Point.F + Humidity.percentage + Sea_Level_Press.in + Precip.in, data = TrainSet, data = TrainSet, method = 'svmLinear3', repeats=100)

# Test SVM Model
SVM_Predict <- predict(SVM_Model, TestSet)

# Fine tune SVM - optimized cost and gamma
SVM_Tune <- tune(svm, Events ~ Temp.F + Dew_Point.F + Humidity.percentage + Sea_Level_Press.in + Precip.in, data = TrainSet, data = TrainSet, ranges = list(epsilon = seq(0,0.2,0.01), cost = 2^(2:9)))

SVM_Model_T <- train(Events ~ Temp.F + Dew_Point.F + Humidity.percentage + Sea_Level_Press.in + Precip.in, data = TrainSet, data = TrainSet, method = 'svmLinear3', cost = 4, gamma= 0, repeats=100)

# Confusion Matrix
table(SVM_Predict, TestEvents)

# Test Error Rate 
mean(SVM_Predict == TestEvents)

# Confusion Matrix ( Accuracy, Precision, Recall )
SVM_Confusion <- confusionMatrix(data = SVM_Predict, reference = TestEvents, mode = "prec_recall")

# Accuracy + Precision + Recall
SVM_Accuracy <- SVM_Confusion$overall["Accuracy"]
SVM_Precision <- SVM_Confusion$byClass[, "Precision"]
SVM_Recall <- SVM_Confusion$byClass[, "Recall"]

# --------------------------------------------------------- KNN ---------------------------------------------------------
# Preprocessing
preprocess <- preProcess(x = TrainSet, method = c("center", "scale"))

# Train Control
TrainCTRL <- trainControl(method="repeatedcv", repeats = 100)

# Build KNN model using Training Data
KNN_Model <- train(Events ~ Temp.F + Dew_Point.F + Humidity.percentage + Sea_Level_Press.in + Precip.in, data = TrainSet, method = "knn", trControl = TrainCTRL, preProcess = c("center","scale"), tuneLength = 50)

# Predict Testing Data using the KNN Model
KNN_Predict <- predict(KNN_Model, newdata = TestSet)

# Confusion Matrix
KNN_Confusion <- confusionMatrix(KNN_Predict, TestEvents, mode = "prec_recall")

# Accuracy + Precision + Recall
KNN_Accuracy <- KNN_Confusion$overall["Accuracy"]
KNN_Precision <- KNN_Confusion$byClass[, "Precision"]
KNN_Recall <- KNN_Confusion$byClass[,"Recall"]

# --------------------------------------------------------- COMPARE MODELS ----------------------------------------------------------
Model <- c("LDA", "QDA", "KNN", "SVM")
Accuracy <- c(LDA_Accuracy, QDA_Accuracy, KNN_Accuracy, SVM_Accuracy)
sum_model_Accuracy <- data.frame(Model, Accuracy)

# Precision/Recall for each Class in each model
# LDA 
LDA_P <- c(LDA_Precision)
LDA_R <- c(LDA_Recall)
# QDA 
QDA_P <- c(QDA_Precision)
QDA_R <- c(QDA_Recall)
# KNN
KNN_P <- c(KNN_Precision)
KNN_R <- c(KNN_Recall)
# SVM
SVM_P <- c(SVM_Precision)
SVM_R <- c(SVM_Recall)

# Summary Precision/Recall
sum_Precision <- data.frame(LDA_P, QDA_P, KNN_P, SVM_P)
sum_P <- tibble::rownames_to_column(sum_Precision, "Class Name")
sum_Recall <- data.frame(LDA_R, QDA_R, KNN_R, SVM_R)
sum_R <- tibble::rownames_to_column(sum_Recall, "Class Name")

# ------------------------------------ Accuracy + Precision + Recall Comparison Graphs ------------------------------------
# Accuracy
AccuracyGraph <- ggplot(sum_model_Accuracy, aes(x=reorder(Model, -Accuracy), Accuracy)) + geom_point(stat="identity", aes(shape = Model, color = Model), size=4) + geom_text(aes(label = Model), hjust = 1.3, srt=0, size = 5, fontface= "bold") + theme(axis.text.x=element_blank())
mashAccuracy <- AccuracyGraph + labs(title = "Accuracy from 4 Models", x = "Model", y = "Accuracy") 

# Precision 
PrecisionG_LDA <- ggplot(sum_P, aes(sum_P$`Class Name`, sum_P$LDA_P), color=sum_P$LDA_P) + geom_bar(stat="identity", aes(fill = sum_P$LDA_P)) + geom_text(aes(label = sum_P$`Class Name`), hjust = 1.05, srt=90, size = 3, color = "white", fontface= "bold") + theme(axis.text.x=element_blank())
mashPrecisionG_LDA <- PrecisionG_LDA + labs(title = "Precision - LDA", x = "LDA", y = "Precision", fill="Precision") + scale_fill_gradient(low="seagreen4",high="darkorange") + ylim(0.0, 0.9)

PrecisionG_QDA <- ggplot(sum_P, aes(sum_P$`Class Name`, sum_P$QDA_P), color=sum_P$QDA_P) + geom_bar(stat="identity", aes(fill = sum_P$QDA_P)) + geom_text(aes(label = sum_P$`Class Name`), hjust = 1.05, srt=90, size = 3, color = "white", fontface= "bold") + theme(axis.text.x=element_blank())
mashPrecisionG_QDA <- PrecisionG_QDA + labs(title = "Precision - QDA", x = "QDA", y = "Precision", fill="Precision") + scale_fill_gradient(low="seagreen4",high="darkorange") + ylim(0.0, 0.9)

PrecisionG_KNN <- ggplot(sum_P, aes(sum_P$`Class Name`, sum_P$KNN_P), color=sum_P$KNN_P) + geom_bar(stat="identity", aes(fill = sum_P$KNN_P)) + geom_text(aes(label = sum_P$`Class Name`), hjust = 1.05, srt=90, size = 3, color = "white", fontface= "bold") + theme(axis.text.x=element_blank())
mashPrecisionG_KNN <- PrecisionG_KNN + labs(title = "Precision - KNN", x = "KNN", y = "Precision", fill="Precision") + scale_fill_gradient(low="seagreen4",high="darkorange") + ylim(0.0, 0.9)

PrecisionG_SVM <- ggplot(sum_P, aes(sum_P$`Class Name`, sum_P$SVM_P), color=sum_P$SVM_P) + geom_bar(stat="identity", aes(fill = sum_P$SVM_P)) + geom_text(aes(label = sum_P$`Class Name`), hjust = 1.05, srt=90, size = 3, color = "white", fontface= "bold") + theme(axis.text.x=element_blank())
mashPrecisionG_SVM <- PrecisionG_SVM + labs(title = "Precision - SVM", x = "SVM", y = "Precision", fill="Precision") + scale_fill_gradient(low="seagreen4",high="darkorange") + ylim(0.0, 0.9)

Precision_Grid <- grid.arrange(mashPrecisionG_LDA, mashPrecisionG_QDA, mashPrecisionG_KNN, mashPrecisionG_SVM, ncol = 4)


# Recall
RecallG_LDA <- ggplot(sum_R, aes(sum_P$`Class Name`, sum_R$LDA_R), color=sum_R$LDA_R) + geom_bar(stat="identity", aes(fill = sum_R$LDA_R)) + geom_text(aes(label = sum_R$`Class Name`), hjust = 1.05, srt=90, size = 3, color = "white", fontface= "bold") + theme(axis.text.x=element_blank())
mashRecallG_LDA <- RecallG_LDA + labs(title = "Recall - LDA", x = "LDA", y = "Recall", fill="Recall") + scale_fill_gradient(low="skyblue2",high="tomato") + ylim(0.0, 0.9)

RecallG_QDA <- ggplot(sum_R, aes(sum_P$`Class Name`, sum_R$QDA_R), color=sum_R$QDA_R) + geom_bar(stat="identity", aes(fill = sum_R$QDA_R)) + geom_text(aes(label = sum_R$`Class Name`), hjust = 1.05, srt=90, size = 3, color = "white", fontface= "bold") + theme(axis.text.x=element_blank())
mashRecallG_QDA <- RecallG_QDA + labs(title = "Recall - QDA", x = "QDA", y = "Recall", fill="Recall") + scale_fill_gradient(low="skyblue2",high="tomato") + ylim(0.0, 0.9)

RecallG_KNN <- ggplot(sum_R, aes(sum_P$`Class Name`, sum_R$KNN_R), color=sum_R$KNN_R) + geom_bar(stat="identity", aes(fill = sum_R$KNN_R)) + geom_text(aes(label = sum_R$`Class Name`), hjust = 1.05, srt=90, size = 3, color = "white", fontface= "bold") + theme(axis.text.x=element_blank())
mashRecallG_KNN <- RecallG_KNN + labs(title = "Recall - KNN", x = "KNN", y = "Recall", fill="Recall") + scale_fill_gradient(low="skyblue2",high="tomato") + ylim(0.0, 0.9)


RecallG_SVM <- ggplot(sum_R, aes(sum_P$`Class Name`, sum_R$SVM_R), color=sum_R$SVM_R) + geom_bar(stat="identity", aes(fill = sum_R$SVM_R)) + geom_text(aes(label = sum_R$`Class Name`), hjust = 1.05, srt=90, size = 3, color = "white", fontface= "bold") + theme(axis.text.x=element_blank())
mashRecallG_SVM <- RecallG_SVM + labs(title = "Recall - SVM", x = "SVM", y = "Recall", fill="Recall") + scale_fill_gradient(low="skyblue2",high="tomato") + ylim(0.0, 0.9)

Recall_Grid <- grid.arrange(mashRecallG_LDA, mashRecallG_QDA, mashRecallG_KNN , mashRecallG_SVM, ncol = 4)
