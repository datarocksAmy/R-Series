# Import Library
library(caret)
library(MASS)
library(e1071)
library(class)

# Read in dataset kc_weather
KCWeather = read.csv("kc_weather.csv", header=TRUE)
head(KCWeather)

# Set KCWeather as Default dataset
#attach(KCWeather)

# Subset for Events Rain and Snow from KCWeather dataset
newKCW <- subset(KCWeather, Events != "Rain_Thunderstorm")
#levels(newKCW$Events)[levels(newKCW$Events)=="Snow"] <- 0
#levels(newKCW$Events)[levels(newKCW$Events)=="Rain"] <- 1

# Change level for subset : Snow and Rain
newKCW$Events <- factor(newKCW$Events)

# Set step size 
#set.seed(500)

# Parse data randomly into 80% Training and 20% Testing
all <- (1:226)
TrainRandom<- sample(1:226, 181, replace=F)
TestRandom <- setdiff(all, TrainRandom)

TrainSet <- newKCW[TrainRandom,]
TestSet <- newKCW[TestRandom,]
TestEvents <- newKCW$Events[TestRandom]
TrainEvents <- newKCW$Events[TrainRandom]

# --------------------------------------------------------------------------------- LOGISTIC REGRESSION ---------------------------------------------------------------------------------
# Build Logitstic Regression Model & Train with train dataset
LogiReg_Model <- glm(newKCW$Events ~ Temp.F + Dew_Point.F + Humidity.percentage + Sea_Level_Press.in + Visibility.mi + Wind.mph + Precip.in, data = TrainSet, family=binomial("logit"))
summary(LogiReg_Model)

# Test Model with test dataset
ProbLR <- predict(LogiReg_Model, TestSet, type = "response")

predict_weather = rep("Rain", nrow(TestSet))
predict_weather[LogiReg_Model > 0.0005] = "Snow"   # Still have prolem right here

# Confusion Matrix
LogisticRegr_Confusion <- confusionMatrix(data = predict_weather, reference = TestEvents, mode = "prec_recall")

# Accuracy + Precision + Recall
LR_Accuracy <- LogisticRegr_Confusion$overall["Accuracy"]
LR_Precision <- LogisticRegr_Confusion$byClass["Precision"]
LR_Recall <- LogisticRegr_Confusion$byClass["Recall"]

# ------------------------------------------------------------------------------------------ LDA ------------------------------------------------------------------------------------------
# Train LDA Model
LDA_Model <- lda(Events ~ Temp.F + Dew_Point.F + Humidity.percentage + Sea_Level_Press.in + Visibility.mi + Wind.mph + Precip.in, data = TrainSet)

# Test LDA Model
LDA_Predict <- predict(LDA_Model, TestSet)

# Confusion Matrix
LDA_Class <- LDA_Predict$class
table(LDA_Class, TestEvents)

# Test Error Rate 
mean(LDA_Class == TestEvents)

# Confusion Matrix ( Accuracy, Precision, Recall )
LDA_Confusion <- confusionMatrix(data = LDA_Class, reference = TestEvents, mode = "prec_recall")

# Accuracy + Precision + Recall
LDA_Accuracy <- LDA_Confusion$overall["Accuracy"]
LDA_Precision <- LDA_Confusion$byClass["Precision"]
LDA_Recall <- LDA_Confusion$byClass["Recall"]

# ------------------------------------------------------------------------------------------ QDA ------------------------------------------------------------------------------------------
# Train QDA Model
QDA_Model <- qda(Events ~ Temp.F + Dew_Point.F + Humidity.percentage + Sea_Level_Press.in + Visibility.mi + Wind.mph + Precip.in, data = TrainSet)

# Test QDA Model
QDA_Predict <- predict(QDA_Model, TestSet)

# Confusion Matrix
QDA_Class <- QDA_Predict$class
table(QDA_Class, TestEvents)

# Test Error Rate 
mean(QDA_Class == TestEvents)

# Confusion Matrix ( Accuracy, Precision, Recall )
QDA_Confusion <- confusionMatrix(data = QDA_Class, reference = TestEvents, mode = "prec_recall")

# Accuracy + Precision + Recall
QDA_Accuracy <- QDA_Confusion$overall["Accuracy"]
QDA_Precision <- QDA_Confusion$byClass["Precision"]
QDA_Recall <- QDA_Confusion$byClass["Recall"]

# ------------------------------------------------------------------------------------------ KNN ------------------------------------------------------------------------------------------
# Preprocessing
preprocess <- preProcess(x = TrainSet, method = c("center", "scale"))

# Train Control
TrainCTRL <- trainControl(method="repeatedcv", repeats = 3)

# Build KNN model using Training Data
KNN_Model <- train(Events ~ ., data = TrainSet, method = "knn", trControl = TrainCTRL, preProcess = c("center","scale"), tuneLength = 20)

# Predict Testing Data using the KNN Model
KNN_Predict <- predict(KNN_Model, newdata = TestSet)

# Confusion Matrix
KNN_Confusion <- confusionMatrix(KNN_Predict, TestEvents, mode = "prec_recall")

# Accuracy + Precision + Recall
KNN_Accuracy <- KNN_Confusion$overall["Accuracy"]
KNN_Precision <- KNN_Confusion$byClass["Precision"]
KNN_Recall <- KNN_Confusion$byClass["Recall"]

# ----------------------------------------------------------------------------------- COMPARE MODELS -----------------------------------------------------------------------------------
Model <- c("LDA", "QDA", "KNN")
Accuracy <- c(LDA_Accuracy, QDA_Accuracy, KNN_Accuracy)
Precision <- c(LDA_Precision, QDA_Precision, KNN_Precision)
Recall <- c(LDA_Recall, QDA_Recall, KNN_Recall)
summary_df <- data.frame(Model, Accuracy, Precision, Recall)