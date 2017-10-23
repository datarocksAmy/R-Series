# Q2 : Naive Bayes for Qualitative Predictors 

# Import Library
library(class)      # KNN
library(lubridate)  # Extract Date to Year, Month, Day
library(e1071)      # Naive Bayes
library(caret)      # Confusion Matrix



# Read in dataset kc_weather
KCWeather = read.csv("kc_weather.csv", header=TRUE)
head(KCWeather)

# Numerical Labels for Events
KCWeather$EventLabel <- ifelse(KCWeather$Events == "Rain",0 , ifelse(KCWeather$Events == "Snow", 1, 2))

# Seperate Date into Year, Month, Day
KCWeather$Year <- year(KCWeather$Date)
KCWeather$Month <- month(KCWeather$Date)
KCWeather$Day <- day(KCWeather$Date)

# Observe data summary for Temp.F, Humidity.percentage, Precip.in
summary(KCWeather$Temp.F)
summary(KCWeather$Humidity.percentage)
summary(KCWeather$Precip.in)

# Qualitative categories for predictors Temp.F, Humidity.percentage, Precip.in
KCWeather$TempF_Q <- ifelse(KCWeather$Temp.F <= 46,"T1" , ifelse(KCWeather$Temp.F > 46 & KCWeather$Temp.F <= 62.5, "Tm", "T3"))
KCWeather$Humidity_Q <- ifelse(KCWeather$Humidity.percentage <= 62,"H1" , ifelse(KCWeather$Humidity.percentage > 62 & KCWeather$Humidity.percentage <= 72, "Hm", "H3"))
KCWeather$Precip_Q <- ifelse(KCWeather$Precip.in <= 0.025,"Pm" , ifelse(KCWeather$Precip.in > 0.025 & KCWeather$Precip.in <=0.1728, "Pme", "P3"))

# Set step size 
set.seed(400)

# Parse data randomly into Training 290 and the rest of data for Testing
all <- (1:366)
TrainRandom<- sample(1:366, 290, replace=F)
TestRandom <- setdiff(all, TrainRandom)

# Train and Test Dataset
TrainSet <- KCWeather[TrainRandom,]
TestSet <- KCWeather[TestRandom,]

# Labels for Training and Testing 
TrainEvents <- KCWeather$Events[TrainRandom]
TestEvents <- KCWeather$Events[TestRandom]
TestLabels <- data.frame(TestSet$TempF_Q, TestSet$Humidity_Q, TestSet$Precip_Q)

# Convert to factors
KCWeather$TempF_Q <- as.factor(KCWeather$TempF_Q)
KCWeather$Humidity_Q <- as.factor(KCWeather$Humidity_Q)
KCWeather$Precip_Q <- as.factor(KCWeather$Precip_Q)

# Naive Bayes
NaiveBayesModel <- naiveBayes(Events ~ TempF_Q + Humidity_Q + Precip_Q, data = TrainSet)

# Predict using Test Dat
#testing <- data.frame(TempF_Q = c("T1"), Humidity_Q=c("H1"), Precip_Q=c("P3"))
NB_Predict <- predict(NaiveBayesModel, TestEvents, type="class")
table(NB_Predict, TestEvents)

# Confusion Matrix
NB_Confusion <- confusionMatrix(NB_Predict, TestEvents,  mode = "prec_recall")

