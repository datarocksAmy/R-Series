# Q1-a-ii : Analyze predictors to observe accuracy, precision and recall changes.

# Import Library
library(lubridate) # Extract Date to Year, Month, Day
library(caret)
library(party)     # Random Forest
library(relaimpo)  # Relative Importance
library(earth)     # MARS
library(dplyr)     # Sort Result




# Read in dataset kc_weather
KCWeather = read.csv("kc_weather.csv", header=TRUE)
head(KCWeather)

# Seperate Date into Year, Month, Day
KCWeather$Year <- year(KCWeather$Date)
KCWeather$Month <- month(KCWeather$Date)
KCWeather$Day <- day(KCWeather$Date)

# Numerical Labels for Events
KCWeather$EventLabel <- ifelse(KCWeather$Events == "Rain",1, ifelse(KCWeather$Events == "Snow", 0, 2))

# -------------------------------------------------- Random Forest Model --------------------------------------------------
# Random Forest Model
RandomForest <- cforest(EventLabel ~ Temp.F + Dew_Point.F + Humidity.percentage + Sea_Level_Press.in + Visibility.mi + Wind.mph + Precip.in + Year + Month + Day , data = KCWeather, control=cforest_unbiased(mtry=2,ntree=50))

# Variable of importance by mean decrease in accuracy
VarImp_mean <- varimp(RandomForest)

# Adject correlations between predictors if the condition is TRUE
AjectCorr <- varimp(RandomForest, conditional=TRUE) 

# Robust class imbalance
RobustClassBal <- data.frame(varimpAUC(RandomForest))
RobustClassBal <- add_rownames(RobustClassBal, "Variables")

# Sort 
sortRobust <- arrange(RobustClassBal, desc(RobustClassBal$varimpAUC.RandomForest.), row.names(RobustClassBal))


# -------------------------------------------------- Relative Importance --------------------------------------------------
# Linear Regression Model
LR <- lm(EventLabel ~ Temp.F + Dew_Point.F + Humidity.percentage + Sea_Level_Press.in + Visibility.mi + Wind.mph + Precip.in + Year + Month + Day , data = KCWeather)

# Calculate relative of importance scaled to 100
relativeImportScore <- calc.relimp(LR, type = "lmg", rela = TRUE)

# Relative of importance in decreasing order
RelativeImportResult <- data.frame(relativeImportScore$lmg)
RelativeImportResult <- add_rownames(RelativeImportResult, "Variables")

# Sort
sortRI <- arrange(RelativeImportResult, desc(RelativeImportResult$relativeImportScore.lmg, row.names(RelativeImportResult)))


# -------------------------------------------------- MARS --------------------------------------------------
# MARS model
marsModel <- earth(EventLabel ~ Temp.F + Dew_Point.F + Humidity.percentage + Sea_Level_Press.in + Visibility.mi + Wind.mph + Precip.in + Year + Month + Day, data = KCWeather)

# Estimate Variable Importance
EstVar <- evimp(marsModel)


