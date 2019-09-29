# 1. Introduction 

# 1.1 Topic Overview

# On April 15, 1912, the largest passenger liner ever made collided with an iceberg during her maiden voyage. When the Titanic sank it killed 1502 out of 2224 passengers and crew. This sensational tragedy shocked the international community and led to better safety regulations for ships. One of the reasons that the shipwreck resulted in such loss of life was that there were not enough lifeboats for the passengers and crew. Although there was some element of luck involved in surviving the sinking, some groups of people were more likely to survive than others.

# 1.2 Description of Dataset

# A dataset is provided in csv format for 891 passengers with information about each passenger. It is clean and imputed for missing values. We represent each passenger in a row with features in columns such as age, cabin, point of embarking the ship, the ticket price, the name of the passenger, the number of parents/children aboard the ship, the unique ID, the class, the gender, the number of siblings or spouses aboard the ship, survival, the ticket number the title of the passenger and the family size. We read below both the original training and test sets as provided on Kaggle [Titanic Dataset Source](https://www.kaggle.com/jamesleslie/titanic-cleaned-data). 
# The original training set has 891 rows while the test set has 418 rows. The test set has no information for survival, only the training set has. For this reason, the original training set has been divided into two sets, one for training and a second for validation. The accuracy of prediction was assessed on the validation set whereas predictions carried out on the test set. Worth mentioning is that in almost all instances predictions where provided on the validation set yet what counts are those on the test set. The training set was used for model derivation and parameters optimizations. 

# 1.3 Project Goals

# The goal of the project is to develop models that best predict whether a passenger would survive or not given his profile described with some of the above features.
# We develop 10 prediction models and assess for each its prediction performance, with the Accuracy metric. Since this is a classification problem, Accuracy is measured on the validation set as the proportion of correct predictions from the total predictions. The models are the following


# 1. Naive Approach
# 2. Naive Best Cutoff
# 3. F1 Sensitivity and Specificity Balancing
# 4. QDA (Quadratic Discriminant Analysis)
# 5. LDA (Linear Discriminant Analysis)
# 6. Linear Regression
# 7. Logistic Regression
# 8. K Nearest Neighbors 
# 9. Classification Tree
# 10. Random Forest 


# 2. Analysis Description 

# 2.1 Importing Data

# We import both the original training  and test sets from the github repository https://github.com/NATabbal/Titanic-Project through the setup procedure.

# 2.2 Data Cleaning

# No data cleaning has been carried out. In fact the dataset provided is clean and imputed for NA's. 

# 2.3 Data Exploration 

# We explore our data with a series of plots and tables for a better insight of variables effects and their different levels. 

# 2.4 Models

# For the first three models, the Naive Approach, the Naive Best Cutoff and F1 Sensitivity and Specifity Balancing, we have included one predictor that is the Fare. As mentioned above we hypothesized that passenger paying higher fares where better served and seated and hence had higher chances for survival. 
# For QDA, LDA and Linear Regression we added the Age predictor to the Fare. 
# For K Nearest Neighbors, we kept only the categorical variable (Class, Point of Embarkation, Sex and Title) as the algorithm will treat inter-category distances equally which doesn't pose a problem.  
# For Logistic Regression, Classification Tree and Random Forest, we expand to six predictors to include Age, Class, Point of Embarkation, Fare, Sex and Title. 

# 2.5 Insights

# Almost no correlation between the age and fare variables. 
# Chances for survival is independent of age
# Chances for survival is dependent of fare: higher fare gives a higher chance of survival 
# Passengers traveling in class "1" managed to survive more than oher classes
# Females managed to survive more than males 


# 3. Setup, Visualization, Modeling and Results


# 3.1 Setup

if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(dplyr)) install.packages("dplyr", repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")
if(!require(data.table)) install.packages("data.table", repos = "http://cran.us.r-project.org")
if(!require(rpart)) install.packages("rpart", repos = "http://cran.us.r-project.org")
if(!require(ggplot2)) install.packages("ggplot2", repos = "http://cran.us.r-project.org")
if(!require(randomForest)) install.packages("randomForest", repos = "http://cran.us.r-project.org")
if(!require(purrr)) install.packages("purrr", repos = "http://cran.us.r-project.org")
if(!require(descr)) install.packages("descr", repos = "http://cran.us.r-project.org")
if(!require(kableExtra)) install.packages("kableExtra", repos = "http://cran.us.r-project.org")
if(!require(plyr)) install.packages("plyr", repos = "http://cran.us.r-project.org")

library(tidyverse)
library(dplyr)
library(caret)
library(data.table)
library(rpart)
library(ggplot2)
library(randomForest)
library(purrr)
library(descr)
library(kableExtra)
library(plyr)

# 3.2 Reading the data from the Github repository
#Reading both the titanic and test sets from the github repository
titanic_train <- read_csv("https://raw.githubusercontent.com/NATabbal/Titanic-Project/master/train_clean.csv")
titanic_test <- read_csv("https://raw.githubusercontent.com/NATabbal/Titanic-Project/master/test_clean.csv")

# First six entries of both titanic_train and titanic_test sets
head(titanic_train)
head(titanic_test)

# Dimension of the original titanic training set
dim(titanic_train)

# Names of the original titanic training set 
names(titanic_train)

# Dimension of the original titanic test set
dim(titanic_test)

# Names of the original titanic test set 
names(titanic_test)

# Below are the description of the train and test sets variables 
# 1. Age: Age of passenger 
# 2. Cabin: Cabin number 
# 3. Embarked: Point of embarking the ship (C = Cherbourg; Q = Queenstown; S = Southampton)
# 4. Fare: Price of ticket 
# 5. Name: Name of passenger 
# 6. Parch: Number of parents/children aboard the ship
# 7. PassengerId: Unique ID of passenger
# 8. Pclass: Class of passenger (1 = 1st; 2 = 2nd; 3 = 3rd)
# 9. Sex: Gender of passenger
# 10. SibSp: Number of siblings/spouses aboard the ship
# 11. Survived: True if passenger survived the disaster (0 = No; 1 = Yes)
# 12. Ticket: Ticket number
# 13. Title: Title of passenger (Mr, Mrs etc.)
# 14. Family_Size: Parch + SibSp

# Set the seed to 1
set.seed(1)

# We generate indices to partition the training set into a training and validation set 
index <- createDataPartition(titanic_train$Survived , times = 1, p = 0.5, list= FALSE)
length(index)

# We partition the training set into a training and validation set
train <- titanic_train[-index,]
validation <- titanic_train[index,]

# We explore the first six entries of the train data set 
options(dplyr.width = Inf)
head(train)

# In the training set we transform the variables Pclass, Embarked, Sex, Title and 
# Survived to factor variables 

train <- train %>% mutate (Class_f = as.factor(train$Pclass),
                           Embarked_f = as.factor(train$Embarked),
                           Sex_f = as.factor(train$Sex),
                           Title_f = as.factor(train$Title), 
                           Survived_f = as.factor(train$Survived)) 

# In the validation set we transform the variables Pclass, Embarked, Sex, Title and 
# Survived to factor variables 

validation <- validation %>% mutate (Class_f = as.factor(validation$Pclass),
                                     Embarked_f = as.factor(validation$Embarked),
                                     Sex_f = as.factor(validation$Sex),
                                     Title_f = as.factor(validation$Title), 
                                     Survived_f = as.factor(validation$Survived)) 


# We transform in the test set the variables Pclass, Embarked, Sex, Title and 
# Survived to factor variables, this is where the final predictions are going to be made

titanic_test <- titanic_test %>% mutate (Class_f = as.factor(titanic_test$Pclass),
                                         Embarked_f = as.factor(titanic_test$Embarked),
                                         Sex_f = as.factor(titanic_test$Sex),
                                         Title_f = as.factor(titanic_test$Title), 
                                         Survived_f = as.factor(titanic_test$Survived))  


# 3.3 Data Visualization

# Exploring and ploting data prevalence in a table
prev <- table(train$Survived_f)
prev
pr_prev <- prop.table(prev)
pr_prev
barplot(prev, main = "Survival Distribution", col=c("blue", "yellow"), legend = rownames(prev), 
        args.legend = list(bty = "n", x = "topright"), ylim=c(0,300))

## The data is moderately imbalanced between survival and non-survival with 60% of non-survival
## (level "0") to be considered as the positive class, and 40% of survival (level "1") 

# Barplot of survival per class of passenger 
S_cla <- table(train$Survived_f, train$Class_f)
barplot(S_cla, main="Survival by Class of Passenger", xlab="Class of passenger", 
        col=c("red", "yellow"), legend = rownames(S_cla), args.legend = list(bty = "n", x = "topleft"), 
        ylim=c(0,200), beside=TRUE)
## Class 1 had a higher survival rate while class 3 had a significant non-survival rate 

# Barplot of survival per point of embarkation 
S_emb <- table(train$Survived_f, train$Embarked_f)
barplot(S_emb, main="Survival by Point of Embarkation", xlab="Points of embarkation", 
        col=c("darkblue", "red"), legend = rownames(S_emb), args.legend = list(bty = "n", x = "topleft"), 
        beside=TRUE)
## Southampton had a significant non-survival rate compared to survival, as opposed to 
## Queenstown and Cherbourg

# Barplot of survival per gender of passenger
S_sex <- table(train$Survived_f, train$Sex_f)
S_sex
prop.table(S_sex, 2)
barplot(S_sex, main="Survival by Gender", xlab="Gender of passenger", 
        col=c("grey", "yellow"), legend = rownames(S_sex), args.legend = list(bty = "n", x = "topright"), 
        ylim=c(0,250), beside=TRUE)
## 76% of females survived against 21% of male

# Barplot of survival per title of passenger
S_title <- table(train$Survived_f, train$Title_f)
S_title
prop.table(S_title, 2)
barplot(S_title, main="Survival by Title", xlab="Title of passenger", 
        col=c("grey", "yellow"), legend = rownames(S_title), args.legend = list(bty = "n", x = "topright"), 
        ylim=c(0,250), beside=TRUE, cex.names = 0.7)
## There were significantly non-surviving passengers among those holding the title "Mr." as opposed to "Mrs."
## and "Ms."

# Boxplot for fare for each of the two survival categories 
fare_box <- ggplot(train, aes(x=Survived_f, y=Fare, fill=Survived_f)) + 
  geom_boxplot() + facet_grid(.~Survived_f)
fare_box
## The higher the Fare the higher the chance of survival 

# Boxplot for Age for each of the two survival categories
Age_box <- ggplot(train, aes(x=Survived_f, y=Age, fill=Survived_f)) + 
  geom_boxplot() + facet_grid(.~Survived_f)
Age_box
## The age distribution was almost identical for those who survived and those who did not

# Plot of Fare vs. Age in order to examine if any linear relationship
ggplot(train, aes(x=Fare, y=Age, color="red")) + geom_point(shape=1)

# The correlation between Fare and Age
cor(train$Fare, train$Age)

## The correlation is as low as 0.07: Both will be included in prediction models


# 3.4 Modeling

# 3.4.1 Naive Approach

#E(Y/ X > x)
#In our case, we predict Y, to be 1 i.e. survival whenever the Fare is larger than 22.5 
#Y = 1  / X >= 22.5 
#Y = 0  / X < 22.5

# We plot a density plot stratified by survival i.e. by Survived_f
mu <- ddply(train, "Survived_f", summarise, grp.mean=mean(Fare))
mu
den_plo_f_Sf <-ggplot(train, aes(x=Fare, color=Survived_f)) +
  geom_density() +
  geom_vline(data=mu, aes(xintercept=grp.mean, color=Survived_f),
             linetype="dashed")
den_plo_f_Sf

## We notice that somewhere below 22.5 the chances for survival are lower than non-survival
## however above 22.5 the chances of survival are higher

# We develop a naive approach to guessing survival on the validation set: if Fare paid 
# is higher than 22.5 we predict survival otherwise non-survival

# Naive prediction model on the validation set
y_hat_naive <- ifelse(validation$Fare > 22.5, "1" , "0") %>% 
  factor(levels = levels(validation$Survived_f))
y_hat_naive
# Naive accuracy achieved on the validation set
naive_acc <- confusionMatrix(y_hat_naive, validation$Survived_f)$overall["Accuracy"]
naive_acc

# Naive prediction model on the test set
y_hat_naive <- ifelse(titanic_test$Fare > 22.5, "1" , "0")
y_hat_naive

# Naive accuracy result in a data frame
df1 <- data_frame(Model="Naive Bayes", Accuracy=naive_acc) 
df1 %>% knitr::kable() %>% kable_styling("striped" , full_width = T) 

# 3.4.2 Naive Best Cutoff

#E(Y/ X > x1, x2, x3...xp)

#In our case, we predict Y, to be 1 i.e. survival whenever the Fare is larger than 22.5 

# We search for the optimal cutoff value of the Fare that maximizes Accuracy
cutoff <- seq (0, 512, 10)
accuracy <- map_dbl (cutoff, function(x) {
  y_hat <- ifelse (train$Fare > x, "1", "0") %>% 
    factor (levels = levels(validation$Survived_f))
  mean(y_hat == train$Survived_f)
})
max(accuracy)

# The best cutoff leading to highest accuracy 
best_cutoff  <-  cutoff[which.max(accuracy)]
best_cutoff

# Naive best cutoff predictions on the validation set
y_hat_naive_bc <- ifelse(validation$Fare > 50, "1" , "0") %>% factor(levels = levels(validation$Survived_f))
y_hat_naive_bc

# Accuracy achieved on the validation set with the best cutoff of 50 
naive_acc_best <- confusionMatrix(y_hat_naive_bc, validation$Survived_f)$overall["Accuracy"]
naive_acc_best

# Predictions on the test set
y_hat_naive_bc_t <- ifelse(titanic_test$Fare > 50, "1" , "0") 
y_hat_naive_bc_t

# Naive best cutoff added to the data frame df1
df2 <- bind_rows(df1, data_frame(Model="Naive Best Cutoff", Accuracy=naive_acc_best))
df2 %>% knitr::kable() %>% kable_styling("striped" , full_width = T)

# 3.4.3 F1 Sensitivity and Specifity Balancing

#Sensitivity = TP/(TP+FN)
#Specificity = TP/(TP + FP) 
#F1 = 2. (Sensitivity . Specificity)/(Sensitivity + Specificity)


# We generate a sequence of cutoff fare values
Cutoff <- seq(0, 512, 10)

# We develop a function that calculates for each cutoff the F1 value that balances 
# sensitivity and specificity

F1 <- map_dbl (cutoff, function(x){
  y_hat <- ifelse (train$Fare > x, "1", "0") %>% 
    factor (levels = levels(validation$Survived_f))
  F_meas(data=y_hat, reference = factor(train$Survived_f))
})

# Plotting the cutoff against F1
plot(Cutoff, F1, xlim = c(0, 165), type = "l", col="green") 

# The maximum F1 value achieved 
max(F1)

## The maximum achieved balance between sensitivity and specifity is at 77.2% 

# The cutoff that balances best sensitivity and specificity is 50 
best_cutoff  <-  cutoff[which.max(F1)]
best_cutoff

# Predictions on the validation set based on best F1 score fare cutoff value
y_hat_f1 <- ifelse(validation$Fare > best_cutoff, "1" , "0") %>% 
  factor (levels= levels(validation$Survived_f))
y_hat_f1

# Confusion Matrix on the validation set 
confusionMatrix(data = y_hat_f1, reference = validation$Survived_f)

# Achieved accuracy on the validation set 
acc_f1 <- confusionMatrix(data = y_hat_f1, reference = validation$Survived_f)$overall["Accuracy"]
acc_f1

# Prediction based on the test set best F1 score fare cutoff value
y_hat_f1_t <- ifelse(titanic_test$Fare > best_cutoff, "1" , "0")
y_hat_f1_t

# F1 balanced accuracy added to the data frame df2
df3 <- bind_rows(df2, data_frame(Model="F1 Balancing", Accuracy=acc_f1))
df3 %>% knitr::kable() %>% kable_styling("striped" , full_width = T)

# 3.4.4. QDA (Quadratic Discriminant Analysis)

#The distributions of Pr X/Y=0  (X) and Pr X/Y=1  (X) are multivariate normal where X in our case represents Age and Fare 
#Fare = a. Age^2 + b. Age + c

# We derive the average, standard deviation and correlation for Age and Fare per survival group
params <- train %>% 
  group_by(Survived_f) %>% 
  summarize(avg_1 = mean(Fare), avg_2 = mean(Age), 
            sd_1= sd(Fare), sd_2 = sd(Age), 
            r = cor(Fare, Age))
params

# We plot a contour plot of the Age and Fare
train  %>% 
  ggplot(aes(Fare, Age, fill = Survived_f, color=Survived_f)) + 
  geom_point(show.legend = FALSE) + 
  stat_ellipse(type="norm", lwd = 1.5)

# We reproduce a faceted plot of the Age and Fare
train %>% 
  ggplot(aes(Fare, Age, fill = Survived_f, color=Survived_f)) + 
  geom_point(show.legend = FALSE) + 
  stat_ellipse(type="norm") +
  facet_wrap(~Survived_f)

# We notice that both conditional distributions are not bivariate normal

# Fitting qda that assumes the conditional distributions are bivariate normal 
train_qda <- train(Survived_f~Fare + Age ,
          method= "qda", data = train)

# Prediction on the validation set with qda
fit_qda <- predict(train_qda, validation, type="raw")
fit_qda

# Accuracy achieved with the quadratic model qda
acc_qda <- confusionMatrix(predict(train_qda, validation), validation$Survived_f)$overall["Accuracy"]
acc_qda

# QDA prediction on the test set 
fit_qda_t <- predict(train_qda, titanic_test)
fit_qda_t

# QDA accuracy added to the data frame df3
df4 <-  bind_rows(df3, data_frame(Model="QDA", Accuracy=acc_qda))
df4 %>% knitr::kable() %>% kable_styling("striped" , full_width = T)
 
# 3.4.5. LDA (Linear Discriminant Analysis)

#Fare = a. Age + b

# We assume that the correlation structure is the same for both classes of survival 
params <- params %>% mutate(Fare_sd = mean(sd_1), Age_sd = mean(sd_2), r=mean(r))
params 

# Fitting lda that assumes the conditional distributions are bivariate normal 
train_lda <- train(Survived_f~Fare + Age,
                   method= "lda", data = train)
train_lda

# Prediction on the validation set with lda
fit_lda <- predict(train_lda, validation, type="raw")
fit_lda

# Accuracy achieved with the linear model lda
acc_lda <- confusionMatrix(predict(train_lda, validation), validation$Survived_f)$overall["Accuracy"]
acc_lda

# LDA prediction on the test set 
fit_lda_t <- predict(train_lda, titanic_test)
fit_lda_t

# LDA accuracy added to the data frame df4
df5 <-  bind_rows(df4, data_frame(Model="LDA", Accuracy=acc_lda))
df5 %>% knitr::kable() %>% kable_styling("striped" , full_width = T)

# 3.4.6. Linear Regression

#Survived = a + b.Fare + c.Age

# We estimate the linear regression coefficients with least squares
lm_fit <- mutate(train, y=as.numeric(Survived_f == "1")) %>% lm(y~Fare + Age, data = .)
lm_fit
# Probability of survival prediction on the validation set
p_hat <- predict(lm_fit, validation, type="response")
p_hat
# We predict survival on the validation set if the probability is greater than 0.5
y_hat_lr <- ifelse(p_hat > 0.5, "1", "0") %>% factor()

# Accuracy of linear regression on the validation set
acc_lr <- confusionMatrix(y_hat_lr, validation$Survived_f)$overall["Accuracy"]
acc_lr

# Probability of survival prediction on the test set 
p_hat_lr_t <- predict(lm_fit, titanic_test, type="response")
p_hat_lr_t
# We predict survival on the test set if the probability is greater than 0.5
y_hat_lr_t <- ifelse(p_hat_lr_t > 0.5, "1", "0") 
y_hat_lr_t

# Linear regression accuracy added to the data frame df5
df6 <-  bind_rows(df5, data_frame(Model="Linear Regression", Accuracy=acc_lr))
df6 %>% knitr::kable() %>% kable_styling("striped" , full_width = T)

## However this model has the risk of predicting negative probabilities the reason why we develop
## a logistic model where the log of odds is modeled by a linear regression

# 3.4.7. Logistic Regression

#g(p(Age, Class_f, Embarked_f, Fare, Sex_f, Title_f)) = a + b.Age + c.Class_f + d.Embarked_f + e.Fare + f.Sex_f + g.Title_f
#where g(p) = log(p/1-p)

# We fit a logistic regression model
glm_fit <- train %>% mutate (y = as.numeric(Survived_f == "1")) %>% 
  glm(y ~ Age + Class_f + Embarked_f + Fare + Sex_f + Title_f, data = ., family = "binomial")
glm_fit

# Probability of surviving on the validation set given a paid fare after logistic smoothing 
p_hat_logit <- predict(glm_fit, newdata = validation, type="response")
p_hat_logit

# Prediction of survival or not on the validation set according to logistic model 
y_hat_logit <- ifelse(p_hat_logit > 0.5, "1", "0") %>% factor ()
y_hat_logit

# Accuracy of logistic regression on the validation set
acc_logit <- confusionMatrix(y_hat_logit, validation$Survived_f)$overall["Accuracy"]
acc_logit

# Probability of surviving on the test set given a paid fare after logistic smoothing 
p_hat_logit_t <- predict(glm_fit, newdata = titanic_test, type="response")
p_hat_logit_t

# Prediction of survival or not on the test set according to logistic model 
y_hat_logit_t <- ifelse(p_hat_logit_t > 0.5, "1", "0") 
y_hat_logit_t

# Logistic regression accuracy added to the data frame df6
df7 <-  bind_rows(df6, data_frame(Model="Logistic Regression", Accuracy=acc_logit))
df7 %>% knitr::kable() %>% kable_styling("striped" , full_width = T)

# 3.4.8 K Nearest Neighbours 

# Training the model with the default values of k from 1 to 9
train_knn <- train(Survived_f~Class_f + Embarked_f + Sex_f + Title_f,
                   method = "knn", 
                   data = train)

# Plot of nearest neighbors versus accuracy before tuning of k 
ggplot(train_knn, highlight = TRUE)

# Knn prediction of survival on the validation set
y_hat_knn <- predict(train_knn, validation, type="raw")

# Best k that leads to the maximum accuracy before tuning
train_knn$bestTune

# Best performing model i.e. the predictions yet on the training set before tuning
train_knn$finalModel

# Accuracy on the validation set without tuning of parameters
confusionMatrix(predict(train_knn, validation), validation$Survived_f)$overall["Accuracy"]

# Knn predictions of survival on the test set
y_hat_knn_t <- predict(train_knn, titanic_test, type="raw")
y_hat_knn_t

# Tuning of parameter k by selecting 300 values (takes several minutes)
train_knn_tune <- train(Survived_f~Class_f + Embarked_f + Sex_f + Title_f,
                  method = "knn", tuneGrid = data.frame(k = seq(3, 200)),
                  data = train)

# Plot of nearest neighbors versus accuracy after tuning 
ggplot(train_knn_tune, highlight = TRUE)

# Knn tuned predictions of survival on the validation set
y_hat_knn_tune <- predict(train_knn_tune, validation, type="raw")

# Best k that leads to the maximum accuracy after tuning
train_knn_tune$bestTune

# Best performing model i.e. the predictions yet on the training set after tuning
train_knn_tune$finalModel

# Accuracy achieved with knn on the validation set after tuning of parameters 
acc_knn <- confusionMatrix(predict(train_knn_tune, validation), validation$Survived_f)$overall["Accuracy"]
acc_knn

# Knn tuned predictions of survival on the test set
y_hat_knn_tune_t <- predict(train_knn_tune, titanic_test, type="raw")
y_hat_knn_tune_t

## We conclude that tuning k does not improve the accuracy much in our case. This is because best 
## accuracies are achieved for low values of k

# KNN accuracy added to the data frame df7
df8 <-  bind_rows(df7, data_frame(Model="KNN", Accuracy=acc_knn))
df8 %>% knitr::kable() %>% kable_styling("striped" , full_width = T)

# 3.4.9. Classification Tree

# We train our model on the caret rpart model in order to form a decision tree
train_rpart <- train(Survived_f~Age + Class_f + Embarked_f + Fare + Sex_f + Title_f, method="rpart",
                     tuneGrid=data.frame(cp = seq(0.0, 0.1, len=25)), 
                     data=train, na.action=na.omit)

train_rpart
# Plotting the decision tree formatted with labels
plot(train_rpart$finalModel, uniform=TRUE, main="Classification Tree")
text(train_rpart$finalModel, use.n=TRUE, all=TRUE, cex=.8)

# Accuracy achieved with the classification tree on the validation set 
acc_class <- confusionMatrix(predict(train_rpart, validation), validation$Survived_f)$overall["Accuracy"]
acc_class

# Classification tree predictions on the validation set
y_hat_ct <- predict(train_rpart, validation)
y_hat_ct

# Classification tree predictions on the test set
y_hat_ct_t <- predict(train_rpart, titanic_test)
y_hat_ct_t

# Classification tree accuracy added to the data frame df8
df9 <-  bind_rows(df8, data_frame(Model="Classification Tree", Accuracy=acc_class))
df9 %>% knitr::kable() %>% kable_styling("striped" , full_width = T)

# 3.4.10. Random Forest


# We build decision trees using the training set. We refer to the fitted models as  T1,T2,…,TB

# For every observation in the test set, form a prediction ^yj using tree Tj

# predict ^y with majority vote (most frequent class among ^y1,…,^yT)

# Random Forest model 
random_Forest <- randomForest(Survived_f~Age + Class_f + Embarked_f + Fare + Sex_f + Title_f,
                              data = train)

random_Forest


# We plot the random forest
plot(random_Forest)
legend("right", colnames(random_Forest$err.rate), col=1:4, cex=0.4, fill=1:4)

## We notice that the OOB error stabilizes after some 170 trees

# Variable Importance  
varImp(random_Forest)
# Title_f and Fare are the two most important features in a sense that they reduce
# mostly impurity whenever used at tree nodes across all trees

# Random Forest accuracy achieved on the validation set
acc_rf<- confusionMatrix(predict(random_Forest, validation), validation$Survived_f)$overall["Accuracy"]
acc_rf

# Predictions using the random forest model on the validation set
y_hat_rf <- predict(random_Forest, validation, type="class")
y_hat_rf

# Prediction using the random forest model on the test set
y_hat_rf_t <- predict(random_Forest, titanic_test, type="class")
y_hat_rf_t

# Random forest accuracy added to the data frame df9
df10 <-  bind_rows(df9, data_frame(Model="Random Forest", Accuracy=acc_rf))
df10 %>% knitr::kable() %>% kable_styling("striped" , full_width = T)


# 4. Conclusion/Brief Summary

# Since Random Forest is the best performing model, we replace the survival variable in the original titanic_testset with the predicted ones using the following code: 

titanic_test_S <- titanic_test %>% mutate (Survived_f = y_hat_rf_t)
head(titanic_test_S)

# Clearly the best performing models are those providing high accuracies. In this respect, we note the Logistic model, the KNN, the Classification Tree and Random Forest. 
# If we were to predict the risk of survival, the Random Forest would answer best yet short of roughly 16% accuracy. The variables used in the Random Forest model are good features
# to estimating patterns of survival in the event of any future ship sinking. 
