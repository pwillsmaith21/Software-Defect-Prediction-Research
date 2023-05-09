install.packages("plyr")
install.packages("caret")
intall.packages("car")
install.packages("nnet") 
install.packages("NeuralNetTools") 
install.packages("UBL")
library(caret)
library(plyr)
library(car)
library(nnet)
library(NeuralNetTools)
library("UBL")


#### Data prep and EDa ####
set.seed(23)
kc1 <- kc1[,-c(1)]
pc1 <- pc1[,-c(1)]
#variable rename
for(x in 1:dim(kc1)[2]){
  if((names(pc1)[x]) != (names(kc1)[x])){
    names(pc1)[x] = names(kc1)[x]
  }
}

summary(kc1)
x = 3
ggplot(kc1, aes_string(colnames(kc1)[x])) +
  geom_histogram(aes(fill = defects),
                 color = "black",
                 position = "fill")+
  ggtitle(paste("normalize histogram of",colnames(kc1)[x],"overlay with defects", sep = " "))
ggplot(kc1, aes_string(colnames(kc1)[x]))+
  geom_histogram(aes(fill = defects),
                 color = "black",
                position = "stack")+
  ggtitle(paste("histogram of",colnames(kc1)[x],"overlay with defects", sep = " "))
summary(pc1)
x = 18
ggplot(pc1, aes_string(colnames(pc1)[x])) +
  geom_histogram(aes(fill = defects),
                 color = "black",
                 position = "fill")+
  ggtitle(paste("normalize histogram of",colnames(pc1)[x],"overlay with defects", sep = " "))
ggplot(pc1, aes_string(colnames(pc1)[x]))+
  geom_histogram(aes(fill = defects),
                 color = "black",
                 position = "stack")+
  ggtitle(paste("histogram of",colnames(pc1)[x],"overlay with defects", sep = " "))

colnames(kc1)
model0 <- lm(formula = defectsNumeric ~ ., data =
                kc1[-c(22)])
model0
plot(model0)
options(scipen = 999)
vif(model0)

#bin v(g) variable 
kc1$v.g.Catergorical <- cut(x = kc1$v.g., breaks = c(0,10, Inf), labels = c("<= 10", "> 10"))
(table(kc1$defects,kc1$v.g.Catergorical))


# cor(kc1$e,kc1$t) =1
#cor(kc1$loc, kc1$lOCode) = .985
cor(kc1$uniq_Op, kc1$uniq_Opnd)
#variable pc1
summary(pc1)

#bin v(g) variable 
pc1$v.g.Catergorical <- cut(x = pc1$v.g., breaks = c(0,10, Inf), labels = c("<= 10", "> 10"))
(table(pc1$defects,pc1$v.g.Catergorical))

#displaying imbalance
summary(pc1$defects)
ggplot(pc1, aes(defects)) +
  geom_bar(aes(fill = defects ))+
  ggtitle("Barplot of defects for pc1")

summary(kc1$defects)
ggplot(kc1, aes(defects)) +
  geom_bar(aes(fill = defects ))+
  ggtitle("Barplot of defects for kc1")

# addressing class imbalance
newpc1 <- AdasynClassif(defects~., pc1, beta = .4)
newkc1 <- AdasynClassif(defects~., kc1, beta = .4, )

#### evaluation metric function ####
eval_metrics <- function(table) {
  
  TN <- table[1,1]
  TP <- table[2,2]
  FN <- table[2,1]
  FP <- table[1,2]
  
  # calculate measures
  Accuracy <- (TN + TP) / (TP + TN + FP + FN) 
  ErrorRate <- 1 - Accuracy  
  Sensitivity <- TP / (TP + FN)  # Sensitivity 
  Specificity <- TN / (FP + TN)  # Specificity 
  Precision <- TP / (TP + FP)  # Precision 
  F1 <- 2 * Precision * Sensitivity / (Precision + Sensitivity)  # F1 score
  F2 <- 5 * Precision * Sensitivity / (4 * Precision + Sensitivity)  # F2 score
  F05 <- 1.25 * Precision * Sensitivity / (0.25 * Precision + Sensitivity)  # F0.5 score
  
  cat("Accuracy: ", Accuracy, "\n")
  cat("Error rate: ", ErrorRate, "\n")
  cat("Sensitivity: ", Sensitivity, "\n")
  cat("Specificity: ", Specificity, "\n")
  cat("Precision: ", Precision, "\n")
  cat("Recall: ", Specificity, "\n")
  cat("F1 score: ", F1, "\n")
  cat("F2 score: ", F2, "\n")
  cat("F0.5 score: ", F05, "\n")
  
}
#neural network model for kc1
#Data partition
inTrain <-  createDataPartition(y = newkc1$defects, p = .75, list = FALSE)
#baseline model using Zero Rate Classifier
#kc1
predBkc1 <-  rep("false", nrow(newkc1))
predBkc1 <-  factor(predBkc1,levels = c("false", "true"),labels = c("false", "true"))
(table2 <- table(newkc1$defects, predBkc1))
eval_metrics(table2)
#pc1
predBpc1 <-  rep("false", nrow(newpc1))
predBpc1 <-  factor(predBpc1,levels = c("false", "true"),labels = c("false", "true"))
(table3 <- table(newpc1$defects, predBpc1))
eval_metrics(table3)
# Creating training & test datasets
kc1.train <- kc1[inTrain,]
kc1.test <- kc1[-inTrain,]

kc1.train$trainortest <-
  rep("train", nrow(kc1.train))
kc1.test$trainortest <-
  rep("test", nrow(kc1.test))

kc1.all <- rbind(kc1.train, kc1.test)
kc1.all$trainortest <- as.factor(kc1.all$trainortest)

# Numeric: loc
boxplot(loc ~ as.factor(trainortest), 
        data = kc1.all)
kruskal.test(loc ~ as.factor(trainortest),
             data = kc1.all)$p.value
#p-value: .77
#p-value >.05 

#standardization 
kc1.trainScale <- as.data.frame(scale(kc1.train[,-c(22,23)]))
kc1.testScale <- as.data.frame(scale(kc1.test[,-c(22,23)]))
#adding target variable back 
defects <- kc1.train$defects
kc1.trainScale <- cbind(kc1.trainScale,defects)
defects <- kc1.test$defects
kc1.testScale <- cbind(kc1.testScale, defects)
#create model
nnet01 <-  nnet(defects ~ ., data = kc1.trainScale, size = 1)
#plot model
plotnet(nnet01)
#obtain weights
nnet01$wts
neuralweights(nnet01)
pred1 <- predict(nnet01, newdata = kc1.testScale, type = "class")
#contigency table 
t1 <- table(kc1.testScale$defects, pred1)
row.names(t1) <-  c("Actual: False", "Actual: True")
colnames(t1) <-  c("Predicted: False", "Predicted: True")
t1

#Evaluation metric for kc1 model
eval_metrics(t1)

#neural network model for pc1

#Data partition
inTrain <-  createDataPartition(y = newpc1$defects, p = .75, list = FALSE)

# Creating training & test datasets
pc1.train <- newpc1[inTrain,]
pc1.test <- newpc1[-inTrain,]

pc1.train$trainortest <-
  rep("train", nrow(pc1.train))
pc1.test$trainortest <-
  rep("test", nrow(pc1.test))

pc1.all <- rbind(pc1.train, pc1.test)
pc1.all$trainortest <- as.factor(pc1.all$trainortest)

# Numeric: loc
boxplot(loc ~ as.factor(trainortest), 
        data = pc1.all)
kruskal.test(loc ~ as.factor(trainortest),
             data = pc1.all)$p.value
#p-value: .859
#p-value >.05 

#standardization 
pc1.trainScale <- as.data.frame(scale(newpc1.train[,-c(22,23)]))
pc1.testScale <- as.data.frame(scale(pc1.test[,-c(22,23)]))
#adding target variable back 
defects <- pc1.train$defects
pc1.trainScale <- cbind(pc1.trainScale,defects)
defects <- pc1.test$defects
pc1.testScale <- cbind(pc1.testScale, defects)
#create model
nnet02 <-  nnet(defects ~ ., data = pc1.trainScale, size = 1)
#plot model
plotnet(nnet02)
#obtain weights
pred2 <- predict(nnet02, newdata = pc1.testScale, type = "class")
#contigency table 
t2 <- table(pc1.testScale$defects, pred2)
row.names(t2) <-  c("Actual: False", "Actual: True")
colnames(t2) <-  c("Predicted: False", "Predicted: True")
t2

#Evaluation metric for pc1
eval_metrics(t2)

#### cross-project kc1 for training and pc1 for testing####
pred3 <- predict(nnet01, newdata = pc1.testScale, type = "class")
t3 <- table(pc1.testScale$defects, pred3)
row.names(t3) <-  c("Actual: False", "Actual: True")
colnames(t3) <-  c("Predicted: False", "Predicted: True")
t3
eval_metrics(t3)

#### cross-project pc1 for training and kc1 for testing ####
pred4 <- predict(nnet02, newdata = kc1.testScale, type = "class")
t4 <- table(kc1.testScale$defects, pred4)
row.names(t4) <-  c("Actual: False", "Actual: True")
colnames(t4) <-  c("Predicted: False", "Predicted: True")
eval_metrics(t4)
