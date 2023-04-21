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
# Data prep and EDa
set.seed(23)
kc1 <- kc1[,-c(1)]
pc1 <- pc1[,-c(1)]

names(pc1)[4] = "iv.g."
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

#addressing class imbalance
newpc1 <- AdasynClassif(defects~., pc1, beta = .4)
newkc1 <- AdasynClassif(defects~., kc1, beta = .4, )
#neural network model for kc1
#Data partition
inTrain <-  createDataPartition(y = newkc1$defects, p = .75, list = FALSE)
#baseline model using linear regression
newkc1$defectsNumeric <- as.numeric(revalue(newkc1$defects,c("false" = 0,"true" = 1)))

newpc1$defectsNumeric <- as.numeric(revalue(newpc1$defects,c("false" = 0,"true" = 1)))
kc1.train2 <- newkc1[inTrain,][,-c(22)]
kc1.test2 <- newkc1[-inTrain,][,-c(22)]
#kc1 baseline model
model1 <- lm(formula = defectsNumeric ~ ., data =
               kc1.train2)
testDefects <-newkc1[-inTrain,][,22] 
BaselinePred1 <- predict(model1,newdata= kc1.test2)
table1 <- table(testDefects, BaselinePred1 > .5)
row.names(table1) <-  c("Actual: False", "Actual: True")
colnames(table1) <-  c("Predicted: False", "Predicted: True")
table1

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
#p-value: .56
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

#accuracy (TN + TP)/ Total
total <- dim(kc1.testScale)[1]
(Testaccuracy <- (t1[1]+t1[4])/total)



#neural network model for pc1

#Data partition
inTrain <-  createDataPartition(y = newpc1$defects, p = .75, list = FALSE)

# Creating training & test datasets
newpc1.train <- newpc1[inTrain,]
newpc1.test <- newpc1[-inTrain,]

newpc1.train$trainortest <-
  rep("train", nrow(newpc1.train))
newpc1.test$trainortest <-
  rep("test", nrow(newpc1.test))

newpc1.all <- rbind(newpc1.train, newpc1.test)
newpc1.all$trainortest <- as.factor(newpc1.all$trainortest)

# Numeric: loc
boxplot(loc ~ as.factor(trainortest), 
        data = newpc1.all)
kruskal.test(loc ~ as.factor(trainortest),
             data = newpc1.all)$p.value
#p-value: .38
#p-value >.05 

#standardization 
newpc1.trainScale <- as.data.frame(scale(newpc1.train[,-c(22,23)]))
newpc1.testScale <- as.data.frame(scale(newpc1.test[,-c(22,23)]))
#adding target variable back 
defects <- newpc1.train$defects
newpc1.trainScale <- cbind(newpc1.trainScale,defects)
defects <- newpc1.test$defects
newpc1.testScale <- cbind(newpc1.testScale, defects)
#create model
nnet02 <-  nnet(defects ~ ., data = newpc1.trainScale, size = 1)
#plot model
plotnet(nnet02)
#obtain weights
pred2 <- predict(nnet02, newdata = newpc1.testScale, type = "class")
#contigency table 
t2 <- table(newpc1.testScale$defects, pred2)
row.names(t2) <-  c("Actual: False", "Actual: True")
colnames(t2) <-  c("Predicted: False", "Predicted: True")
t2

#accuracy (TN + TP)/ Total
total <- dim(newpc1.testScale)[1]
(Testaccuracy <- (t2[1]+t2[4])/total)

pred3 <- predict(nnet01, newdata = newpc1.testScale, type = "class")
