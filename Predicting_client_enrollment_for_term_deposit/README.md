# Machine-Learning

author: "Parth Pandya"
---

## PROJECT/ DA 5030

**Bank marketing Dataset**

* Acquiring the dataset

Using the same dataset twice for comparison.

```{r}
banking_test<-read.csv("bank_test.csv",sep = ";")
bank_test<-read.csv("bank_test.csv",sep = ";",stringsAsFactors = F)
bank_test$y<-as.factor(bank_test$y)
summary(banking_test)
str(banking_test)
```

From summary we can see that marital,default,housing,loan, poutcome have "unknown"" values


Dataset comprises of 21 features:
20 independent variables: 
(10 categorical features)- job,marital,education,default,housing,loan,contact,month, day_of_week, poutcome

10 numeric features


* Analysing the features
```{r} 
#duration
duration<- which(bank_test$duration==0)
bank_test[duration,]
duration1<- which(bank_test$duration<100)
bank_test[duration1,]
```
There is 1 value which is equal to zero which indicate that the customer did not subscribe for the term. 

If we test for values less than 100 second there are about 1000 customers who did not subscribe for the term. 

Hence we get biased decision for less number of duration. We will build a model with and without this feature and compare the performance. 

Age and euribor has some outlier values but they are not any random values or miscalcualted values hence we will not replace those values.

* Outlier detection

```{r}
library(DMwR)
hist(banking_test$age)

z_age<-abs(scale(banking_test$age,center = TRUE,scale=TRUE))
out_age<-which(z_age>3)
str(out_age)
str(out_age)
age_out<-banking_test[out_age,]
nrow(age_out)
n<-nrow(banking_test)



bank_out<-banking_test[,c("age","euribor3m","duration","emp.var.rate","nr.employed","pdays")]
outlier.scores<-lofactor(bank_out,k=60)
plot(density(outlier.scores))

#visualize outlier
pch <- rep(".", n)
pch[outlier.scores] <- "+"
col <- rep("black", n)
col[outlier.scores] <- "red"
pairs(bank_out, pch=pch, col=col)


#pdays feature analysis
pastdays<-which(banking_test$pdays==999)
str(pastdays)
hist(banking_test$pdays)

#heavily skewed towards left
#applying some transformations
hist(log(banking_test$pdays+1))
hist(sqrt(banking_test$pdays))

```
* The features have some outliers which skew the data minimally but these are not random values and it is possible to have such values in real world scenario hence we will keep the outlier values as it is and it won't affect out model that much.

* pdays has majority of values equal to 999 which means there was no previous contact hence it is better to eliminate this feature from our analysis as again imputing will consist biased values.

* For example age has outlier values above 73 years of age in the dataset. It is possible to have the age of a client to be 73 or above.




* Replacing "unknown" "nonexitent" cases and imputing them.
```{r}

banking_testimpute<-read.csv("bank_test.csv",sep=";",na.strings = c("unknown","nonexistent"))

#we will use the test data to reduce time complexity of model training and imputing

#numeric features
bankingtest_num<-as.data.frame(banking_testimpute[c('age','day_of_week','duration','campaign','pdays','previous','emp.var.rate','cons.price.idx','cons.conf.idx','euribor3m','nr.employed')])

bankingtest_cat<-as.data.frame(banking_testimpute[c(2,3,4,5,6,7,8,9,10,15,21)])




#imputation using mice
library(mice)
md.pattern(banking_testimpute)
banking_testimpute$y<-as.factor(ifelse(banking_testimpute$y=="no",0,1))
#install.packages("VIM")
library(VIM)
mice_plot <- aggr(banking_testimpute, col=c('navyblue','yellow'),
                    numbers=TRUE, sortVars=TRUE,
                    labels=names(banking_testimpute), cex.axis=.7,
                    gap=3, ylab=c("Missing data","Pattern"))
imputed_Data <- mice(banking_testimpute, m=5, maxit = 5, method = 'pmm', seed = 500)
model1<-complete(imputed_Data,2)



```

We have imputed the model using mice package now we will compare the performance of the model with imputed data and one without imputation

```{r}
library(caret)
set.seed(1111)
m <- train(y ~ ., data = model1, method = "C5.0")
predict_m<-predict(m,model1)
table(predict_m,model1$y)

#kappa statistic
library(vcd)
Kappa(table(predict_m,model1$y))
```

```{r}

set.seed(1112)
m1<-train(y~.,data = banking_test,method="C5.0")
predict_m1<-predict(m1,banking_test)
library(gmodels)
table(predict_m1,banking_test$y)
Kappa(table(predict_m1,banking_test$y))

```

Comparing imputed model with original model without imputation we can see that the false negative for the model without imputation is less. Hence we will use the given model for building our actual model for prediction.


* Eliminating pdays  feature
```{r}
banking_test<-banking_test[-13]
banking_test<-banking_test[-3927,]
#removing illiterate value from education feature as there is only one value

bank_test<-bank_test[-13]
bank_test<-bank_test[-3927,]

#segmenting features on their type to apply transformations for different models
bank_num<-as.data.frame(banking_test[c('age','campaign','previous','emp.var.rate','cons.price.idx','cons.conf.idx','euribor3m','nr.employed')])

bank_num1<-as.data.frame(bank_test[c('age','campaign','previous','emp.var.rate','cons.price.idx','cons.conf.idx','euribor3m','nr.employed')])

bank_cat<-as.data.frame(banking_test[c(2,3,4,5,6,7,8,9,10,13)])

bank_cat1<-as.data.frame(bank_test[c(2,3,4,5,6,7,8,9,10,13)])
y<-as.factor(banking_test$y) #target variable
y2<-as.factor(banking_test$y)
```

Normalising the numeric feature
```{r}
normalize <- function(x) {
return ((x - min(x)) / (max(x) - min(x))) } #normalizing the function with Min-Max method

bank_n <- as.data.frame(lapply(bank_num, normalize)) #normalised numeric feature

bank_n1<-as.data.frame(lapply(bank_num1, normalize))

#dummy coding the catergorical features
library(ade4)
bank_dummy<- acm.disjonctif(bank_cat) #dummy coded categorical feature

bank_dummy1<-acm.disjonctif(bank_cat1)
y1<-ifelse(banking_test$y=="yes",1,0)
y2<-as.factor(ifelse(banking_test$y=="yes",1,0))


bank_nd<-cbind(bank_n,bank_dummy,y2)
str(bank_nd)



```


* Data correlation/collinearity
```{r}
library(psych)
qn<-cbind(y1,bank_n)
pairs.panels(qn) #correlation with numeric feature
qc<-cbind(y1,bank_cat)
pairs.panels(qc) #correlation with categorical features
```
From the plots we see that there is not much correlation between the numeric features. But some features have significant amount collinearity 

employee variance rate and consumer price index- 0.75
employee variance rate and euribor- 0.97
euribor and number of employees - 0.94\

For categorical features we do not see much collinearity as it has factored values and using dummy coded variables cannot give us significant answers for that specific feature. For example there will be 10 variables for job feature as it has 10 different classes. Hence the plot won't show the correlation for "job" as a feature if we use dummy coded variables.



**PCA**

This dataset consists of categorical variables hence I have performed PCA on only numeric features of the dataset.
```{r}
q<-cbind(bank_num,y1)
pca<-princomp(q,scores = TRUE,cor=TRUE)
loadings(pca)
plot(pca)
biplot(pca)
 pca$scores[1:10,]


```



**Creating new feature column**

We will be creating two classes for age i.e. adult and senior citizen
```{r}
table(banking_test$age)
age_new<-ifelse(banking_test$age<50,"adult",banking_test$age)
age_new1<-ifelse(banking_test$age>50,"senior citizen",age_new)
age_new1

```


* Data partition


```{r}
#dataset without normalization
set.seed(1003)
train <- createDataPartition(banking_test$y, p=0.75, list = FALSE)  
banking_trainx<-banking_test[train,]
banking_validx<-banking_test[-train,]
yx<-ifelse(banking_validx$y=="yes",1,0)

#normalized values
set.seed(1002)
train1<-createDataPartition(bank_nd$y2,p=0.75,list=F)
banking_train1<-bank_nd[train1,]
banking_valid1<-bank_nd[-train1,]


prop.table(table(banking_train1$y)) 
prop.table(table(banking_valid1$y))


#confirming the split of our target variable is similar to training and validation set

#strings without factors split
set.seed(1009)
trainx<-createDataPartition(bank_test$y,p=0.75,list=F)
banking_train<-bank_test[trainx,]
banking_valid<-bank_test[-trainx,]
```
The above created dataframe consisits of 62 variables. We can use this dataframe for logistic regression and neural network model. Though logistic regression does not need dummy coding as it does the dummy coding itself.

**Logistic regression model**

```{r}
logistic_model<-glm(formula=y~ .,data=banking_trainx,family=binomial)
summary(logistic_model)

#backward fitting the model with AIC
logistic_model<-step(glm(formula=y~ .,data=banking_trainx,family=binomial),direction = "backward")

anova(logistic_model, test = 'Chisq')


prediction_logistic<-predict(logistic_model,banking_validx,type="response")
prediction_logistic<-ifelse(prediction_logistic>0.5,1,0)

table(prediction_logistic,yx)
confusionMatrix(prediction_logistic,yx)
library(ROCR)
library(Metrics)
auc_glm<-auc(yx,prediction_logistic)
auc_glm
```
Accuracy=92%
Kappa statistic=0.5422
Auc=0.74


**Neural network model**
```{r}
nn_model<-train(y~.,data=banking_trainx,method="nnet",trControl=trainControl(method='cv',number=10)) #k-fold cross validation model

nn_model1<-train(y~.,data=banking_trainx,method="nnet",trControl=trainControl(method='LGOCV')) #holdout sampling

nn_predict<-predict(nn_model,banking_validx)
table(nn_predict,banking_validx$y)
confusionMatrix(nn_predict,banking_validx$y)

nn_predict1<-predict(nn_model1,banking_validx)
table(nn_predict1,banking_valid1$y)
confusionMatrix(nn_predict1,banking_validx$y)

nn_predictx<-ifelse(nn_predict=="yes",1,0)
nn_predictx1<-ifelse(nn_predict1=="yes",1,0)

library(ROCR) 
library(Metrics)
prediction_nn<-prediction(nn_predictx,yx)
perf <- performance(prediction_nn,measure = "tpr",x.measure = "fpr") 
plot(perf) 
auc(yx,nn_predictx)


prediction_nn1<-prediction(nn_predictx1,yx)
perf1 <- performance(prediction_nn1,measure = "tpr",x.measure = "fpr") 
plot(perf1) 
auc(yx,nn_predictx1)

```
k-fold model
Accuracy=92%
Kappa statistic=0.56
auc=0.766

Hold out model
Accuracy=91.83%
Kappa stistic=0.5355
auc=0.7425


**SVM model**
```{r}


set.seed(1005)
svm_model<-train(y~.,data=banking_trainx,method="svmRadial",trControl=trainControl(method='cv',number=10))



svm_predict<-predict(svm_model,banking_validx)
table(svm_predict,banking_validx$y)
confusionMatrix(svm_predict,banking_validx$y)

```
Accuracy=90.47
Kappa= 0.3447


**Rule Learner Model**
```{r}
library(RWeka)
banking_JRip <- JRip(y ~ ., data = banking_trainx)
predict_Jrip<-predict(banking_JRip,banking_validx)
table(predict_Jrip,banking_validx$y)
confusionMatrix(predict_Jrip,banking_validx$y)




predict_Jripn<-ifelse(predict_Jrip=="yes",1,0)
prediction_Jrip<-prediction(predict_Jripn,yx)
perf <- performance(prediction_Jrip,measure = "tpr",x.measure = "fpr") 
plot(perf) 
auc_Jrip<-auc(yx,predict_Jripn)
print(auc_Jrip)
```
Accuracy=91.25%
Kappa=0.5725
auc=0.802

**KNN model**

```{r}
m_knn<-train(y~.,data=banking_trainx,method="knn",metric="Kappa",trControl=trainControl(method = "cv",number=10),tuneGrid=expand.grid(.k=c(50,55,60)))

m_knn1<-train(y2~.,data=banking_train1,method="knn",metric="Kappa",trControl=trainControl(method = "cv",number=10),tuneGrid=expand.grid(.k=c(50,55,60)))

predict_knn<-predict(m_knn,banking_validx)
table(predict_knn,banking_validx$y)
confusionMatrix(predict_knn,banking_validx$y)




predict_knn1<-ifelse(predict_knn=="yes",1,0)

auc_knn<-auc(yx,predict_knn1)


```
Accuracy=90.76
Kappa=0.4276
auc=0.677


**Regression Tree model**

```{r}
library(rpart)
banking_rpart <- rpart(y~ ., data = banking_trainx)
print(summary(banking_rpart))

predict_rpart <- predict(banking_rpart, banking_validx,type="prob")


library(rpart.plot)
rpart.plot(banking_rpart, digits = 4, fallen.leaves = TRUE,
type = 3, extra = 101)



```
**Random forest**

```{r}
ctrl_rf <- trainControl(method = "repeatedcv",
number = 10, repeats = 10)

grid_rf <- expand.grid(.mtry = c(2, 4, 8, 16))

set.seed(300)
m_rf <- train(y ~ ., data = banking_trainx, method = "rf",
metric = "Kappa", trControl = ctrl_rf,
tuneGrid = grid_rf)




predict_rf<-predict(m_rf,banking_validx)
table(predict_rf,banking_validx$y)
confusionMatrix(predict_rf,banking_validx$y)

predict_rfn<-ifelse(predict_rf=="yes",1,0)
auc_rf<-auc(yx,predict_rfn)
auc_rf
```
auc=0.718
Kappa=0.509
Accuracy=0.909

**Boosting Decision trees**

```{r}
#install.packages("adabag")
library(adabag)

bankin_boost<-boosting(y~.,data=banking_trainx)

predict_boost<-predict(bankin_boost,banking_validx)

predict_boostn<-ifelse(predict_boost$class=="yes",1,0)

confusionMatrix(predict_boostn,yx)

auc_boost<-auc(predict_boostn,yx)

```


**Ensemble models (knn,svm)**

```{r}
control <- trainControl(method="repeatedcv", number=10, repeats=3, savePredictions=TRUE, classProbs=TRUE)
algorithmList <- c('knn', 'svmRadial','rf')
set.seed(444)

install.packages("caretEnsemble")
library(caretEnsemble)
models <- caretList(y~., data=banking_trainx, trControl=control, methodList=algorithmList,metric="Kappa")
summary(models)


# stacking the models
stackControl <- trainControl(method="repeatedcv", number=10, repeats=3, savePredictions=TRUE, classProbs=TRUE)
set.seed(445)
stack.glm <- caretStack(models, method="glm", metric="Kappa", trControl=stackControl)
print(stack.glm)

predict_stack<-predict(stack.glm,banking_validx)
confusionMatrix(predict_stack,banking_validx$y)
```
Kappa=0.3934
Accuracy=0.8988
