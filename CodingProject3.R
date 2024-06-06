# Load the dataset from URL
df <- read.csv("https://raw.githubusercontent.com/JaySquare87/DTA395-1Rollins/main/CodeProjects/RPG.csv", header = TRUE, sep = ",")


###############################################################################

library(tidyverse)

#Prepare the data
df$Class <- as.numeric(as.factor(df$Class)) - 1
df <- df |> mutate(FBoss = ifelse(FBoss == "True", 1, 0))


# Constructing Training and testing subsets
set.seed(123)
indices <-  sample(1:nrow(df), size = 0.8 * nrow(df))
train <- df[indices,]
test <- df[-indices,]

###############################################################################

# Models from Code Project 1:

# Logistic Regression (fboss only)
logistic_regression <- function(beta, X, y) {
  
  # Calculate probabilities using logistic function
  z <- X %*% beta  # Matrix multiplication
  
  # z is the dot product of X and beta
  # beta_0 * x_0 + beta_1 * x_1 + ... + beta_p * x_p
  
  p <- exp(z) / (1 + exp(z))
  # This is the logistic function
  
  # Likelihood function. Same as above.
  likelihood <- -sum(log(y * p + (1 - y) * (1 - p)))
  
  return(likelihood)
}


# Add an intercept column to the table
train <- train |> mutate(intercept = 1)
test <- test |> mutate(intercept = 1)

# Create matrix of independent variables
X <- as.matrix(train |> dplyr::select(intercept, Armor, Weapon, Physical, Magic))

# Dependent Variable
log.y <- train$FBoss

# Initial guess for beta
init_betas <- rep(0, ncol(X))

# Calculate the optimal betas for the logistic regression model
opt_betas <- optim(init_betas, logistic_regression, X = X, y = log.y)
opt_betas$par

# Calculate the accuracy of the model
test.values <- as.matrix(test |> dplyr::select(intercept, Armor, Weapon, Physical, Magic))

# Check accuracy of the model
z <- test.values %*% opt_betas$par  

prob <- exp(z) / (1 + exp(z))

# Calculate Accuracy
logistic.accuracy <- mean(test$FBoss == ifelse(prob > 0.5, 1, 0))

# Add to data frame
accuracies <- data.frame("Model" = "Logistic Regression Model", 
                         "Accuracy (Class)" = NA, 
                         "Accuracy (FBoss)" = logistic.accuracy)
#--------------------------------------------------------------------------------------

# LDA
library(MASS)

# Calculate Priori probabilities
priori_0 <- sum(train$Class == 0) / nrow(train)
priori_1 <- sum(train$Class == 1) / nrow(train)
priori_2 <- sum(train$Class == 2) / nrow(train)
priori_3 <- sum(train$Class == 3) / nrow(train)
priori_4 <- sum(train$Class == 4) / nrow(train)
priori_5 <- sum(train$Class == 5) / nrow(train)

True.priori <- sum(train$FBoss == 1) / nrow(train)
False.priori <- sum(train$FBoss == 0) / nrow(train)

class.lda <- lda(Class ~ Armor + Weapon + Physical + Magic, data = train, 
                 prior = c(priori_0, priori_1, priori_2, priori_3, priori_4, priori_5))

fboss.lda <- lda(FBoss ~ Armor + Weapon + Physical + Magic, data = train,
                 prior = c(True.priori, False.priori))

# Predict on testing data
lda.class.predictions <- predict(class.lda, newdata=test)
lda.fboss.predictions <- predict(fboss.lda, newdata=test)


# Accuracy
lda.class.accuracy <- mean(test$Class == lda.class.predictions$class)
lda.fboss.accuracy <- mean(test$FBoss == lda.fboss.predictions$class)

accuracies <- rbind(accuracies, c("LDA", lda.class.accuracy, lda.fboss.accuracy))

# QDA
class.qda <- qda(Class ~ Armor + Weapon + Physical + Magic, data = train, 
                 prior = c(priori_0, priori_1, priori_2, priori_3, priori_4, priori_5))

fboss.qda <- qda(FBoss ~ Armor + Weapon + Physical + Magic, data = train,
                 prior = c(True.priori, False.priori))

# Predict on testing data
qda.class.predictions <- predict(class.qda, newdata=test)
qda.fboss.predictions <- predict(fboss.qda, newdata=test)


# Accuracy
qda.class.accuracy <- mean(test$Class == qda.class.predictions$class)
qda.fboss.accuracy <- mean(test$FBoss == qda.fboss.predictions$class)

accuracies <- rbind(accuracies, c("QDA", qda.class.accuracy, qda.fboss.accuracy))

##############################################################################
# Models from coding project 2
# Decision tree rpart
library(rpart)

# Construct the tree
rpart.tree.class <- rpart(Class ~ Armor + Weapon + Physical + Magic, data = train, method = "class")
rpart.tree.fboss <- rpart(FBoss ~ Armor + Weapon + Physical + Magic, data = train, method = "class")

# Predict
rpart.class.predictions <- predict(rpart.tree.class, newdata=test, type="class")
rpart.fboss.predictions <- predict(rpart.tree.fboss, newdata=test, type="class")

# Accuracy
rpart.class.accuracy <- sum(rpart.class.predictions == test$Class) / nrow(test)
rpart.fboss.accuracy <- sum(rpart.fboss.predictions == test$FBoss) / nrow(test)

accuracies <- rbind(accuracies, c("Decision Tree (rpart)", rpart.class.accuracy, rpart.fboss.accuracy))

# Decision tree C50
library(C50)

# Convert to factors for C50 algorithm
train$Class <- as.factor(train$Class)
test$Class <- as.factor(test$Class)
train$FBoss <- as.factor(train$FBoss)
test$FBoss <- as.factor(test$FBoss)

# Construct the trees
c50.tree.class <- C5.0(Class ~ Armor + Weapon + Physical + Magic, data = train)
c50.tree.fboss <- C5.0(FBoss ~ Armor + Weapon + Physical + Magic, data = train)

# Make predictions on test data
c50.class.predictions <- predict(c50.tree.class, newdata = test, type="class")
c50.fboss.predictions <- predict(c50.tree.fboss, newdata = test, type="class")

# Calculate accuracy
c50.class.accuracy <- sum(c50.class.predictions == test$Class) / nrow(test)
c50.fboss.accuracy <- sum(c50.fboss.predictions == test$FBoss) / nrow(test)

# Append accuracy to accuracy data frame
accuracies <- rbind(accuracies, c("Decision Tree (C50)", c50.class.accuracy, c50.fboss.accuracy))


# SVM
library(e1071)

# SVM to predict Class
svm.class <- svm(Class ~ Armor + Weapon + Physical + Magic, data = train, scale = TRUE)

 # tuning the model
tune.out.class <- e1071::tune(svm, Class ~ Armor + Weapon + Physical + Magic, data=train, 
                        kernel="radial", 
                        ranges=list(cost=c(0.1, 1, 10, 100, 1000, 10000), 
                                    gamma=c(0.5, 1, 2, 3, 4)), 
                        scale= TRUE)
summary(tune.out.class)

# Optimized SVM to predict class
svm.class.opt <- svm(Class ~ Armor + Weapon + Physical + Magic, data = train, cost = 0.1, gamma = 0.5, scale = TRUE)



# SVM to predict FBoss
svm.fboss <- svm(FBoss ~ Armor + Weapon + Physical + Magic, data = train, scale = TRUE)

# tuning the model
tune.out.fboss <- e1071::tune(svm, FBoss ~ Armor + Weapon + Physical + Magic, data=train, 
                              kernel="radial", 
                              ranges=list(cost=c(0.1, 1, 10, 100, 1000, 10000), 
                                          gamma=c(0.5, 1, 2, 3, 4)), 
                              scale= TRUE)
summary(tune.out.fboss)

# Optimized SVM to predict FBoss
svm.fboss.opt <- svm(FBoss ~ Armor + Weapon + Physical + Magic, data = train, cost = 10, gamma = 0.5, scale = TRUE)

# Predictions
svm.class.predictions <- predict(svm.class.opt, newdata=test, type = "class")
svm.fboss.predictions <- predict(svm.fboss.opt, newdata=test, type = "class")


svm.class.accuracy <- sum(svm.class.predictions == test$Class) / nrow(test)
svm.fboss.accuracy <- sum(svm.fboss.predictions == test$FBoss) / nrow(test)

# Append accuracy to accuracy data frame
accuracies <- rbind(accuracies, c("SVM", svm.class.accuracy, svm.fboss.accuracy))


##############################################################################
# 
# NN code Starts Here.
#
library(keras3)



# Splitting dataset into features and labels
class.features <- as.matrix(df[, c("Armor", "Weapon", "Physical", "Magic")])
class.labels <- to_categorical(df$Class)

# Normalizing features
class.mean <- apply(class.features, 2, mean)
class.std <- apply(class.features, 2, sd)
class.features <- scale(class.features, center = class.mean, scale = class.std)

# Splitting into training and testing sets
set.seed(123)
class.indices <- sample(1:nrow(class.features), size = 0.8 * nrow(class.features))
class.x_train <- class.features[class.indices,]
class.y_train <- class.labels[class.indices,]
class.x_test <- class.features[-class.indices,]
class.y_test <- class.labels[-class.indices,]


class.model <- keras_model_sequential(input_shape = 4) |>
  layer_dense(units = 16, activation = 'relu') |>
  layer_dropout(rate=.02) |>
  layer_dense(units = 6, activation = 'softmax')

class.model |> compile(
  loss = 'categorical_crossentropy',
  optimizer = optimizer_adam(learning_rate = .001), # adam is another optimizer that works well in practice
  metrics = 'accuracy'
)

# Step 4: Train the Model
class.model |> fit(class.x_train, class.y_train, epochs = 25, batch_size = 5, validation_split = 0.2)


# Step 5: Evaluate the Model
nn.class.evaluation <- class.model |> evaluate(class.x_test, class.y_test)

nn.class.accuracy <- round(nn.class.evaluation$accuracy, digits = 3)

################################################################################
# FBoss
################################################################################

# Splitting dataset into features and labels
fboss.features <- as.matrix(df[, c("Armor", "Weapon", "Physical", "Magic")])
fboss.labels <- to_categorical(df$FBoss)

# Normalizing features
fboss.mean <- apply(fboss.features, 2, mean)
fboss.std <- apply(fboss.features, 2, sd)
fboss.features <- scale(fboss.features, center = fboss.mean, scale = fboss.std)

# Splitting into training and testing sets
set.seed(123)
fboss.indices <- sample(1:nrow(fboss.features), size = 0.8 * nrow(fboss.features))
fboss.x_train <- fboss.features[fboss.indices,]
fboss.y_train <- fboss.labels[fboss.indices,]
fboss.x_test <- fboss.features[-fboss.indices,]
fboss.y_test <- fboss.labels[-fboss.indices,]


fboss.model <- keras_model_sequential(input_shape = 4) |>
  layer_dense(units = 16, activation = 'relu') |>
  layer_dropout(rate=0.2) |>
  layer_dense(units = 2, activation = 'sigmoid')

fboss.model |> compile(
  loss = 'categorical_crossentropy',
  optimizer = optimizer_adam(learning_rate = .01), # adam is another optimizer that works well in practice
  metrics = 'accuracy'
)

# Step 4: Train the Model
fboss.model |> fit(fboss.x_train, fboss.y_train, epochs = 25, batch_size = 5, validation_split = 0.2)


# Step 5: Evaluate the Model
nn.fboss.evaluation <- fboss.model |> evaluate(fboss.x_test, fboss.y_test)

nn.fboss.accuracy <- nn.fboss.evaluation$accuracy

# Append accuracy to accuracy data frame
accuracies <- rbind(accuracies, c("NN", nn.class.accuracy, nn.fboss.accuracy))

print(accuracies)


# My opinion regarding the use of a neural network on this dataset:
#
# The neural network performed well in regards to accuracy. My best run for Class classification
# registered a 73% accuracy, which is as good or better than every other model. FBoss classification
# was even better, with an accuracy of 99%-100%. SVM performed the second best, with accuracies of 72%
# and 99%, but tuning the model proved to be computationally expensive as well, though not to the extent of
# the NN. This is a very complex dataset so any model will have issues with classification. 71-73% is among
# the best of all the models we learned this year. Furthermore, I will admit I did not do too much experimentation
# on the format of the NN, only experimenting up to 4 layers with combination 32, 16, and 8 node layers. According
# to light research it is possible to perform optimization techniques such as a grid search to find better hyperparameters, 
# which could make the NN even better at determining Class. The biggest drawback to the neural network appeared to be 
# the computational expense of the training time, but even then a 15-30 second wait time for the model to be fit
# is not a problem in my eyes. So I believe the Neural Network to be a viable option for predicting Class, and 
# it appears to perform as good as and potentially better than most of the other models. I believe the SVM is the 
# runner up, though. As far as predicting FBoss goes, the NN was near perfect, but so was SVM and the Logistic regression. 
# While the near-perfect testing accuracy is definitely a benefit, he accuracy of the SVM or the Logistic regression
# models rivals the NN. Given the increased interpretability of these models over the NN, and lower runtime, I believe
# that the NN is not the best model for predicting FBoss, though it performs exceptionally well. 
# Overall, I think NN is a reliable and well performing choice of models for this dataset, however it is not 
# the only option. For Class, LDA and SVM performed nearly as well, and for FBoss, logistic regression and SVM were 
# nearly as accurate. 


