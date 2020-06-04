knitr::opts_chunk$set(echo = TRUE)
library(data.table) 
library(ranger)
library(xgboost)
library(tidyverse) # importing, cleaning, visualising 
library(tidytext) # working with text
set.seed(22)

# Preparation

## Data cleaning and formatting
members <- fread("D:/music/members.csv", verbose=FALSE ) 
songs <- fread("D:/music/songs.csv", verbose=FALSE)
training <- fread("D:/music/train.csv", verbose=FALSE)
testing <- fread("D:/music/test.csv", verbose=FALSE)

date <- function(i){
  ymd<- as.character(i)
  paste0(substr(ymd, 1, 4), "-",  substr(ymd, 5, 6), "-",substr(ymd, 7, 8))}

members[, regyear := as.integer(substr(date(registration_init_time), 1, 4))]
members[, regmonth := as.integer(substr( date(registration_init_time), 6,7))]
members[, expyear := as.integer(substr(date(expiration_date), 1, 4))]
members[, expmonth := as.integer(substr(date(expiration_date), 6,7))]
members[, registration_init_time := as.Date(date(registration_init_time))]
members[, expiration_date := as.Date(date(expiration_date))]

M <- with(songs, union(artist_name, union(composer, lyricist)))

songs[, ":=" (artist_name= match(artist_name, M), composer = match(composer, M), lyricist = match(lyricist, M))]
songs[, singers := as.integer(1+ ifelse( composer == "" | artist_name == composer, 1, 0 )
                              + ifelse( lyricist == "" | artist_name == lyricist, 1, 0 ))]
M<- songs[, .(nrArtistSongs = .N), by= artist_name]

songs<- merge(songs, M, by= "artist_name", all.x=TRUE)
songs[ is.na(songs$language), language := -1] 
songs[, language := as.integer(language)]

training [, id := 1:nrow(training)]
testing [, target := -1]
combined <- rbind(training, testing)

#Merge combined with songs and members
combined <- merge(combined, members, by = "msno", all.x=TRUE)
combined <- merge(combined, songs, by = "song_id", all.x=TRUE)

for (i in names(combined)){
  if(class(combined[[i]]) == "character"){
    combined[is.na(combined[[i]]), eval(i) := ""]
    combined[, eval(i) := as.integer(
      as.factor(combined[[i]]))]
  } 
  else combined[is.na(combined[[i]]), eval(i) := -1]
}

combined[, registration_init_time := julian(registration_init_time)]
combined[, expiration_date := julian(expiration_date)]
combined[, length_membership := expiration_date - registration_init_time]

setDF(combined)
train_1 <- combined[combined$target != -1,]
test_1 <- combined[combined$target == -1,]
train_id <- train_1$id
train_1$id <- NULL
test_1$target <- NULL
y<- train_1$target
test_id <- test_1$id
train_1$target <- NULL
test_1$id <- NULL

#delete these original variables
rm(training); rm(testing);rm(combined); rm(songs); rm(members); rm(M); gc()

train_1<- train_1[order(train_id), ]
y<- y[order(train_id)]
test_1 <- test_1[order(test_id), ]

## Auxiliary fucntions
aucnum <- function(y_true, probs) {
  N <- length(y_true)
  if (length(probs) != N)
    return (NULL) # error
  if (is.factor(y_true)) 
    y_true <- as.numeric(as.character(y_true))
  roc <- y_true[order(probs, decreasing = FALSE)]
  stack_x = cumsum(roc == 1) / sum(roc == 1)
  stack_y = cumsum(roc == 0) / sum(roc == 0)
  auc = sum((stack_x[2:N] - stack_x[1:(N - 1)]) * stack_y[2:N])
  return(auc)
}
auc <- function(a,p) aucnum(a,p)

# mean logloss
meanll <- function(actual, probs){
  probs <- ifelse(probs >0, probs, 10^-10)
  return ( mean(Metrics::ll(actual, probs)))
}

# root mean squared error
rmse <- function(actual, prediction) sqrt(mean((actual-prediction)^2))

#binary cross_entropy
bce <- function(actual, probs){
  probs <- ifelse(probs >0, probs, 10^-10)
  return ( - mean(actual* log(probs)))
}

# accuracy
accuracy <- function(actual, probs, theta=0.5){
  probs <- ifelse(probs > theta, 1, 0)
  return(mean(probs == actual))
}

diagnosis <- function(actual, probs, title=""){
  cat("\nSummary results for", title
      , "\nauc:", auc(actual, probs)
      , "\nacc:", accuracy(actual, probs)
      , "\nbce:", bce(actual, probs)
      , "\nmll:", meanll(actual, probs)
      , "\nrmse:", rmse(actual, probs)
      , "\n"
  )
}

# primitive (0,1) calibration
primitive <- function(r) {
  r <- r - min(r)
  return(r / max(r))
}

#Split off the ensemble evalation set. 
a <- 0.2
h<- sample(nrow(train_1), a*nrow(train_1))
#h<- sample(nrow(train_1))
ees <- train_1[h, ]
y_ees <- y[h]
train_1 <- train_1[-h, ]
y <- y[-h]

# now we can scale the columns
scale_column <- names(train_1)
for (i in scale_column){
  mm<- mean(train_1[[i]])
  ss<- sd(train_1[[i]])
  train_1[[i]] <- (train_1[[i]] -mm)/ss
  test_1[[i]] <- (test_1[[i]] -mm)/ss
  ees[[i]] <- (ees[[i]] -mm)/ss
}

subs <- sample(nrow(train_1), 0.8 *nrow(train_1))


#Predictions

## Faster random forest with ranger

model_rf <- ranger(y[subs] ~ . , data = train_1[subs,], num.trees = 12, verbose= FALSE)

pred_rf<-predict(model_rf, ees, type = "response")
pred_rf <- pred_rf$predictions
pred_rft<-predict(model_rf, test_1, type = "response")
pred_rft <- pred_rft$predictions

diagnosis(y_ees, pred_rf, title="ranger")

## xgboost

param = list(
  objective="binary:logistic",
  eval_metric= "auc",
  subsample= 0.9,
  colsample_bytree=0.4, 
  max_depth= 10,
  min_child= 6,
  tree_method= "approx", 
  eta  = 0.9 , 
  nthreads = 8
)

x_train <- xgb.DMatrix(
  as.matrix(train_1),
  label = y, 
  missing=-1)
x_val <- xgb.DMatrix(
  as.matrix(ees), 
  label = y_ees, missing=-1)
x_test <- xgb.DMatrix(as.matrix(test_1), missing= -1)

model <- xgb.train(
  data = x_train,
  nrounds = 150, 
  params = param,  
  maximize= T,
  watchlist = list(val = x_val),
  print_every_n = 5
)

pred_xgb <- predict(model, x_val)
pred_xgbt  <- predict(model, x_test) 
diagnosis(y_ees, pred_xgb, title="xgb")



