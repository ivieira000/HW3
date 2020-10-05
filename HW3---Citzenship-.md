HW3 (K-NN Classification) - Lab3
================
Isabela Vieira
10/3/2020

What I want to predict: Citzenship status of foreigners What I think
that might be useful in predicting it: Birth Place (I have a previous
bias that individuals from certain countries may have more difficulty in
getting naturalized) and Education Level (My bias is that individuals
with higher education attainment don’t have as much of a hard time
getting their
papers).

``` r
load("~/Documents/College/Fall 2020/Econometrics/R-Projects/acs2017_ny_data.RData")

#Convert the "BPL" from text to numerical values:

acs2017_ny$BirthPlace <- factor(acs2017_ny$BPL)
acs2017_ny$BirthPlaceCode <- as.numeric(acs2017_ny$BirthPlace)

#Same for "EDUCD"

acs2017_ny$Education <- factor(acs2017_ny$EDUCD)
acs2017_ny$EducationCode <- as.numeric(acs2017_ny$Education)

#Create a subset of the data that only contains foreigners:
data_NYC <- subset(acs2017_ny, (acs2017_ny$BirthPlaceCode > 100))
attach(data_NYC)
#Adds labels to the numbers according to the txt file of the data:
citzenship_f <- factor((CITIZEN), levels=c(1,2,3,4,5),labels = c("Born abroad of American parents","Naturalized citizen","Not a citizen","Not a citizen, but has received first papers","Foreign born, citizenship status not reported"))
```

Creates a function to normalize the data:

``` r
norm_varb <- function(X_in) {
  (X_in - min(X_in, na.rm = TRUE))/( max(X_in, na.rm = TRUE) - min(X_in, na.rm = TRUE) )
}
```

Fix the
Data:

``` r
#Identify NA's (I wonder what NA means within the dataset, because there is an option in which the foreign citzen chooses not to report citzenship status)
is.na(citzenship_f) <- which(citzenship_f == 0)

#Normalizes the data to avoid misclassification:
norm_educ <- norm_varb(data_NYC$EducationCode)
norm_BPL <- norm_varb(data_NYC$BirthPlaceCode)
norm_wage <-norm_varb(data_NYC$INCWAGE)
norm_INCTOT <- norm_varb(data_NYC$INCTOT)
norm_Race <- norm_varb(data_NYC$RACE)
```

Create a Dataframe to
use:

``` r
#This is the "neighbors" we will be comparing our chosen variable (citzenship) with. 
data_use_prelim <- data.frame(norm_BPL,norm_educ)
#Code for excluding N/As from our data
good_obs_data_use <- complete.cases(data_use_prelim,citzenship_f)
#Create a dataset that combines the "neighbors" we want to use and our chosen variable now free from N/As
dat_use <- subset(data_use_prelim,good_obs_data_use)
#Not sure about this line of code: (It would be nice if I could get an explanation for this) I guess this y_use will be the untouched data that we will compare our results with?
y_use <- subset(citzenship_f,good_obs_data_use)
```

Split the data: We do that to have a portion of the data to be
classified and trained (identify patterns between the level of
education, the country of origin, and the citzenship status, using those
patterns to determine the citzenship status of a random individual in
our
dataset).

``` r
set.seed(123456) #Generates a sequence of random numbers that have the same results from 1 to 6 everytime we run this code (I'm nor sure why this is necessary)
NN_obs <- sum(good_obs_data_use == 1)
select1 <- (runif(NN_obs) < 0.8)
#Extract the Train data which is 80% of our chosen dataset 
train_data <- subset(dat_use,select1)
#Extract the Test data which is the remaining 20% of our chosen dataset
test_data <- subset(dat_use,(!select1))
cl_data <- y_use[select1]
true_data <- y_use[!select1]
```

Now, we apply the K-nn Algorithm to see if we can predict the data and
use the last lines of code to check the level of accuracy (compare the
true data to the results we got from using the k-nn function).

``` r
summary(cl_data)
```

    ##               Born abroad of American parents 
    ##                                           355 
    ##                           Naturalized citizen 
    ##                                          3863 
    ##                                 Not a citizen 
    ##                                          2373 
    ##  Not a citizen, but has received first papers 
    ##                                             0 
    ## Foreign born, citizenship status not reported 
    ##                                             0

``` r
prop.table(summary(cl_data))
```

    ##               Born abroad of American parents 
    ##                                    0.05386133 
    ##                           Naturalized citizen 
    ##                                    0.58610226 
    ##                                 Not a citizen 
    ##                                    0.36003641 
    ##  Not a citizen, but has received first papers 
    ##                                    0.00000000 
    ## Foreign born, citizenship status not reported 
    ##                                    0.00000000

``` r
summary(train_data)
```

    ##     norm_BPL        norm_educ     
    ##  Min.   :0.0000   Min.   :0.0000  
    ##  1st Qu.:0.2273   1st Qu.:0.6667  
    ##  Median :0.2273   Median :0.7917  
    ##  Mean   :0.3797   Mean   :0.7241  
    ##  3rd Qu.:0.6818   3rd Qu.:0.8750  
    ##  Max.   :1.0000   Max.   :1.0000

``` r
require(class)
```

    ## Loading required package: class

    ## Warning: package 'class' was built under R version 3.6.2

``` r
#I don't understand why we run this sequence:
for (indx in seq(1, 9, by= 2)) {
 pred_citzenship <- knn(train_data, test_data, cl_data, k = indx, l = 0, prob = FALSE, use.all = TRUE)
 #Here we compare the test data with our original data to see how accurate is the prediction
num_correct_labels <- sum(pred_citzenship == true_data)
correct_rate <- num_correct_labels/length(true_data)
print(c(indx,correct_rate))
}
```

    ## [1] 1.0000000 0.6144874
    ## [1] 3.0000000 0.6120319
    ## [1] 5.0000000 0.6095764
    ## [1] 7.0000000 0.6095764
    ## [1] 9.0000000 0.6089626

``` r
print(summary(pred_citzenship))
```

    ##               Born abroad of American parents 
    ##                                            16 
    ##                           Naturalized citizen 
    ##                                          1373 
    ##                                 Not a citizen 
    ##                                           240 
    ##  Not a citizen, but has received first papers 
    ##                                             0 
    ## Foreign born, citizenship status not reported 
    ##                                             0

We can see that the level of accuracy for this test is very poor (61%
only in the first trial). There are a few issues that might’ve impacted
the result, especially in the early stages of the code where I wasn’t
sure how to work the data that was innitially stored as factors (BPL and
EDUCD). Or maybe there isn’t much of a correlation between the birth
place of an individual/ level of education attainment and the
likeability of getting naturalized. Another thing is that I excluded all
the NAs from this analysis - I wonder if the NAs are just people who
decided not to declare their nationality, which would be important for
the analysis if so. I need to do further research on that.

Nevertheless, the original code we used in class to predict whether a
person would be in a certain borough in NYC was only able to produce a
35% accuracy level, which is much less than I got running the code to
predict the citzenship status. I think that a good algorithm would
predict a datapoint with 80-90% accurancy to be at least trusted. I
think that this level of accuracy can be improved by adding more
variables to the test. Both in the exercise we did in class with the
boroughs and in this citzenship exercise I only used 2 variables to
predict the third variable (the chosen one).

Previously, I thought that adding more variables to the knn would
enhance the predictability, however I tried adding race (Chance of
getting a naturalization could be affected by racism), total income
(individuals with higher incomes can invest/buy property which helps
with getting a citzenship), and wage (individuals working legally for
firms might have greater chances of getting naturalized than individuals
working illegaly I think), and in all 3 cases the level of accuracy
slightly diminished, as below:

I’m not sure why this happens, but it might be because the variables I
chose to add are not really correlated to the citzenship status and do
not classify as well as I expected them
to.

``` r
data_use_prelim_2 <- data.frame(norm_educ,norm_BPL,norm_INCTOT, norm_wage, norm_Race )
good_obs_data_use_2 <- complete.cases(data_use_prelim_2,citzenship_f)
dat_use_2 <- subset(data_use_prelim_2,good_obs_data_use_2)
y_use_2 <- subset(citzenship_f,good_obs_data_use_2)

set.seed(12345)
NN_obs_2 <- sum(good_obs_data_use_2 == 1)
select1_2 <- (runif(NN_obs_2) < 0.8)
train_data_2 <- subset(dat_use_2,select1_2)
test_data_2 <- subset(dat_use_2,(!select1_2))
cl_data_2 <- y_use_2[select1_2]
true_data_2 <- y_use_2[!select1_2]

summary(cl_data_2)
```

    ##               Born abroad of American parents 
    ##                                           245 
    ##                           Naturalized citizen 
    ##                                          3752 
    ##                                 Not a citizen 
    ##                                          2141 
    ##  Not a citizen, but has received first papers 
    ##                                             0 
    ## Foreign born, citizenship status not reported 
    ##                                             0

``` r
prop.table(summary(cl_data_2))
```

    ##               Born abroad of American parents 
    ##                                    0.03991528 
    ##                           Naturalized citizen 
    ##                                    0.61127403 
    ##                                 Not a citizen 
    ##                                    0.34881069 
    ##  Not a citizen, but has received first papers 
    ##                                    0.00000000 
    ## Foreign born, citizenship status not reported 
    ##                                    0.00000000

``` r
summary(train_data_2)
```

    ##    norm_educ          norm_BPL       norm_INCTOT         norm_wage      
    ##  Min.   :0.04167   Min.   :0.0000   Min.   :0.000000   Min.   :0.00000  
    ##  1st Qu.:0.66667   1st Qu.:0.2273   1st Qu.:0.009847   1st Qu.:0.00000  
    ##  Median :0.83333   Median :0.2273   Median :0.028763   Median :0.02053  
    ##  Mean   :0.75788   Mean   :0.3753   Mean   :0.052892   Mean   :0.06369  
    ##  3rd Qu.:0.87500   3rd Qu.:0.6818   3rd Qu.:0.063115   3rd Qu.:0.08621  
    ##  Max.   :1.00000   Max.   :1.0000   Max.   :0.996249   Max.   :1.00000  
    ##    norm_Race     
    ##  Min.   :0.0000  
    ##  1st Qu.:0.1250  
    ##  Median :0.6250  
    ##  Mean   :0.4227  
    ##  3rd Qu.:0.6250  
    ##  Max.   :1.0000

``` r
require(class)
for (indx in seq(1, 9, by= 2)) {
 pred_citzenship_2 <- knn(train_data_2, test_data_2, cl_data_2, k = indx, l = 0, prob = FALSE, use.all = TRUE)
num_correct_labels_2 <- sum(pred_citzenship_2 == true_data_2)
correct_rate_2 <- num_correct_labels_2/length(true_data_2)
print(c(indx,correct_rate_2))
}
```

    ## [1] 1.0000000 0.5727392
    ## [1] 3.000000 0.595675
    ## [1] 5.0000000 0.6055046
    ## [1] 7.000000 0.621232
    ## [1] 9.0000000 0.6225426

``` r
print(summary(pred_citzenship_2))
```

    ##               Born abroad of American parents 
    ##                                            10 
    ##                           Naturalized citizen 
    ##                                          1037 
    ##                                 Not a citizen 
    ##                                           479 
    ##  Not a citizen, but has received first papers 
    ##                                             0 
    ## Foreign born, citizenship status not reported 
    ##                                             0
