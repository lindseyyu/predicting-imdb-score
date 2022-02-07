# predicting-imdb-score

## Background 
The IMDb 5000 dataset provides information about 5000 movies and their statistics on the IMDb platform. 

**Motivation:** It is common to use a movie's IMDb score to evaluate whether you want to stream or go out to see a movie. If a movie studio is trying to promote their movie on IMDb, can we provide it with a strategy that can maximize the movie's IMDb score?

**Overview of Solution:** This problem assumes new-coming studio movies, so we do not use historical data like a studio's previous movie scores. 
The types of information can be categorized into 3 groups:


*  **Numerical IMDb data**: score, # user reviews, # critic reviews, etc.
*  **Numerical/Ordinal Movie-Specific data**: budget, gross, year released, director Facebook likes, etc.  
*   **Categorical Movie-Specific data**: country, language, director name, etc. 

Due to the variable formats of the data, this problem presents the challenge of finding an intuitive and robust method to represent the features in the models. 

A linear regression, random forest regression, and gradient boosting machine are evaluated using 3-fold cross-validation to determine the feature importance when predicting IMDb score. 

You can find the code in this repository or at this [notebook link](https://colab.research.google.com/drive/1tLG9X9CWBKjdBsuMmakW6Gy9uGGYdUzx?usp=sharing).

## Numerical Columns EDA

There are 12 numerical columns in the dataset, with distributions as shown by the following histograms:
![image](https://user-images.githubusercontent.com/17552526/152823563-8a0a2197-1e3e-4764-a154-de90956e99b9.png)

Given their right-skewed nature, we will use a log-transformation before scaling and processing our data in the model.
### Imputation Methods: 
Depending on the distribution of the column data, we impute values based on the mean, median, or mode of the column

## Categorical Columns EDA

There are 12 categorical columns in the dataset, which we consolidate or drop according the frequency or similarity of values.

### Examples: 
**Genre:** We take the first genre listed in the 'genres' column (a list of all genres of the movie). Using a bar plot, we see the average IMDb score for the movies in each genre.  
![image](https://user-images.githubusercontent.com/17552526/152824767-4d13cb7c-acdd-4bcd-8f61-1b5c8dc53d20.png)

**Content Rating:** We combine similar content ratings (GP -> PG) and cast Null values to 'Not Rated." We can plot this data to see the average IMDb score for the movies in each content rating.

![image](https://user-images.githubusercontent.com/17552526/152825083-723bd461-24cf-4b39-86a9-61c10631ee76.png)

## Multicollinearity and RFE 
We look at the correlation data of the continuous variables after being log-transformed and scaled. 

![image](https://user-images.githubusercontent.com/17552526/152825443-3a29470c-05ee-4a69-b17f-0f291563502b.png)

After looking for correlations of > 75% , we also run SKLearn's Recursive Feature Elimination method to find the 20 most important features. 

## Modeling: Multiple Linear Regression, Random Forest Regression, Gradient Boosting 

After running a multiple linear regression, we explore ensemble models (Random Forest, GBM) and use Random Search and GridSearchCV to tune the hyperparamaters. The following are the model performances with their corresponding MSE and RSquared. 

![image](https://user-images.githubusercontent.com/17552526/152826278-36d15384-c7c7-43d3-a003-e9d6b73b1e71.png)


## Feature Importance

Using our best estimator from the Random Forest model, we obtain the following feature importances for imdb_score 

![image](https://user-images.githubusercontent.com/17552526/152826426-311ce337-d9cf-4a80-966d-fb42b9416db6.png)

## Takeaways

1.  **Feature Importance:** As displayed above, we can identify the importance of the Top 20 Predictors of imdb_score. The top 5 most prominenta are (in order) : Number of Voted Users, Duration, Title Year, Budget, and the Number of Critic for Reviews
2.   **Model Accuracy**: The most accurate estimator was a Random Forest Regressor with an R^2 of 0.488 and a MSE of 0.573

**Areas for Improvement**


*   **Missing Value Imputation:** More robust NaN imputation for both numerical and categorical variables technique such as KNN or Random Forest Imputation 
*   **Further Model Exploration:** Alternative modeling methods could have been explored including Support Vector Machines and the ensemble model, XGBoost 
*  **Inaccurate Genre Labeling:** Given the genre list, I parsed for the first genre listed and converted the remaining into a count of additional genres. As I'm not sure how the genres were initially ordered, it's possible that some specificity was lost by choosing only the first element. 



