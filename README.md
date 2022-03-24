# Predicting-U.S. Public School Reading Performance
### A Classification Modeling Project

By: Alexis Kedo

This project uses test score, financial, and descriptive data (numeric and categorical) for over 72,000 public K-12 schools in the United States to attempt to predict precisely a certain school's reading proficiency rate. 

Reading Proficiency is defined as the percentage of students who are labelled as "proficient" or above on the 2020-2021 reading assessment for that state. The final model was eventually trained on a subset of 14,000 schools. 
## Research Understanding 
Since the National Assessment of Educational Progress (NAEP) release in 
October 2021 saw reading scores fall for the first time in the nation's history, researchers have been concerned on the implications the recent reliance on virtual instructions may have on student learning. 

The presence or absence of full-time virtual instruction was one of my primary features starting out on this project. My main questions going into the project were the following: 

1. Does virtual schooling have as big of a detrimental effect on learning (and, more specifically, reading performance) as many people fear? 
2. If not, can we determine the other predictive features of schools that do well when it comes to teaching students how to read? 
## Data Understanding
Test score, financial, and descriptive data of K-12 public schools came from the U.S. Department of Education's [Common Core of Data](https://nces.ed.gov/ccd/files.asp#Fiscal:1,Page:1). I also used income data from the [2019 American Community Survey, published by the U.S. Census Bureau](https://www.census.gov/programs-surveys/acs/news/data-releases/2019/release-schedule.html). 

Initial categorical features included a school's current Title I status, the presence/absence of virtual learning, and level (i.e., elementary, middle, high). Initial numerical features included the median income and population for that school's zipcode, the locality property tax level, and student-teacher ratio.

Initial exploratory analysis found that reading performance is uneven across all 50 states (see the map [here](https://nbviewer.org/github/alexiskedo/Predicting-School-Performance/blob/main/us_reading_map.html)). However, my dataset had balanced classes of higher- (greater than 50% of students proficient) vs. lower-performing (fewer than 50% of student proficient) schools: 
![matrix](https://github.com/alexiskedo/Predicting-School-Performance/blob/main/Visualizations/School%20Performance%20Bar%20Chart.png)
In addition, most schools reported that they did not implement full-virtual instruction in the 2020-2021 school year: 
![matrix](https://github.com/alexiskedo/Predicting-School-Performance/blob/main/Visualizations/Virtual%20Instruction%20Stacked%20Bar%20Chart.png)
## Methods 
I treated this as a classification problem and initially used logistic regression and majority-class prediction to predict whether a school was "lower-performing" (fewer than 50% of students on-grade level for reading) or "higher-performing (50% or more of students reading on grade-level). 

I compared several different classification algorithms: K-Nearest Neighbors, Random Forest, Adaboost, and Gradient Boosting. After some feature engineering, my most precist model was a random forest model with 70% precision on the training set and 69.6% precision on the testing set:
![matrix](https://github.com/alexiskedo/Predicting-School-Performance/blob/main/Visualizations/Final%20Random%20Forest%20Confusion%20Matrix.png)
## Results 
Ultimately, financial and location data seemed to feed my model the most valuable information about a school's performance, with many budget-related features showing up as some of the most important features in my final model: 
![matrix](https://github.com/alexiskedo/Predicting-School-Performance/blob/main/Visualizations/Top%2010%20Feature%20Importances.png)
The two most significant predictors were the **county** in which a school is located, and the average **student-teacher ratio** reported by the school. 
## Thoughts and Next Steps
Virtual learning may not be having as big of a negative impact (at least in isolation) as many people fear. A school's relative success or failure with virtual learning may instead be dependent on the existent resources of its families and the infrastructure already in place. 

Many features that ended up being the most useful for my model (for example, county, superintendent salaries, state revenue) are indicative of a certain community's level of wealth, not so much the type of instructional strategies used or the measure of how much the schools dedicate to the students themselves. 

However, a school's student-teacher ratio may carry a lot of leverage when it comes to how well its students learn to read. A possible next step would be adapting this into a linear regression problem (with reading scores as a continuous target variable), so that promising values like these can be further explored using p-values.
## Repository Structure
```
├── Visualizations (Containing all charts and graphs)
├── Data Prep & Cleaning Notebook (Data Prep & Cleaning.ipynb)
├── Exploratory Data Analysis Notebook (Exploratory Data Analysis.ipynb)
├── Modeling Notebook (Preprocessing & Modeling.ipynb)
├── README.md
├── functions.py
├── Link to Map (us_reading_map.html)
├── Presentation (Presentation Deck - Predicting Reading Performance.pdf)
```
