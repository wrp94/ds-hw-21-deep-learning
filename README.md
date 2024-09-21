# ds-hw-21-deep-learning

Homework for Module 21 - Deep Learning

## Overview

In this assignment, I created a deep neural network, using TensorFlow, to predict the success of projects funded by a non-profit. The goal was to create a model with an accuracy of 75% or higher. What I found was, with the data processed in the way the assignment asked, it was nearly impossible too create a model that meets the requirements. I was able to get close to the 75% accuracy goal by taking a few extra steps to remove erroneous features and bucket additional features. It would probably help to encode the `NAME` column to use as a feature. Future work should focus on including this feature in the model and increasing the size of the dataset.

## Results

### Data Preprocessing

The initial data had 12 columns; mostly `strings` with a few `int` columns.

![inital columns and datatypes](/images/init_data_types.png)

---

The target for this model is the `IS_SUCCESSFUL` column. The breakdown of the `IS_SUCCESSFUL` column is shown below.

![target distribution](/images/target_count.png)

---

Some of the columns were dropped as features since they added no meaningful predictive value to the model. These included `EIN`, `NAME`, `STATUS`, and `SPECIAL_CONSIDERATIONS`. `EIN` is a unique identifier, encoding `NAME` was beyond the scope of this project, and `STATUS` and `SPECIAL_CONSIDERATIONS` are irrelevant to the model because they are so imbalanced.

![status counts](/images/status_init_count.png)
![special consideration counts](/images/spec_considerations_init_count.png)

---

The other columns were bucketed to facilitate one-hot encoding. Here are the column names and the number of unique values in each column before bucketing:

![inital affiliation counts](/images/affiliation_init_counts.png)
![initial organization counts](/images/org_init_count.png)
![inital application type counts](/images/app_type_init_count.png)
![initial classification counts](/images/classification_init_counts.png)
![initial use case counts](/images/use_case_init_count.png)

---

After bucketing, the column names and number of unique values are shown below:

![affiliation counts](/images/affiliation_bucketed.png)
![organization counts](/images/org_bucketed.png)
![application type counts](/images/app_type_bucketed.png)
![classification counts](/images/classification_bucketed.png)
![use case counts](/images/use_case_bucketed.png)

The `ASK_AMT` column was scaled using a `MinMaxScaler` to normalize the values between 0 and 1. Finally, all of the other columns were one-hot encoded and all of the features were split into training and testing data.

---

### Compiling, Training and Evaluating the Model

The best model was the first one that achieved an accuracy of 73% and a loss of 0.55. This model had 3 layers with 75, 50, and 1 node(s) respectively. The first two layers had the `relu` activation function and the third layer had the `sigmoid` activation function.

![best model summary](/images/model1_summary.png)
![best model results](/images/model1_results.png)

I was unable to achieve 75% accuracy with any of the models. I tried adding another 25 node layer after the 50 node layer, changing the activation functions, and another 100 node layer at the beginning. I was still unable to achieve 75% accuracy.

### Conclusion

The [best nn model](models/AlphabetSoupNNModel1.h5) was the one that achieved an accuracy of 73% and a loss of 0.55. A boosted decision tree would be a better model. I was able to achieve similar results without tuning using several types of [supervised learning algorithms](notebooks/supervised_learning.ipynb).
