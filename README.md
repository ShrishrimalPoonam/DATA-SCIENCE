**<div align = "justify">Naive Bayes Projects</div>**
<div align = "justify">This repository contains three projects demonstrating the application of Naive Bayes algorithms across various domains. Each project explores different datasets and preprocessing techniques, leading to building classification models using either Multinomial Naive Bayes or Gaussian Naive Bayes. </div>

**<div align = "justify">1. Disaster Tweets Classification (Multinomial Naive Bayes)</div>**

**<div align = "justify">Overview: </div>**
<div align = "justify">This project focuses on classifying tweets as related to real disasters or not. The dataset consists of tweets with binary labels indicating whether each tweet references a disaster.</div>

**<div align = "justify">Key Steps:</div>**

**<div align = "justify">Data Cleaning: </div>**
<div align = "justify">Removal of irrelevant columns (id, keyword, location). </div>
<div align = "justify">Preprocessing the text data by removing non-alphabet characters and stopwords, followed by tokenization.</div>

**<div align = "justify">Bag of Words & TF-IDF Transformation:</div>**
<div align = "justify">The text data is converted into a matrix of token counts using CountVectorizer. </div>
<div align = "justify">Term weighting is applied using TF-IDF transformation. </div>

**<div align = "justify">Modeling:</div>**
<div align = "justify">Multinomial Naive Bayes is used to classify the tweets. </div>
<div align = "justify">The model's performance is evaluated using accuracy scores and confusion matrices. </div>

**<div align = "justify">Results:</div>**
<div align = "justify">The model achieves a good balance between training and test accuracy. Adjustments using Laplace smoothing further improved test accuracy. </div>

**<div align = "justify">2. Car Purchase Prediction (Gaussian Naive Bayes) </div>**

**<div align = "justify">Overview:</div>**
<div align = "justify">This project uses demographic information (e.g., gender, age, estimated salary) to predict whether a user will purchase a car based on the given advertisement.</div>

**<div align = "justify">Key Steps:</div>**

**<div align = "justify">Data Preprocessing:</div>**
<div align = "justify">Categorical variables are converted to numerical form using one-hot encoding. </div>
<div align = "justify">Irrelevant columns such as User ID are dropped. </div>

**<div align = "justify">Train-Test Split:</div>**
<div align = "justify">The dataset is split into training and testing sets. </div>

**<div align = "justify">Modeling:</div>**
<div align = "justify">Gaussian Naive Bayes is applied to predict the likelihood of a purchase. </div>
<div align = "justify">Model performance is evaluated with the accuracy score and prediction probabilities. </div>

**<div align = "justify">Results: </div>**
<div align = "justify">The model provides reasonable accuracy on test data, with acceptable classification errors based on the confusion matrix. </div>


**<div align = "justify">3. Salary Classification (Gaussian Naive Bayes) </div>**

**<div align = "justify">Overview:</div>**
<div align = "justify">In this project, the goal is to classify whether a person's salary is greater than or less than $50K based on various demographic features. </div>

**<div align = "justify">Key Steps:</div>**

**<div align = "justify">Data Preprocessing:</div>**
<div align = "justify">Categorical columns such as work class, occupation, and marital status are label-encoded. </div>
<div align = "justify">Certain irrelevant columns are dropped for better model performance. </div>

**<div align = "justify">Modeling:</div>**
<div align = "justify">Gaussian Naive Bayes is used to classify salary levels. </div>
<div align = "justify">Model performance is evaluated using accuracy metrics and comparing predictions with actual values. </div>

**<div align = "justify">Results:</div>**
<div align = "justify">The model achieves acceptable accuracy on both training and testing data, with minor misclassification errors.</div>

<div align = "justify">Each project demonstrates the power of Naive Bayes for text classification and binary prediction tasks, offering insights into model performance and enhancements through preprocessing techniques. </div>
