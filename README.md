**Association Rule Mining Projects:**
<div align = "justify">This repository contains four projects that apply association rule mining using the Apriori algorithm across different datasets. The goal of each project is to discover frequent itemsets and generate association rules to explore interesting relationships within the data. </div>

**<div align = "justify"> Overview</div>**

**<div align = "justify"> Techniques Used:</div>**

**<div align = "justify">Apriori Algorithm:** To mine frequent itemsets with a minimum support threshold. </div>
**<div align = "justify">Association Rules:** To identify strong associations between items based on the lift metric. </div>
**<div align = "justify">Data Preprocessing:** Custom parsing and cleaning of datasets as needed. </div>
**<div align = "justify">Visualization:** Bar plots to display the most frequent itemsets. </div>

**<div align = "justify">Tools & Libraries: </div>**
**<div align = "justify">pandas** for data manipulation. </div>
**<div align = "justify">mlxtend** for implementing the Apriori algorithm and generating association rules. </div>
**<div align = "justify">matplotlib** for visualizations. </div>
**<div align = "justify">collections.Counter** for frequency analysis of individual items (used in the groceries project). </div>

**<div align = "justify"> Datasets and Project Description: :</div>**

**<div align = "justify"> 1. Book Dataset:</div>**
<div align = "justify">This project analyzes book purchase data to discover frequent itemsets and association rules, identifying patterns in customers' reading preferences. </div>

**<div align = "justify"> Key Steps:</div>**
<div align = "justify">Load and preprocess the book dataset.</div>
<div align = "justify">Apply Apriori to generate frequent itemsets.</div>
<div align = "justify">Generate association rules with a focus on lift.</div>
<div align = "justify">Visualize the top 11 frequent itemsets by support.</div>

**<div align = "justify"> 2. Groceries Dataset:</div>**
<div align = "justify">A detailed analysis of grocery transaction data to uncover relationships between frequently bought items. </div>

**<div align = "justify"> Key Steps:</div>**
<div align = "justify">Parse raw transaction data and extract individual items.</div>
<div align = "justify">Perform frequency analysis to find the most purchased items.</div>
<div align = "justify">Use the Apriori algorithm to identify frequent itemsets.</div>
<div align = "justify">Generate and visualize association rules based on lift.</div>

**<div align = "justify"> 3. Movie Dataset:</div>**
<div align = "justify">This project focuses on mining frequent itemsets and association rules from a dataset of movie viewings to identify popular movie combinations. </div>

**<div align = "justify"> Key Steps:</div>**
<div align = "justify">Preprocess the dataset by removing unnecessary columns.</div>
<div align = "justify">Use Apriori to find frequent movie combinations.</div>
<div align = "justify">Generate and analyze association rules using lift.</div>
<div align = "justify">Visualize the top 9 frequent movie itemsets by support.</div>

**<div align = "justify"> 4. Phone Color Dataset:</div>**
<div align = "justify">This project identifies frequent combinations of phone colors purchased together from a phone dataset.

**<div align = "justify"> Key Steps:</div>**
<div align = "justify">Preprocess the dataset by dropping irrelevant columns.</div>
<div align = "justify">Apply Apriori to find frequent phone color combinations.</div>
<div align = "justify">Generate association rules and filter them by lift.</div>
<div align = "justify">Visualize the top 9 frequent color combinations by support.</div>

**<div align = "justify"> Results & Insights: </div>**
<div align = "justify">Across all projects, frequent itemsets were mined successfully using the Apriori algorithm, and association rules were generated to uncover interesting relationships: </div>

**<div align = "justify">Frequent Itemsets:** These represent combinations of items (books, groceries, movies, or phone colors) that occur together more often than expected by chance. <div align = "justify">

**<div align = "justify">Association Rules:** Using lift as the metric, strong rules were identified that highlight how the presence of certain items can predict the purchase of others. </div>

**Visualizations:**
<div align = "justify">Each project includes visualizations of the most frequent itemsets and a bar plot for the top combinations by support. These visual aids help in interpreting the results and understanding the patterns in the data. </div>

**<div align = "justify">Conclusion: </div>**
<div align = "justify">These projects showcase how association rule mining, particularly the Apriori algorithm, can be used to explore large datasets and uncover meaningful relationships between items. Whether analyzing books, groceries, movies, or phone colors, the same principles of association rule mining apply, providing valuable insights into customer behaviors and item associations. </div>
