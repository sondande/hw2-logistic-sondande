[![Open in Visual Studio Code](https://classroom.github.com/assets/open-in-vscode-c66648af7eb3fe8bc4f294546bfd86ef473780cde1dea487d3c4ff354943c9ae.svg)](https://classroom.github.com/online_ide?assignment_repo_id=8608742&assignment_repo_type=AssignmentRepo)
# hw2-logistic
HW2: Logistic Regression

This assignment contains four data sets that are based on four publicly available benchmarks, each representing a binary classification task:

1. monks1.csv: A data set describing two classes of robots using all nominal attributes and a binary label.  This data set has a simple rule set for determining the label: if head_shape = body_shape ∨ jacket_color = red, then yes (1), else no (0). Each of the attributes in the monks1 data set are nominal.  Monks1 was one of the first machine learning challenge problems (http://www.mli.gmu.edu/papers/91-95/91-28.pdf).  This data set comes from the UCI Machine Learning Repository:  http://archive.ics.uci.edu/ml/datasets/MONK%27s+Problems

2. banknotes.csv: A data set describing observed measurements about banknotes (i.e., cash) under an industrial print inspection camera.  The task in this data set is to predict whether a given bank note is authentic or a forgery.  The four attributes are each continuous measurements.  The label is 0 if the note is authentic, and 1 if it is a forgery. This data set comes the UCI Machine Learning Repository: https://archive.ics.uci.edu/ml/datasets/banknote+authentication

3. occupancy.csv: A data set of measurements describing a room in a building for a Smart Home application.  The task in this data set is to predict whether or not the room is occupied by people.  Each of the five attributes are continuous measurements.  The label is 0 if the room is unoccupied, and a 1 if it is occupied by a person.  This data set comes the UCI Machine Learning Repository: https://archive.ics.uci.edu/ml/datasets/Occupancy+Detection+

4. seismic.csv: A data set of measurements describing seismic activity in the earth, measured from a wall in a Polish coal mine.  The task in this data set is to predict whether there will be a high energy seismic event within the next 8 hours.  The 18 attributes have a mix of types of values: 4 are nominal attributes, and the other 14 are continuous.  The label is a 0 if there was no high energy seismic event in the next 8 hours, and a 1 if there was such an event.  This data set comes the UCI Machine Learning Repository: https://archive.ics.uci.edu/ml/datasets/seismic-bumps

## README Questions:
### 1) Research Questions: 
1) Pick a single random seed (your choice, document in your README file), 0.01 as the learning rate, 60% as the training percentage, and 20% as the validation percentage.    
    
    a. Record the accuracies of your program on the test set for each of the four data
sets.
   - **Random Seed:** 12345
     - **monks1.csv**
       * **Accuracy:** 77% _(0.77)_ 
     - **banknotes.csv**
       * **Accuracy:** 95% _(0.95)_
     - **occupancy.csv**_
       * **Accuracy:** 98% _(0.98)_
     - **seismic.csv**
       * **Accuracy:** 92% _(0.92)_
    
    b. Create confidence intervals for each of your test set accuracies.
    - **monks1.csv**
      * **Confidence Intervals:** [0.7303, 0.8097] 
    - **banknotes.csv**
      * **Confidence Intervals:** [0.9385, 0.9615] 
    - **occupancy.csv**
      * **Confidence Intervals:** [0.9781, 0.9819] 
    - **seismic.csv**
      * **Confidence Intervals:** [0.9095, 0.9305]

1) Pick 30 different random seeds (document them in your README file). Rerun your program on each data set using the same parameters as Question 1 (0.01 learning rate, 60% training percentage, 20% validation percentage).

    a. What was the average accuracy you observed across all 30 seeds for each data set?
    - **monks1.csv**
       - **Random Seed**
       - **1:** 70.11% _(0.7011)_
       - **2:** 73.56% _(0.7356)_
       - **3:** 73.56% _(0.7356)_
       - **4:** 77.01% _(0.7701)_
       - **5:** 73.56% _(0.7356)_
       - **6:** 74.71% _(0.7471)_
       - **7:** 71.26% _(0.7126)_
       - **8:** 70.11% _(0.7011)_
       - **9:** 70.11% _(0.7011)_
       - **10:** 75.86% _(0.7586)_
       - **11:** 80.56% _(0.8056)_
       - **12:** 79.31% _(0.7931)_
       - **13:** 78.16% _(0.7816)_
       - **14:** 77.01% _(0.7701)_
       - **15:** 74.71% _(0.7471)_
       - **16:** 74.71% _(0.7471)_
       - **17:** 64.36% _(0.6436)_
       - **18:** 73.56% _(0.7356)_
       - **19:** 79.31% _(0.7931)_
       - **20:** 71.26%- _(0.7126)_
       - **21:** 75.86% _(0.7586)_
       - **22:** 78.16% _(0.7816)_
       - **23:** 71.26% _(0.7126)_
       - **24:** 74.71% _(0.7471)_
       - **25:** 78.16% _(0.7816)_
       - **26:** 75.86% _(0.7586)_
       - **27:** 83.91% _(0.8391)_
       - **28:** 78.16% _(0.7816)_
       - **29:** 74.71% _(0.7471)_
       - **30:** 73.56% _(0.7356)_
    - **Average:** 74.91% _(0.74905)_

    - **banknotes.csv**
       - **Random Seed**
         - **1:** 93.81% _(0.9381)_
         - **2:** 95.64% _(0.9564)_
         - **3:** 94.91% _(0.9491)_
         - **4:** 98.18% _(0.9818)_
         - **5:** 94.54% _(0.9454)_
         - **6:** 96.36% _(0.9636)_
         - **7:** 91.64% _(0.9164)_
         - **8:** 96.36% _(0.9636)_
         - **9:** 98.54% _(0.9854)_
         - **10:** 94.54% _(0.9454)_
         - **11:** 97.82% _(0.9782)_
         - **12:** 94.91% _(0.9491)_
         - **13:** 96.36% _(0.9636)_
         - **14:** 96.00% _(0.9600)_
         - **15:** 96.73% _(0.9673)_
         - **16:** 94.91% _(0.9491)_
         - **17:** 96.73% _(0.9673)_
         - **18:** 96.73% _(0.9673)_
         - **19:** 97.45% _(0.9745)_
         - **20:** 96.00% _(0.9600)_
         - **21:** 96.36% _(0.9636)_
         - **22:** 94.91% _(0.9491)_
         - **23:** 95.27% _(0.9527)_
         - **24:** 97.45% _(0.9745)_
         - **25:** 93.81% _(0.9381)_
         - **26:** 95.27% _(0.9527)_
         - **27:** 96.36% _(0.9636)_
         - **28:** 97.82% _(0.9782)_
         - **29:** 97.09% _(0.9709)_
         - **30:** 96.36% _(0.9636)_
      - **Average:** 95.96% _(0.95962)_
  
    - **occupancy.csv**
       - **Random Seed**
         - **1:** 98.54% _(0.9854085603112841)_
         - **2:** 96.47% _(0.9647373540856031)_
         - **3:** 99.03% _(0.9902723735408561)_
         - **4:** 98.08% _(0.9880836575875487)_
         - **5:** 98.08% _(0.9880836575875487)_
         - **6:** 98.66% _(0.9866245136186771)_
         - **7:** 98.88% _(0.9888132295719845)_
         - **8:** 98.74% _(0.9873540856031129)_
         - **9:** 98.71% _(0.9871108949416343)_
         - **10:** 99.00% _(0.9900291828793775)_
         - **11:** 95.36% _(0.9535505836575876)_
         - **12:** 98.95% _(0.9895428015564203)_
         - **13:** 99.10% _(0.9910019455252919)_
         - **14:** 95.60% _(0.9559824902723736)_
         - **15:** 98.78% _(0.9878404669260701)_
         - **16:** 98.88% _(0.9888132295719845)_
         - **17:** 96.21% _(0.9620622568093385)_
         - **18:** 95.45% _(0.954523346303502)_
         - **19:** 96.86% _(0.9686284046692607)_
         - **20:** 98.10% _(0.9810311284046692)_
         - **21:** 98.97% _(0.9897859922178989)_
         - **22:** 98.74% _(0.9873540856031129)_
         - **23:** 98.93% _(0.9892996108949417)_
         - **24:** 97.88% _(0.9788424124513618)_
         - **25:** 98.61% _(0.9861381322957199)_
         - **26:** 98.91% _(0.9890564202334631)_
         - **27:** 96.64% _(0.9664396887159533)_
         - **28:** 98.83% _(0.9883268482490273)_
         - **29:** 98.78% _(0.9878404669260701)_
         - **30:** 94.72% _(0.9472276264591439)_
       - **Average:** 97.99% _(0.9799935149)_

    - **seismic.csv**
       - **Random Seed**
         - **1:** 92.84% _(0.9284332688588007)_
         - **2:** 93.03% _(0.9303675048355899)_
         - **3:** 93.81% _(0.9381044487427466)_
         - **4:** 92.26% _(0.9226305609284333)_
         - **5:** 93.61% _(0.9361702127659575)_
         - **6:** 94.19% _(0.941972920696325)_
         - **7:** 93.23% _(0.9323017408123792)_
         - **8:** 94.39% _(0.9439071566731141)_
         - **9:** 92.64% _(0.9264990328820116)_
         - **10:** 95.35% _(0.9535783365570599)_
         - **11:** 90.71% _(0.90715667311412)_
         - **12:** 92.64% _(0.9264990328820116)_
         - **13:** 92.64% _(0.9264990328820116)_
         - **14:** 92.26% _(0.9226305609284333)_
         - **15:** 94.58% _(0.9458413926499033)_
         - **16:** 92.45% _(0.9245647969052224)_
         - **17:** 94.00% _(0.9400386847195358)_
         - **18:** 92.06% _(0.9206963249516441)_
         - **19:** 93.23% _(0.9323017408123792)_
         - **20:** 93.03% _(0.9303675048355899)_
         - **21:** 92.64% _(0.9264990328820116)_
         - **22:** 93.81% _(0.9381044487427466)_
         - **23:** 92.64% _(0.9264990328820116)_
         - **24:** 94.00% _(0.9400386847195358)_
         - **25:** 92.84% _(0.9284332688588007)_
         - **26:** 92.06% _(0.9206963249516441)_
         - **27:** 94.39% _(0.9439071566731141)_
         - **28:** 94.19% _(0.941972920696325)_
         - **29:** 93.03% _(0.9303675048355899)_ 
         - **30:** 92.84% _(0.9284332688588007)_  
       - **Average:** 93.18% _(0.9318504191)_ 
      
    b. Did this average fall inside the confidence interval you calculated for Question 1?

All the value fall in the confidence intervals calculated in part 1 besides the average of seismic.csv. This one is above the confidence interval calculated slightly.

c. How many of the 30 seeds produced a test set accuracy that fell within your confidence interval calculated in Question 1? Does this match your expectation, given that you calculated 95% confidence intervals in Question 1?
In **monks1.csv:** 23 of the seeds fall inside of the confidence interval.
In **banknotes.csv:** 11 of the seeds fall inside of the confidence interval.
In **occupancy.csv:** 3 of the seeds fall inside of the confidence interval.
In **seismic.csv:** 22 of the seeds fall inside of the confidence interval.

2) Pick 10 different random seeds (document them in your README file). For the occupancy.csv data set using a learning rate of 0.01, track the accuracy of your model on the validation set after each epoch of stochastic gradient descent (i.e., after you feed the entire training set in).
   
a. Plot a line chart of the validation set accuracies for each epoch (the epochs go on the x-axis, and the accuracy goes on the y-axis). You can use any tool of your choice to create the line charts: Excel, R, matplotlib in Python, etc. Include your line chart as an image in your GitHub repository.
   
The random seeds we picked are as follows: 1, 5, 10, 25, 29, 30, 50, 250, 500, 1000. The respective graphs could be found under the file called "Graphs for Question 3." 
 
b.   What trends do you see across the 10 random seeds? How does the accuracy of validation set change over time as the epoch increases? Were there differences between the 10 seeds, or did they all produce similar results?

Across the 10 random seeds it looks like the accuracy is getting more consistent with each increasing random seed. For example, in graph 1, we observe a sudden increase in accuracy around epoch 25, it reaches 0.965 (96.5%), but then we see a decrease and then leveling off at around 0.959 for the rest of the epochs, which is 500 and at times 400 (when accuracy is closer to 1). So there are differently differences between the 10 seeds, especially between lower and higher values of seeds. As the randomSeed increases, (50,250,500,1000), the results seem closer to each other with slight differences in accuracy and number of epochs.

c. What do these results imply? 

When randomSeed is 1, the data is not shuffled much. Therefore, the reason behind this weird accuracy result could be due to this chosen value of randomSeed. However, with the increase of randomSeed to 5, then 10, the accuracy values increase and the number of epochs decrease. When we increase randomSeed even more, it seems that there are times where the accuracy is almost 1, which means it's nearly perfect. We also see a decrease in number of epochs as the accuracy is closer to 1. After randomSeed is at least 10, the accuracy is at least 0.96, most of the time even more and accuracy seems to level off at around epoch 50. Compared to lower number of randomSeeds, 50 epochs is a higher number of epochs but the model reaches better accuracy levels. It seems that the larger the randomSeed is, the more shuffled the dataset is, therefore it takes longer for the model to learn, yet it learns better!

### 2) A short paragraph describing your experience during the assignment (what did you enjoy, what was difficult, etc.)

Ada~

Personally, I enjoyed working on this lab because as we learn about neural networks in class, it makes more sense, I can visualize what happens in behind the scenes. I also really enjoyed the analysis part of this lab, seeing the model compare itself to the validation set of the dataset and to see this get plotted was very helpful for my understanding of the topic. I was unclear about a few things on the details of logistic regression but implementing them cleared those up! I think it was difficult to understand what was happening when we kept getting a value error specifically for seismic.csv with randomSeed 18, but the model worked fine with seeds 17 or 19 of the same data set. We realized there was a problem with our normalization, and not only it fixed the error but also our accuracies went up. What debugging can do is amazing.

Sagana~

I enjoyed working on the lab. I thought it was interesting to learn about how stats played a role in machine learning, but also how it can be used in classification problems. I thought the process of learning how to do a ML pipeline through the data preprocessing, utilization of Pandas for the data preprocessing, and how to train a model through the use of a training set and a validation set. The overall process facinated me and I'm excited to learn more about how we can use the model for this lab in the next one with neural networks. Some of the difficulties was noticing some small mistakes that played a crucial role in the inaccuracies of the model. Example of this was not dropping one of the One-Hot Encoding columns as we found the accuracies were better when we dropped one of the columns verses having one for all them. We found once we dropped the column, the accuracies for datasets with nomial values grew. We also confirmed that normalization, we noticed it wasn't normalizing properly and saw how the accuracies were effected. Through this, I found it interesting to see how small mistakes played a huge factor in the accuracies of the model.


### 3) An estimation of how much time you spent on the assignment

We estimate it was around 15 hours, including the research questions and their write-ups. But maybe more? Keeping track of time is hard.

### 4) An affirmation that you adhered to the honor code
   We have adhered to the honor code on this assignment. 
    - Sagana Ondande & Ada Ates
