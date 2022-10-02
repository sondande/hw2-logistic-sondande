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
    - **monks1.csv**
      * **Accuracy:** 77% _(0.77)_ 
    - **banknotes.csv**
      * **Accuracy:** 77% _(0.77)_
    - **occupancy.csv**_
      * **Accuracy:** 91% _(0.91)_
    - **seismic.csv**
      * **Accuracy:** 95% _(0.95)_
    
    b. Create confidence intervals for each of your test set accuracies.
    - **monks1.csv**
      * **Confidence Intervals:** [0.7303, 0.8097] 
    - **banknotes.csv**
      * **Confidence Intervals:** [0.7758, 0.7643] 
    - **occupancy.csv**
      * **Confidence Intervals:** [0.899, 0.9210] 
    - **seismic.csv**
      * **Confidence Intervals:** [0.9385, 0.9615]

1) Pick 30 different random seeds (document them in your README file). Rerun your program on each data set using the same parameters as Question 1 (0.01 learning rate, 60% training percentage, 20% validation percentage).
   
    a. What was the average accuracy you observed across all 30 seeds for each data set?
    - **monks1.csv**
       - **Random Seed**
         - **1:**
         - **2:**
         - **3:**
         - **4:**
         - **5:**
         - **6:**
         - **7:**
         - **8:**
         - **9:**
         - **10:**
         - **11:**
         - **12:**
         - **13:**
         - **14:**
         - **15:** 
         - **16:** 
         - **17:** 
         - **18:** 
         - **19:**
         - **20:**
         - **21:**
         - **22:**
         - **23:**
         - **24:**
         - **25:**
         - **26:**
         - **27:**
         - **28:**
         - **29:**
         - **30:** 0.735632183908046
    - **banknotes.csv**
       - **Random Seed**
         - **1:**
         - **2:**
         - **3:**
         - **4:**
         - **5:**
         - **6:**
         - **7:**
         - **8:**
         - **9:**
         - **10:**
         - **11:**
         - **12:**
         - **13:**
         - **14:**
         - **15:**
         - **16:**
         - **17:**
         - **18:**
         - **19:**
         - **20:**
         - **21:**
         - **22:**
         - **23:**
         - **24:**
         - **25:**
         - **26:**
         - **27:**
         - **28:**
         - **29:**
         - **30:**
      - **Average:**
    - **occupancy.csv**
       - **Random Seed**
         - **1:** 77.11% _(0.7711575875486382)_
         - **2:** 77.38% _(0.7738326848249028)_
         - **3:** 76.75% _(0.7675097276264592)_
         - **4:** 76.33% _(0.7633754863813229)_
         - **5:** 77.94% _(0.7794260700389105)_
         - **6:** 75.68% _(0.7568093385214008)_
         - **7:** 77.06% _(0.770671206225681)_
         - **8:** 76.67% _(0.7667801556420234)_
         - **9:** 76.99% _(0.7699416342412452)_
         - **10:** 76.31% _(0.7631322957198443)_
         - **11:** 77.50% _(0.7750486381322957)_
         - **12:** 75.63% _(0.7563229571984436)_
         - **13:** 76.60% _(0.7660505836575876)_
         - **14:** 77.14% _(0.7714007782101168)_
         - **15:** 75.07% _(0.7507295719844358)_
         - **16:** 77.11% _(0.7711575875486382)_
         - **17:** 77.14% _(0.7714007782101168)_
         - **18:** 76.99% _(0.7699416342412452)_
         - **19:** 77.50% _(0.7750486381322957)_
         - **20:** 77.74% _(0.7774805447470817)_
         - **21:** 77.91% _(0.7791828793774319)_
         - **22:** 76.04% _(0.7604571984435797)_
         - **23:** 76.79% _(0.7679961089494164)_
         - **24:** 78.30% _(0.7830739299610895)_
         - **25:** 76.21% _(0.7602140077821011)_
         - **26:** 77.11% _(0.7711575875486382)_
         - **27:** 75.72% _(0.757295719844358)_
         - **28:** 75.51% _(0.7551070038910506)_
         - **29:** 76.21% _(0.7621595330739299)_
         - **30:** 78.42% _(0.7842898832684825)_
       - **Average:** 95.96% _(0.95962)_ 
    - **seismic.csv**
       - **Random Seed**
         - **1:** 92.84% _(0.9284332688588007)_
         - **2:** 93.61% _(0.9361702127659575)_
         - **3:** 94.77% _(0.9477756286266924)_
         - **4:** 92.64% _(0.9264990328820116)_
         - **5:** 93.61% _(0.9361702127659575)_
         - **6:** 94.97% _(0.9497098646034816)_
         - **7:** 93.42% _(0.9342359767891683)_
         - **8:** 94.77% _(0.9477756286266924)_
         - **9:** 92.84% _(0.9284332688588007)_
         - **10:** 94.97% _(0.9497098646034816)_
         - **11:** 90.71% _(0.90715667311412)_
         - **12:** 92.84% _(0.9284332688588007)_
         - **13:** 59.57% _(0.5957446808510638)_
         - **14:** 91.87% _(0.9187620889748549)_
         - **15:** 94.77% _(0.9477756286266924)_
         - **16:** 92.84% _(0.9284332688588007)_
         - **17:** 94.00% _(0.9400386847195358)_
         - **18:** 92.06% _(0.9206963249516441)_
         - **19:** 93.81% _(0.9381044487427466)_
         - **20:** 59.76% _(0.597678916827853)_
         - **21:** 93.23% _(0.9323017408123792)_
         - **22:** 94.19% _(0.941972920696325)_
         - **23:** 93.42% _(0.9342359767891683)_
         - **24:** 93.61% _(0.9361702127659575)_
         - **25:** 92.84% _(0.9284332688588007)_
         - **26:** 92.06% _(0.9206963249516441)_
         - **27:** 94.77% _(0.9477756286266924)_
         - **28:** 63.44% _(0.6344294003868471)_
         - **29:** 62.08% _(0.620889748549323)_ 
         - **30:** 93.03% _(0.9303675048355899)_  
    b. Did this average fall inside the confidence interval you calculated for Question 1?

    c. How many of the 30 seeds produced a test set accuracy that fell within your confidence interval calculated in Question 1? Does this match your expectation, given that you calculated 95% confidence intervals in Question 1?

2) Pick 10 different random seeds (document them in your README file). For the occupancy.csv data set using a learning rate of 0.01, track the accuracy of your model on the validation set after each epoch of stochastic gradient descent (i.e., after you feed the entire training set in).
   
    a. Plot a line chart of the validation set accuracies for each epoch (the epochs go on the x-axis, and the accuracy goes on the y-axis). You can use any tool of your choice to create the line charts: Excel, R, matplotlib in Python, etc. Include your line chart as an image in your GitHub repository.

    b. What trends do you see across the 10 random seeds? How does the accuracy of validation set change over time as the epoch increases? Were there differences between the 10 seeds, or did they all produce similar results?

    c. What do these results imply? 

### 2) A short paragraph describing your experience during the assignment (what did you enjoy, what was difficult, etc.)

### 3) An estimation of how much time you spent on the assignment

### 4) An affirmation that you adhered to the honor code
   We have adhered to the honor code on this assignment. 
    - Sagana Ondande 
