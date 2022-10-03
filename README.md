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
      * **Confidence Intervals:** [0.7758, 0.7643] 
    - **occupancy.csv**
      * **Confidence Intervals:** [0.899, 0.9210] 
    - **seismic.csv**
      * **Confidence Intervals:** [0.9385, 0.9615]

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
       - **Average:** 89.12% _(0.89117)_ 
      
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
    - Sagana Ondande & Ada Ates
