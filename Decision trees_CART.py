##Decision trees with classification and regression
   #A decision tree is a decision-support tool that models decisions in order to predict probable outcomes of those decisions
       #3-types of Nodes:
        #1)Root Node: comprises the entire population or sample
        #2)Decision nodes: sub-nodes split into further sub-node
        #3)Leaf nodes: End node or Base nodes
    #Decision Tree Algorithms:
        #a class of supervised ML methods that useful for making predictions from nonlinear data
        #appropriate for CONTINUOUS input/output and CATEGORICAL input/output
    #Types:
       #CATEGORICAL VDT: when you use a decision tree to predict for a categorical target variable. also known as classification trees
          #Uses SSE to calcu the loss function to identify the best split (for line 24)
          #Use when target is binary, categorical variable
          #Output values from terminal nodes rep the mode response and values of the new data pt will be predicted from that mode
       #CONTINUOUS VDT: when you use DT to predict for a continuous target variable. also known as regression trees
          #Uses GINI index to calc the loss func to indentify the best split (for line 24)
          #Use when target is continuous and linear relationship btw features and target
          #Output from terminal nodes rep the mean response and values of new data points will be predicted from that mean
    #Benefits:
       #1)They are nonlinear models and can fit the data even if it's non-linear
       #2)Easy to implement and interpret. It mirrors human thinking
       #3)It can be easily represented graphically, helping in interpretation
       #4)Decision trees require less data preparation.
    #Disadvantges:
       #1)Very non-robust
       #2)Sensitive to training data
       #3)Global optimum tree not guaranteed
    #Assumptions:
       #1)Root node = Entire training set
       #2)Predictive features are either categorical, or (if continuous) they aare binned prior to model deployment
       #3)Rows in the dataset have a recursive distribution based on the values of attributes 
    #Recursive Binary Splitting:
       #process used to segment the predictor space into regions in order to create a binary decision tree
       #splitting stops when the user-defined criteria is met
       #Approach is TOP-DOWN, GREEDY
##TREE PRUNING:
    #process used to overcome model overfitting by removing subnodes of a decision tree(i.e. replacing a whole subtree by a leaf node)
    #To avoid bad performance nd overfitting. its necessary if expected error rate(in subtree) > Single leaf
  #Prunning methods:
     #1) Hold-out test and 2) Cost-complexity prunning

