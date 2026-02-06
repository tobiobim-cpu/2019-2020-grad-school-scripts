##Association rules models with Apriori
    #Association Rule:
        #Is the process that deploys pattern recognition to identify and quantify relationships between different, yet related items
        #Popular use case: Product placement optimiztion at groceries stores and ecom stores. Will she buy butter or egg if we place them beside bread?
        #Feature Engineering:
           #is the process of engineering data into a predictive feature that fits the requirements(and/or improves the performance) of a ML model
    #Ways to measure association:
       #1)Support: relative freq of an item within a data(how popular is the item). It can be calc as; support(A -> C) = support(A U C)
       #2)Confidence: prob of seing the consequent item(a "then" term) within the data, given that the data also contains the antecedent(the 'if' term) item.
           #THEN[How likely is it for an item to be purchased given that], IF[another item is purchased]
           #It determines how many if-then statements are found to be true within a dataset
           #confidence(A->C) = [support(A->C) / support(A)(500/5000)], where A is the antecedent and C is the consequent . Total transaction = 5000
           #confidence(bread[500]->eggs[350]) = (150[both bread&egg purchased]/5000) / (500/5000) = 30% likehod that eggs will be bought if bread is purchased
       #3)Lift: measures how much often the antecedent and consequent ocur together rather than independently.
           #lift(A->C) = confidence(A->C) / support(C)
           #Lift scores explained:
              #lift score > 1: A is highly associated with C. If A is purchased, it is likely that C will be purchased
              #lift score < 1: If A is purchased, it is unlikely that C will be purchased.
              #lift score = 1: Indicates that there is no association btw A and C
              #Lift(bread->eggs) = 0.3 / (350/5000) = 4.28[Customers are more likely to buy eggs if they buy bread]
    #Apriori Algorithm:
       #is the algorithm that you use to implement association rule minning over structured data.

#%%
from mlxtend import frequent_patterns
import pandas as pd
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules

data = pd.read_csv(r'C:\Users\dobimuyiwa\Documents\Exercise Files\Ex_Files_Python_Data_Science_EssT_Pt2\Exercise Files\Data\groceries.csv')
#print(data.head())
##Transaction data needs to be in sparse format for association rules. we need to covert the data
basket_sets = pd.get_dummies(data)
print(basket_sets.head())

##Support Calculation
apriori(basket_sets, min_support=0.02) #35 items
#To get the name of the items that was purchased
apriori(basket_sets, min_support=0.02, use_colnames=True) #returns only single popular items purchased from the stores and their combinations>
                                                          #It doesn't really help much for marketing if we going to make upsells and cross-sells. 
                                                          # we have to reduce the support                                                      
DF = basket_sets
frequent_itemsets = apriori(basket_sets, min_support=0.002, use_colnames=True)
#create a new column that shows the length of the combination. one item = 1, two items =2
frequent_itemsets['length'] = frequent_itemsets['itemsets'].apply(lambda x: len(x)) #it will calculate the length of the itemset
print(frequent_itemsets) #filter out the one item purchases coz it's not useful for our analysis
frequent_itemsets[frequent_itemsets['length'] >= 3] #returns purchases greater than or equals to 3

###Generating Association rules
##Confidence
rules = association_rules(frequent_itemsets, metric='confidence', min_threshold=0.5)
rules.head() #row 0 -> if sausage is purchased, it is extremely likely for frankfuter to be purchased coz confidence = 1
##Lift
rules = association_rules(frequent_itemsets, metric='lift', min_threshold=1)
rules.head() #row 0&1 have the same lift but different confidence. Row 1 would be considered first coz of its higher confidence
##Lift and Confidence
rules[(rules['lift'] >= 5) & (rules['confidence'] >= 0.5)]
#Looking @ row index 779, if whole milk is purchased, then it's very likely that 'other vegetables' would be purchased in the same transaction
  #bcoz lift is 129.65 and confidence is 0.92 (high degree of confidence in this prediction)


      
# %%
