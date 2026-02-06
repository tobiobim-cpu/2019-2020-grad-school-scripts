
#import pandas as pd
import pandas as pd

#import the grocery data
sales_data = pd.read_csv("grocery_sales.csv")

#fill in the missing values
avg_sales = sales_data["sales"].mean()
sales_data["sales"].fillna(value = avg_sales, inplace = True)

#sum sales by day
sales_summ = sales_data.groupby("transaction_date")["sales"].sum()

#plot sales over time
sales_summ.plot(rot = 45)