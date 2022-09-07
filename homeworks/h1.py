import numpy as np
import pandas as pd


if __name__=='__main__':

    cars = pd.read_csv('../datasets/data.csv')
    # print(cars.head())
    
    # Q1 - What's the version of NumPy that you installed?
    print(f'Q1- numpy version: {np.__version__}')
    # Answer1: 1.23.2
    
    print('**' * 20)
    
    # Q2 How many records are in the dataset?
    print(f'Q2- The number of records is : {cars.shape[0]}') 
    # Answer: 11914
    
    print('**' * 20)
    
    # Q3 Who are the most frequent car manufacturers (top-3) according to the dataset?
    print(
        f"Q3- Top 3 manufacturers : \
        {cars['Make'].value_counts(ascending=False).head(3).index.to_list()}") 
    # Answer: [Chevrolet:1123,
                # Ford:881,
                # Volkswagen:809
                # ]
                
    print('**' * 20)
    
    # Q4 What's the number of unique Audi car models in the dataset?
    cond = cars['Make'] == 'Audi'
    # print(cars[['Make', 'Model']][cond])
    print(f"Q4- Unique Audi: {cars[['Make', 'Model']][cond].drop_duplicates().count().iloc[0]}")
    # Answer: 34
    
    print('**' * 20)
    
    # Q5 How many columns in the dataset have missing values?
    def missing_values(df):
        cnt = 0
        for i in df.isna().sum():
            if i != 0:
                cnt +=1
        return f'Number of columns with missing values: {cnt}'
        
    print(f'Q5- {missing_values(cars)}')        
    # Answer: 5
    
    print('**' * 20)
    
    # Q6:
    # 1 - Find the median value of "Engine Cylinders" column in the dataset.
    eng_cy_median = cars['Engine Cylinders'].median()
    print(f'Q6.1- The median value of Engine Cylinders: {eng_cy_median}')
    # Answer 6.1:  6
    
    print('**' * 20)
    
    # 2- Calculate the most frequent value of the same "Engine Cylinders".
    # print(cars['Engine Cylinders'].value_counts(dropna=False)) # 4
    most_freq_value = cars['Engine Cylinders'].mode()
    print(f'Q6.2- The most frequent value of the same Engine Cylinders: {most_freq_value.iloc[0]}')
    
    print('**' * 20)
    
    # 3- Use the fillna method to fill the missing values
    new_eng_cy = cars['Engine Cylinders'].fillna(most_freq_value).median()
    print(f'Q6- The median value: {new_eng_cy}. The median has NO changed')
    # Has it changed? NO
    
    print('**' * 20)
    
    # Q7
    # 1- Select all the "Lotus" cars from the dataset.
    lotus = cars[["Engine HP", "Engine Cylinders"]]
    # 2- Select only columns "Engine HP", "Engine Cylinders".
    lotus =lotus[cars['Make']=='Lotus']
    # 3- drop all duplicated rows
    lotus.drop_duplicates(inplace=True)
    # print(lotus)

    # 4- Get the underlying NumPy arra
    X = lotus.to_numpy()
    # print(X)
    
    # 5-matrix-matrix multiplication
    XTX = X.T.dot(X)
    # print(XTX)
    
    # 6-Create an array y
    y = [1100, 800, 750, 850, 1300, 1000, 1000, 1300, 800]
    
    # 7- # What's the value of the first element of w? 
    w = np.linalg.inv(XTX).dot(X.T).dot(y)
    print(f'Q7- Value of the first element of w:{ np.round(w[0], 4)}')
    # Answer: 4.59494481






