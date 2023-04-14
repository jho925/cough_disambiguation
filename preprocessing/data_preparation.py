import pandas as pd
import random
from sklearn.model_selection import train_test_split


def trim(csv,trim_csv):
    df = pd.read_csv(csv)
    data = list(set(df['id'].tolist()))

    df2 = pd.DataFrame(
        {'id': [],
         'path': [],
         'label': [],
        })

    for patient in data:
        df1 = df.loc[df['id'] == patient]
        coughs = 0
        speech = 0 
        for index, row in df1.iterrows():
            if row['label'] == 'cough':
                coughs += 1
            else:
                speech += 1
        
        if coughs >= 25 and speech >= 15:
            df2 = pd.concat([df2,df1])  

    df2.to_csv(trim_csv,index=False)

def trim_speech(csv,trim_csv):
    df = pd.read_csv(csv)
    df1 = df.loc[df['label'] == 'cough']    
    df1.to_csv(trim_csv,index=False)

def add_speech(csv,add_csv,fin_csv):
    df = pd.read_csv(add_csv)
    df1 = df.loc[df['label'] == 'speech']  
    df2 = pd.read_csv(csv)
    df3 = pd.concat([df2,df1])

    df3.to_csv(fin_csv,index=False)


def patient_split(csv,test_csv,train_csv):
    df = pd.read_csv(csv)
    data = list(set(df['id'].tolist()))

    test_df = pd.DataFrame(
        {'id': [],
         'path': [],
         'label': [],
        })

    train_df = pd.DataFrame(
        {'id': [],
         'path': [],
         'label': [],
        })

    for patient in data:
        df1 = df.loc[df['id'] == patient]
        train, test = train_test_split(df1, test_size=0.2, random_state=0)
        train_df = pd.concat([train_df,train])
        test_df = pd.concat([test_df,test])
        
    
    train_df.to_csv(train_csv,index=False)
    test_df.to_csv(test_csv,index=False)

def main():
   add_speech('flusense_train_coughs.csv','flusense_trimmed.csv','flusense_train.csv')

if __name__ == '__main__':
    main()