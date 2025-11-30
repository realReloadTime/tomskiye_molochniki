import kagglehub
import pandas as pd

df = pd.read_json('dataset/train.json')
print(df.head())

df['ID'] = df['id']
df.drop(['id'], axis=1, inplace=True)

def antisentiment(sentiment) -> int:
    if sentiment == 'negative':
        return 2
    elif sentiment == 'positive':
        return 0
    else:
        return 1

df['label'] = df['sentiment'].apply(antisentiment)
df.drop(['sentiment'], axis=1, inplace=True)

df['src'] = 'rbc.ru'

print(df.head())
df.to_csv('dataset/news_train.csv', index=False, sep='\t')