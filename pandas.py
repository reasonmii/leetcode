## 핵심 코드 정리

# merge
df1.merge(df2, on='key col', how='left')
df1.merge(df2, left_on='left key col', right_on='right key col', how='inner', suffixes=['_a', '_b'])

# drop duplicates
df.drop_duplicates(['col'])

# count unique values
len(df['col'].unique())

# sort
df.sort_values('col', ascending=False, inplace=True)

# drop columns
df.drop('col', axis=1, inplace=True)

# change column name
df.rename({'col':'new name'}, axis=1, inplace=True)
df.rename(columns={'col':'new name'})

# get the 2nd
df.head(2).tail(1)

# create DataFrame
pd.DataFrame({'name_{}'.format(num) : [np.NaN]})

# rank
df['col'].rank(method='dense', ascending=False)

# 직전 값, 전전 값 모두 차이 0
df[(df.col.diff() == 0) & (df.num.diff().diff()==0)]

# group by
# reset_index : series -> DataFrame
df.groupby('col').size().reset_index(name='col')
