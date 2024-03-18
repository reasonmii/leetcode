## PANDAS

# get the 2nd
df.head(2).tail(1)

# count unique values
len(df['col'].unique())

# change column name
df.rename({'col':'new name'}, axis=1, inplace=True)
df.rename(columns={'col':'new name'})

# sort
df.sort_values('col', ascending=False, inplace=True)

# drop columns
df.drop('col', axis=1, inplace=True)

# drop duplicates rows
df.drop_duplicates(['col'])
df.drop_duplicates(subset='col', keep='first', inplace=True) # keep the first row

# create DataFrame
pd.DataFrame({'name_{}'.format(num) : [np.NaN]})

# isin
df[~df['col1'].isin(df['col2'])]

# merge
df1.merge(df2, on='key col', how='left')
df1.merge(df2, left_on='left key col', right_on='right key col', how='inner', suffixes=['_a', '_b']) # how : left, right, inner

# group by & calculate
# reset_index : series -> DataFrame
df.groupby('col').size().reset_index(name='col') # count rows
df.groupby('col')['cal col'].min().reset_index()

# group by & calculate -> column
df['newCol'] = df.groupby('col')['cal col'].transform('count')
df['newCol'] = df.groupby('col')['cal col'].transform('min')
df['newCol'] = df.groupby('col')['cal col'].transform('max')
df['newCol'] = df.groupby('col')['cal col'].transform('mean')

# rank
df['col'].rank(method='dense', ascending=False)
df.groupby('col')['rank col'].rank(method='dense', ascending=False)

# 직전 행 값과의 차이
df['col'].diff().fillna(0)
df[(df.col.diff() == 0) & (df.num.diff().diff()==0)] # 직전 값, 전전 값 모두 차이 0

# shift
df['new col'] = df['col'].shift(1) # row 한 칸씩 밑으로 내리기

# 문자 값이 'blabla'로 시작하는 행
df[df['col'].str.startswith('blabla')]

# time
df['new col'] = df['date col'] + pd.to_timedelta(1, unit='D') # add 1 day


