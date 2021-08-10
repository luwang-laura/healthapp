subset_varible=["SEQN","ECD010","ECQ020","ECQ060","ECD070A","MCQ080E","WHQ030E"]  # Filter specified variables
data=pd.read_csv(input_file_name)
data1 = data[subset_varible]
data1 = data1.dropna(axis = 0, how ='any') #Remove Empty data
data1 = data1[data1.ECD010 <= 60] ; data = data[data.ECQ060 <= 2]
data1 = data1[data1.ECD070A <= 20] ; data = data[data.WHQ030E <= 3]
data1 = data1[data1.MCQ080E <= 2] ; data = data[data.ECQ020 <= 2]#Remove abnormal data
data1.MCQ080E[data1['MCQ080E'] == 2] = 0 #Change value 2 to 0 which means overweight
print(data1.shape) #Check data dimension
data1=data1[~data1['ECD010'].isin([999])] # Remove column contains infinity value: 999
data1.head()
import sqlite3
# import cx_Oracle 'username/password@hostname:port/service_name'
# connect function opens a connection to the SQLite database file,
conn = sqlite3.connect('database.db')
#Similarly we will make connection with other databases like Oracle, DB2 etc.
print(conn)
# Drop a table name Crypto if it exists already
try:
    conn.execute('DROP TABLE IF EXISTS `tan2345` ')
except Exception as e:
    raise(e)
finally:
    print('Table dropped')

# python check whether table in sqlite exist or not，if not, then create one
def create_table():
    conn = sqlite3.connect("database.db")
    try:
        create_tb_cmd='''
       create table tan2345(SEQN integer ,ECD010 integer,ECQ020 integer,ECQ060 integer,WHQ030E integer,ECD070A integer,MCQ080E integer);
        '''
        conn.execute(create_tb_cmd)
    except:
        print("Create table failed")
        return False
    #conn.execute(insert_dt_cmd)
    conn.commit()
create_table()

conn = sqlite3.connect("database.db")
cu=conn.cursor()
data1.to_sql('tan2345',conn, if_exists='append', index=False)  # Inject the newly read data into the database table data in the form of a data frame
conn.commit()
##Read data from database
conn = sqlite3.connect("database.db")
#print(conn)
sql="SELECT * from tan2345"
newdata=pd.read_sql(sql,conn)
newdata.head()

num_list = list(newdata['ECQ020'].value_counts()) #获取列ECQ020数据
#name_list = list(newdata['ECQ020'].value_counts().index) #获取列ECQ020有哪些值
plt.axes(aspect = 1) #饼图类型
plt.pie(x = num_list, labels = num_list, autopct = '%3.1f %%') #饼图
plt.title('ECQ020' + ' Pie Chart') #设置标题
plt.show()
count_mean = newdata.mean()['ECD070A']  # 计算整体均值
mean_result = newdata.groupby('ECD010').mean()['ECD070A']  # 计算不同年龄段均值
#mean_result = mean_result[mean_result.index <= 35]  # 筛选出年龄小于等于35岁
x_data = [str(int(i)) for i in mean_result.index] ; y_data = mean_result.values

plt.plot(x_data,y_data, color = 'b', label = 'mean_result') #折线图
plt.axhline(y = count_mean, color = 'r', label = 'count_mean') #添加均值线
plt.text(38, 6.85, round(count_mean, 2), fontdict = {'size' : '20','color' : 'r'}) #添加值标签
plt.legend() #图例
plt.ylabel('ECD070A') ; plt.xlabel('ECD010') #设置x轴，y轴标签
plt.show()

# Using the elbow method to find the optimal number of clusters
SSE = []  # Store the sum of squared errors of each result
for k in range(1, 15):
    estimator = KMeans(n_clusters = k)  # Construct a clusterer
    estimator.fit(Kdata)
    SSE.append(estimator.inertia_)
X = range(1, 15)
plt.plot(X, SSE, 'o-')
#plt.xlabel('k') ; plt.ylabel('SSE')
plt.title('The Elbow Method')
plt.xlabel('Number of Clusters')
plt.ylabel('array')
plt.show()
kmodel = KMeans(n_clusters = 5) #聚5类
kmodel.fit(Kdata) #训练模型
label = pd.Series(kmodel.labels_)  # 各样本的类别
num = pd.Series(kmodel.labels_).value_counts()  # 统计各样本对应的类别的数目
center = pd.DataFrame(kmodel.cluster_centers_)  # 找出聚类中心,横变量，纵类别
Max = center.values.max() ; Min = center.values.min()
X = pd.concat([center, num], axis = 1)  # 横向连接（0是纵向），得到聚类中心对应的类别数目  <class 'pandas.core.frame.DataFrame'>
X.columns = list(Kdata.columns) + ['NUM']  # 表头加上一列
X
Kdata['Type'] = label.values
Kdata.head()
np.ravel(Kdata)
plt.scatter(Kdata.ECD010[Kdata['Type'] == 0], Kdata.ECD070A[Kdata['Type'] == 0], s = 10, c = 'red', label = '0')
plt.scatter(Kdata.ECD010[Kdata['Type'] == 1], Kdata.ECD070A[Kdata['Type'] == 1], s = 10, c = 'blue', label = '1')
plt.scatter(Kdata.ECD010[Kdata['Type'] == 2], Kdata.ECD070A[Kdata['Type'] == 2], s = 10, c = 'green', label = '2')
plt.scatter(Kdata.ECD010[Kdata['Type'] == 3], Kdata.ECD070A[Kdata['Type'] == 3], s = 10, c = 'cyan', label = '3')
plt.scatter(Kdata.ECD010[Kdata['Type'] == 4], Kdata.ECD070A[Kdata['Type'] == 4], s = 10, c = 'magenta', label = '4')
plt.legend()
plt.show()
