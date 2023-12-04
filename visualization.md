数据统计分析
基于python和第三方库进行数据处理和分析，主要使用pandas、plotly、matplotlib等库，具体案例：

电子产品（手机）销售分析：
（1）不同内存下的销量(代码片段)
nei_cun = color_size["Number_GB"].value_counts().reset_index()
nei_cun.columns = ["Number_of_GB","Count"]  # 重命名
nei_cun["Number_of_GB"] = nei_cun["Number_of_GB"].apply(lambda x: str(x) + "GB")

fig = px.pie(nei_cun,
             values="Count",
             names="Number_of_GB")
            fig.show()

（2）不同闪存Ram下的价格分布（代码片段）

fig = px.box(df, y="Sale Price",color="Ram")

fig.update_layout(height=600, width=800, showlegend=False)

fig.update_layout(
    title={ "text":'不同<b>闪存</b>下的价格分布', 
            "y":0.96,  
            "x":0.5,  
            "xanchor":"center",  
            "yanchor":"top"  
          },

    xaxis_tickfont_size=12,   
    yaxis=dict(
        title='Distribution',  
        titlefont_size=16,  
        tickfont_size=12,  
    ),
    legend=dict(
        x=0,  
        y=1,
        bgcolor='rgba(255, 255, 255, 0)',  
        bordercolor='rgba(2, 255, 255, 0)'   
    )
)

fig.show()
<img width="686" alt="image" src="https://github.com/lulumengyi/small-tools-python/assets/32284131/0ec1f12c-6c92-4ac3-b990-e4801d876d7b">



7万条餐饮数据分析
fig = px.bar(df2_top3,x="行政区",y="店铺数量",color="类别",text="店铺数量")
fig.update_layout(title="不同行政区下不同类别的店铺数量对比")
fig.show()
复制

不同店铺下的点评数量对比：
![1441701705875_ pic](https://github.com/lulumengyi/small-tools-python/assets/32284131/0e860a90-66f6-4f8b-95f7-5a8019a634ff)

4个指标的关系：口味、环境、服务和人均消费

![1451701705891_ pic](https://github.com/lulumengyi/small-tools-python/assets/32284131/6fa15eb1-11b0-401f-a6f5-cdba5e0be81b)

![1461701705902_ pic](https://github.com/lulumengyi/small-tools-python/assets/32284131/06ac3633-c9f0-40cf-98fb-8b9c6d486d16)

基于python实现RFM模型（用户画像）
RFM模型是客户关系管理（CRM）中的一种重要分析模型，用于衡量客户价值和客户创利能力。该模型通过以下三个指标来评估客户的价值和发展潜力：

近期购买行为（R）：指的是客户最近一次购买的时间间隔。这个指标可以反映客户的活跃程度和购买意向，进而判断客户的质量和潜在价值。
购买的总体频率（F）：指的是客户在一定时间内购买商品的次数。这个指标可以反映客户对品牌的忠诚度和消费习惯，进而判断客户的潜力和价值。
花了多少钱（M）：指的是客户在一定时间内购买商品的总金额。这个指标可以反映客户的消费能力和对品牌的认可度，进而判断客户的价值和潜力。
计算R、F、M三个指标值：

data['Recency'] = (datetime.now().date() - data['PurchaseDate'].dt.date).dt.days

frequency_data = data.groupby('CustomerID')['OrderID'].count().reset_index()
# 重命名
frequency_data.rename(columns={'OrderID': 'Frequency'}, inplace=True)

monetary_data = data.groupby('CustomerID')['TransactionAmount'].sum().reset_index()
monetary_data.rename(columns={'TransactionAmount': 'MonetaryValue'}, inplace=True)

可视化主要是讲解了matplotlib的3D图和统计相关图形的绘制和plotly_express的入门：

(1) matplotlib的3D图形绘制

plt.style.use('fivethirtyeight')
fig = plt.figure(figsize=(8,6))

ax = fig.gca(projection='3d')

z = np.linspace(0, 20, 1000)
x = np.sin(z)
y = np.cos(z)

surf=ax.plot3D(x,y,z)

z = 15 * np.random.random(200)
x = np.sin(z) + 0.1 * np.random.randn(200)
y = np.cos(z) + 0.1 * np.random.randn(200)
ax.scatter3D(x, y, z, c=z, cmap='Greens')

plt.show()
![1501701705975_ pic](https://github.com/lulumengyi/small-tools-python/assets/32284131/a2fe8132-cc9e-4037-9261-1be9902bdb06)

plt.style.use('fivethirtyeight')
fig = plt.figure(figsize=(14,8))

ax = plt.axes(projection='3d')
ax.plot_surface(x, 
                y,
                z, 
                rstride=1,
                cstride=1, 
                cmap='viridis',
                edgecolor='none')

ax.set_title('surface')

# ax.set(xticklabels=[],  # 隐藏刻度
#        yticklabels=[],
#        zticklabels=[])

plt.show()

![1481701705937_ pic](https://github.com/lulumengyi/small-tools-python/assets/32284131/afc06007-8b81-4ac5-8e8c-67cc0dc2944b)

(2) 统计图形绘制

绘制箱型图：
![Uploading 1501701705975_.pic.jpg…]()

np.random.seed(10)
D = np.random.normal((3, 5, 4), (1.25, 1.00, 1.25), (100, 3))

fig, ax = plt.subplots(2, 2, figsize=(9,6), constrained_layout=True)

ax[0,0].boxplot(D, positions=[1, 2, 3])
ax[0,0].set_title('positions=[1, 2, 3]')

ax[0,1].boxplot(D, positions=[1, 2, 3], notch=True)  # 凹槽显示
ax[0,1].set_title('notch=True')

ax[1,0].boxplot(D, positions=[1, 2, 3], sym='+')  # 设置标记符号
ax[1,0].set_title("sym='+'")

ax[1,1].boxplot(D, positions=[1, 2, 3], 
                patch_artist=True,
                showmeans=False, 
                showfliers=False,
                medianprops={"color": "white", "linewidth": 0.5},
                boxprops={"facecolor": "C0", "edgecolor": "white", "linewidth": 0.5},
                whiskerprops={"color": "C0", "linewidth": 1.5},
                capprops={"color": "C0", "linewidth": 1.5})
ax[1,1].set_title("patch_artist=True")

# 设置每个子图的x-y轴的刻度范围
for i in np.arange(2):
    for j in np.arange(2):
        ax[i,j].set(xlim=(0, 4), xticks=[1,2,3],
                    ylim=(0, 8), yticks=np.arange(0, 9))

plt.show()


绘制栅格图：

np.random.seed(1)
x = [2, 4, 6]
D = np.random.gamma(4, size=(3, 50))

# plt.style.use('fivethirtyeight')

fig, ax = plt.subplots(2, 2, figsize=(9,6), constrained_layout=True)

# 默认栅格图-水平方向
ax[0,0].eventplot(D)
ax[0,0].set_title('default')


# 垂直方向
ax[0,1].eventplot(D, 
                  orientation='vertical', 
                  lineoffsets=[1,2,3])
ax[0,1].set_title("orientation='vertical', lineoffsets=[1,2,3]")

ax[1,0].eventplot(D, 
                  orientation='vertical',
                  lineoffsets=[1,2,3],
                  linelengths=0.5) # 线条长度
ax[1,0].set_title('linelengths=0.5')

ax[1,1].eventplot(D, 
                  orientation='vertical',
                  lineoffsets=[1,2,3],
                  linelengths=0.5,
                 colors='orange')
ax[1,1].set_title("colors='orange'")


plt.show()
复制

(3) plotly_express入门      使用plotly_express如何快速绘制散点图、散点矩阵图、气泡图、箱型图、小提琴图、经验累积分布图、旭日图等
![1491701705963_ pic](https://github.com/lulumengyi/small-tools-python/assets/32284131/1d9ca6bf-4773-4b79-a5ac-e0c0c28320c8)


机器学习
基于机器学习的Titanic生存预测
目标变量分析：
相关性分析：
![1511701705984_ pic](https://github.com/lulumengyi/small-tools-python/assets/32284131/262b672b-6415-467e-9d24-ea5a1606be61)

![1521701705991_ pic](https://github.com/lulumengyi/small-tools-python/assets/32284131/fb493c7e-bedf-4783-a6dc-4fc8f4f48e09)
![1531701705997_ pic](https://github.com/lulumengyi/small-tools-python/assets/32284131/3a50add8-fccc-4c4f-a8d7-06c70d3a0330)
![1541701706003_ pic](https://github.com/lulumengyi/small-tools-python/assets/32284131/81f53dbe-cb3a-4ba5-b3d0-083dcb430e9a)

基于树模型的特征重要性排序代码：

f,ax=plt.subplots(2,2,figsize=(15,12))

# 1、模型
rf=RandomForestClassifier(n_estimators=500,random_state=0)
# 2、训练
rf.fit(X,Y)
# 3、重要性排序
pd.Series(rf.feature_importances_, X.columns).sort_values(ascending=True).plot.barh(width=0.8,ax=ax[0,0])
# 4、添加标题
ax[0,0].set_title('Feature Importance in Random Forests')

ada=AdaBoostClassifier(n_estimators=200,learning_rate=0.05,random_state=0)
ada.fit(X,Y)
pd.Series(ada.feature_importances_, X.columns).sort_values(ascending=True).plot.barh(width=0.8,ax=ax[0,1],color='#9dff11')
ax[0,1].set_title('Feature Importance in AdaBoost')

gbc=GradientBoostingClassifier(n_estimators=500,learning_rate=0.1,random_state=0)
gbc.fit(X,Y)
pd.Series(gbc.feature_importances_, X.columns).sort_values(ascending=True).plot.barh(width=0.8,ax=ax[1,0],cmap='RdYlGn_r')
ax[1,0].set_title('Feature Importance in Gradient Boosting')

xgbc=xg.XGBClassifier(n_estimators=900,learning_rate=0.1)
xgbc.fit(X,Y)
pd.Series(xgbc.feature_importances_, X.columns).sort_values(ascending=True).plot.barh(width=0.8,ax=ax[1,1],color='#FD0F00')
ax[1,1].set_title('Feature Importance in XgBoost')

plt.show()      
复制
不同模型对比：
![1551701706009_ pic](https://github.com/lulumengyi/small-tools-python/assets/32284131/aa69eb1b-f84d-447a-a7f5-8c453a70a8d2)

![1561701706016_ pic](https://github.com/lulumengyi/small-tools-python/assets/32284131/9e2cde92-9464-4e00-b28f-9d4fff426094)
![1571701706022_ pic](https://github.com/lulumengyi/small-tools-python/assets/32284131/5851baa1-0c9c-4b49-8e05-62d5256661e8)

基于KNN算法的iris数据集分类
特征分布情况：

pd.plotting.scatter_matrix(X_train, 
                           c=y_train, 
                           figsize=(15, 15),
                           marker='o', 
                           hist_kwds={'bins': 20}, 
                           s=60,
                           alpha=.8
                          )

plt.show()
复制

混淆矩阵：![1611701706052_ pic](https://github.com/lulumengyi/small-tools-python/assets/32284131/26cd9262-7135-4deb-a4bd-8b3f89d1537d)


from sklearn.metrics import classification_report,f1_score,accuracy_score,confusion_matrix
sns.heatmap(confusion_matrix(y_pred, y_test), annot=True)
plt.show()
复制

对新数据预测：

x_new = np.array([[5, 2.9, 1, 0.2]])

prediction = knn.predict(x_new)
复制
基于随机森林算法的员工流失预测
不同教育背景下的人群对比：
![1621701706061_ pic](https://github.com/lulumengyi/small-tools-python/assets/32284131/5ed92b03-fec5-4172-b099-8ef69a7de599)

fig = go.Figure(data=[go.Pie(
    labels=attrition_by['EducationField'],
    values=attrition_by['Count'],
    hole=0.4,
    marker=dict(colors=['#3CAEA3', '#F6D55C']),
    textposition='inside'
)])


fig.update_layout(title='Attrition by Educational Field', 
                  font=dict(size=12), 
                  legend=dict(
                      orientation="h",
                      yanchor="bottom",
                      y=1.02, 
                      xanchor="right",
                      x=1
))

fig.show()
复制

年龄和月收入关系：
![1661701706089_ pic](https://github.com/lulumengyi/small-tools-python/assets/32284131/7c1e0a2c-0303-47ed-801a-8882212d3d90)

![1631701706068_ pic](https://github.com/lulumengyi/small-tools-python/assets/32284131/2c919032-a75e-488f-b25e-51e70b7dd985)

类型编码：

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()

df['Attrition'] = le.fit_transform(df['Attrition'])
df['BusinessTravel'] = le.fit_transform(df['BusinessTravel'])
df['Department'] = le.fit_transform(df['Department'])
df['EducationField'] = le.fit_transform(df['EducationField'])
df['Gender'] = le.fit_transform(df['Gender'])
df['JobRole'] = le.fit_transform(df['JobRole'])
df['MaritalStatus'] = le.fit_transform(df['MaritalStatus'])
df['Over18'] = le.fit_transform(df['Over18'])
df['OverTime'] = le.fit_transform(df['OverTime'])
复制
相关性分析：

![1671701706095_ pic](https://github.com/lulumengyi/small-tools-python/assets/32284131/3d25e29c-931d-4128-b9ae-d3079bb917ef)

基于LSTM的股价预测
LSTM网络模型搭建：

from keras.models import Sequential
from keras.layers import Dense, LSTM

model = Sequential()
# 输入层
model.add(LSTM(128, return_sequences=True, input_shape= (xtrain.shape[1], 1)))
# 隐藏层
model.add(LSTM(64, return_sequences=False))
model.add(Dense(25))
# 输出层
model.add(Dense(1))
# 模型概览
model.summary()
复制
交叉验证实现：

k = 5
number_val = len(xtrain) // k  # 验证数据集的大小
number_epochs = 20
all_mae_scores = []
all_loss_scores = []

for i in range(k):
    # 只取i到i+1部分作为验证集
    vali_X = xtrain[i * number_val: (i+1) * number_val]
    vali_y = ytrain[i * number_val: (i+1) * number_val]

    # 训练集
    part_X_train = np.concatenate([xtrain[:i * number_val],
                                  xtrain[(i+1) * number_val:]],
                                  axis=0
                                 ) 
    part_y_train = np.concatenate([ytrain[:i * number_val],
                                  ytrain[(i+1) * number_val:]],
                                  axis=0
                                 )
    
    print("pxt: \n",part_X_train[:3])
    print("pyt: \n",part_y_train[:3])
    
    # 模型训练
    history = model.fit(part_X_train,
                        part_y_train,
                        epochs=number_epochs,
                        # 传入验证集的数据
                        validation_data=(vali_X, vali_y),
                        batch_size=300,
                        verbose=0  # 0-静默模式 1-日志模式
                       )
    
    mae_history = history.history["mae"]
    loss_history = history.history["loss"]
    all_mae_scores.append(mae_history)
    all_loss_scores.append(loss_history)
复制
时序预测
基于AMIRA的销量预测
自相关性图：
![1681701706110_ pic](https://github.com/lulumengyi/small-tools-python/assets/32284131/97ee0eb0-978c-47e6-b8e4-4cf7df43386b)

![1641701706076_ pic](https://github.com/lulumengyi/small-tools-python/assets/32284131/f47d8493-43ba-452b-8049-14058d849b23)

偏自相关性：
![1691701706116_ pic](https://github.com/lulumengyi/small-tools-python/assets/32284131/6f29d3d7-3409-42b1-8c59-2c45182e1590)


预测未来10天

p,d,q = 5,1,2
model = sm.tsa.statespace.SARIMAX(df['Revenue'],
                                order=(p, d, q),
                                seasonal_order=(p, d, q, 12))
model = model.fit()
model.summary()
ten_predictions = model.predict(len(df), len(df) + 10)  # 预测10天
复制![1701701706123_ pic](https://github.com/lulumengyi/small-tools-python/assets/32284131/2d31bebe-c4b9-4ec0-acdc-3d3213a8a2ec)


基于prophet的天气预测
特征间的关系：
![1711701706128_ pic](https://github.com/lulumengyi/small-tools-python/assets/32284131/836edcc3-e8a3-43d6-b3dd-3d508b427285)



预测效果：
![1721701706134_ pic](https://github.com/lulumengyi/small-tools-python/assets/32284131/76435cc2-4f07-4a2f-8b4f-8ebab52a05e0)
![1731701706139_ pic](https://github.com/lulumengyi/small-tools-python/assets/32284131/0952d8f9-43b7-40a3-843d-2825b44c0d8c)


其他案例
python的6种实现99乘法表
提供2种：

for i in range(1, 10):
    for j in range(1, i+1):  # 例如3*3、4*4的情况，必须保证j能取到i值，所以i+1；range函数本身是不包含尾部数据
        print(f'{j}x{i}={i*j} ', end="")  # end默认是换行；需要改成空格
    print("\n")  # 末尾自动换空行
复制
for i in range(1, 10):       # 外层循环
    j = 1      # 内层循环初始值
    while j <= i:      # 内层循环条件：从1开始循环
        print("{}x{}={}".format(i,j,(i*j)), end=' ')  # 输出格式
        j += 1  # j每循环一次加1，进入下次，直到j<=i的条件不满足，再进入下个i的循环中
    print("\n")
复制
i = 1  # i初始值

while i <= 9:  # 循环终止条件
    j = 1  # j初始值
    while j <= i:    # j的大小由i来控制
        print(f'{i}x{j}={i*j} ', end='')
        j += 1   # j每循环一次都+1，直到j<=i不再满足，跳出这个while循环 
    i += 1  # 跳出上面的while循环后i+1，只要i<9就换行进入下一轮的循环；否则结束整个循环
    print('\n')
复制
python实现简易计算器（GUI界面）
提供部分代码：

import tkinter as tk

root = tk.Tk()  
root.title("Standard Calculator")  
root.resizable(0, 0)  


e = tk.Entry(root,
             width=35,
             bg='#f0ffff',
             fg='black',
             borderwidth=5,
             justify='right',
             font='Calibri 15')

e.grid(row=0, column=0, columnspan=3, padx=12, pady=12)

# 点击按钮
def buttonClick(num): 
    temp = e.get(
    )  
    e.delete(0, tk.END)  
    e.insert(0, temp + num)  

# 清除按钮
def buttonClear():  
    e.delete(0, tk.END)


def buttonGet(oper):  
    global num1, math  
    num1 = e.get()  
    math = oper  
    e.insert(tk.END, math)
    try:
        num1 = float(num1)  
    except ValueError:  
        buttonClear()
