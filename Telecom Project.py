#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


# # Parte 1.

# In[3]:


data=pd.read_csv(r"C:\Users\damic\OneDrive\Documentos\2023\2S 2023\Minería de Datos\Telecom-customer-churn.csv", sep=",")


# In[4]:


data.set_index(["customerID"], inplace = True)


# In[25]:


data.head()


# In[5]:


data['SeniorCitizen'].value_counts() #debido a que esta variable toma valores 1/0, la voy a hacer objeto
data['SeniorCitizen']=data['SeniorCitizen'].astype('object')


# In[6]:


data.describe()


# - La media de clientes tiene una antiguedad de 32 meses.
# - Menos del 25% de los clientes tiene antiguedad mayor a 55 meses.
# - En promedio, los cargos mensuales asociados a los clientes son de $64 y los cargos totales de $2283

# In[18]:


data['Churn'].value_counts()


# In[17]:


## Promedios según Abandono
data.groupby('Churn').mean()


# - La antiguedad media de los clientes que abandonan la empresa es de 17 meses, con cargos mensuales promedio de $74 y cargos totales promedio por debajo de los clientes que no abandonaron el servicio

# In[9]:


## Tabla de Contingencia
pd.crosstab(data["Contract"], data["Churn"], margins=True,  margins_name='Total')


# In[10]:


## Tabla de Contingencia con frecuencias
pd.crosstab(data["Contract"], data["Churn"], margins=True,  margins_name='Total', normalize='all') #norm: all,index,columns


# - El 73.4% de los clientes no ha abandonado (26.6% de abandonos)
# - El 55% de los clientes tiene un contrato Month-to-Month
# - El 20.9% de los clientes tiene un contrato por un año
# - El 24.1% de los clientes tiene un contrato por dos años

# In[13]:


#Histograma Tenure en función de Abandono
data["tenure"].hist(by=data["Churn"], bins=15);


# Los clientes suelen quedarse una vez pasados los 60 meses mientras que la mayor cantidad de abandonos se da en los primeros 5 meses

# In[32]:


#Gráfico de barras comparando Género - Abandono
## Gráfico de barras apiladas
pd.crosstab(data["Churn"], data["gender"]).plot(kind="bar",stacked=True)
plt.xlabel("Churn")
plt.ylabel("Total")
plt.grid(axis='y')
#Parace no haber relación entre las variables Género y Churn


# In[40]:


cat_cols


# In[57]:


#Gráfico de barras para las demás variables cualitativas para encontrar posibles relaciones con la variable Abandono
cat_cols = data.select_dtypes(include=['object', 'category']).columns
cat_cols = cat_cols.drop('Churn')
for var in cat_cols:
    pd.crosstab(data[var], data["Churn"],normalize='index').plot(kind="bar",rot=0,stacked = True)
    plt.legend(title='Churn',bbox_to_anchor=(1, 0.5),loc=10)

#Parece no haber relación únicamente en los siguientes pares
#Gender-Churn / PhoneService-Churn


# In[62]:


#boxplots de MonthlyCharges con respecto a Churn
data.boxplot(column="MonthlyCharges", by="Churn",figsize=(5,3));
plt.title(" ");


#boxplots de TotalCharges con respecto a Churn
data.boxplot(column="TotalCharges", by="Churn",figsize=(5,3));
plt.title(" ");

#No parece haber relación significativa entre variables


# In[14]:


data.plot(kind="scatter", x="tenure", y="TotalCharges") ;

#Pinta los puntos según la categoría de otra variable
sns.scatterplot(data=data,x="tenure", y="TotalCharges",hue='Churn');


# Como vemos la mayoría de los abandonos se dan en los primeros meses, siendo los cargos mas bajos.
# A medida que van pasando los meses abandonan los que tienen mayores cargos pero la cantidad luego de 60 meses es muy baja

# In[20]:


data.corr(method='spearman').round(3)
#method='pearson', method='spearman', method='‘kendall'}


# Debido a una fuerte correlación positiva entre la antiguedad y los cargos totales, podemos decir que los clientes de mayor antiguedad suelen tener mayores cargos totales asociados. Esto se debe a que se está sumando el historial de cargos del cliente.
# Si analizamos los cargos incurridos en el mes, vemos que no hay una fuerte relación entre estos y la antiguedad del cliente.

# In[71]:


from ydata_profiling import ProfileReport
get_ipython().system('jupyter nbextension enable --py widgetsnbextension')


# In[72]:


profile=ProfileReport(data)
profile.to_file("report.html") #guarda el html


# In[73]:


#para ver en notebook
profile.to_notebook_iframe()


# # Parte 2.

# In[32]:


data2=data
data2=data2.drop('Dependents', axis=1)


# In[39]:


data2.isnull().any()


# In[44]:


pd.isnull(data2["TotalCharges"]).values.ravel().sum()


# In[43]:


pd.notnull(data2["TotalCharges"]).values.ravel().sum()


# Dado a que solo hay 11 valores nulos en Total Charges y es la única columna con Missings. Representando estos el % 0.15 de los datos, se procede a borrarlos.

# In[46]:


data3=data2.dropna(how='any')


# In[ ]:


data2=data2.drop('Dependents', axis=1)


# In[47]:


#Nos quedamos solo con las filas que tienen una antiguedad menor a 71
seleccion1 = data2.loc[data['tenure'] < 71]


# In[55]:


#Agrupamiento de Monthly Charges en 5 grupos de igual amplitud
pd.cut(data["MonthlyCharges"], bins=5).value_counts().plot(kind='bar')


# In[58]:


#Variable combinada entre Gender y Tenure
conditions= [
    (data["tenure"]<18) & (data["gender"]=="Male"),
    (data["tenure"]>=18) & (data["gender"]=="Male"),
    (data["tenure"]<18) & (data["gender"]=="Female"),
    (data["tenure"]>=18) & (data["gender"]=="Female"), 
]
values = ['0', '1', '2', '3']

data2["tenure_gender"] = np.select(conditions, values)
data2[['tenure', 'gender', 'tenure_gender']]


# In[62]:


data2["tenure_gender"].value_counts().plot(kind='bar')


# La antiguedad no está relacionada con el género.

# In[63]:


data2["PaymentMethod"].value_counts()
#No parece necesario agrupar las variables debido a que son solamente 4.


# In[64]:


# Estandarizando variables "Monthly Charges" y "Total Charges"
from sklearn.preprocessing import StandardScaler


# In[67]:


num_cols = ['MonthlyCharges', 'TotalCharges']
data_num = data[num_cols]
X_std = StandardScaler().fit_transform(data_num)
pd.DataFrame(X_std, columns = num_cols)


# In[68]:


#Dummies Internet Service
dummy = pd.get_dummies(data["InternetService"], prefix="internet")
data_d = data.drop(["InternetService"], axis=1)
data_d = pd.concat([data_d, dummy], axis=1) #crea las dummies, borra la columna original y concatena las dummies con el resto
data_d.head()


# In[69]:


#Dummies Payment Method
dummy = pd.get_dummies(data["PaymentMethod"], prefix="method")
data_d = data.drop(["PaymentMethod"], axis=1)
data_d = pd.concat([data_d, dummy], axis=1) #crea las dummies, borra la columna original y concatena las dummies con el resto
data_d.head()

