import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules



################################################################################################
# Görev1: Veri Ön İşleme İşlemlerini Gerçekleştiriniz
# Germany seçimi sonraki basamakta olacaktır.
################################################################################################


#########################
# Veri Ön İşleme
#########################



# Test Excel sheet isimleri
e = pd.ExcelFile("datasets/online_retail_II.xlsx")
e.sheet_names

# Verinin Excel'den Okunması
df2 = pd.read_excel("datasets/online_retail_II.xlsx", sheet_name='Year 2010-2011')
df = df2.copy()

df.head(5)
df.info()

pd.set_option('display.max_columns', None)
# pd.set_option('display.max_rows', None)
pd.set_option('display.width', 500)
# çıktının tek bir satırda olmasını sağlar.
pd.set_option('display.expand_frame_repr', False)


def outlier_thresholds(dataframe, variable):
    quartile1 = dataframe[variable].quantile(0.01)
    quartile3 = dataframe[variable].quantile(0.99)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit

def replace_with_thresholds(dataframe, variable):
    low_limit, up_limit = outlier_thresholds(dataframe, variable)
    dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit

def retail_data_prep(dataframe):
    dataframe.dropna(inplace=True)
    dataframe = dataframe[~dataframe["Invoice"].str.contains("C", na=False)]
    dataframe = dataframe[dataframe["Quantity"] > 0]
    dataframe = dataframe[dataframe["Price"] > 0]
    replace_with_thresholds(dataframe, "Quantity")
    replace_with_thresholds(dataframe, "Price")
    return dataframe

df = retail_data_prep(df)

df.shape

################################################################################################
# Görev 2: Germany müşterileri üzerinden birliktelik kuralları üretiniz.
################################################################################################

df['Country'].unique()

df_de = df[df["Country"] == "Germany"]

df_de.shape
############################################
# ARL Veri Yapısını Hazırlama (Invoice-Product Matrix)
############################################


# Verinin gelmesini istediğimiz durum: her bir ürün için satın alma oldu mu olmadı mı Binary coding.

# Description   NINE DRAWER OFFICE TIDY   SET 2 TEA TOWELS I LOVE LONDON    SPACEBOY BABY GIFT SET
# Invoice
# 536370                              0                                 1                       0
# 536852                              1                                 0                       1
# 536974                              0                                 0                       0
# 537065                              1                                 0                       0
# 537463                              0                                 0                       1


df_de.groupby(['Invoice', 'Description']).agg({"Quantity": "sum"}).head(20)


# Pivot the table with unstack()
df_de.groupby(['Invoice', 'Description']).agg({"Quantity": "sum"}).unstack().iloc[0:5, 0:5]

# Replace NaN with 0.0
df_de.groupby(['Invoice', 'Description']).agg({"Quantity": "sum"}).unstack().fillna(0).iloc[0:5, 0:5]

# Binary encoding. Replace all values more that 1 with 1 and the rest with 0
df_de.groupby(['Invoice', 'Description']).agg({"Quantity": "sum"}).unstack().fillna(0).applymap(
    lambda x: 1 if x > 0 else 0).iloc[0:5, 0:5]


# 2. method: Binary coding process with a single code block

df_de.groupby(['Invoice', 'Description']). \
    agg({"Quantity": "sum"}). \
    unstack(). \
    fillna(0). \
    applymap(lambda x: 1 if x > 0 else 0).iloc[0:5, 0:5]


# Same ARL data structure created with a function
# Soru: id neden parametre şartlı yapıda
def create_invoice_product_df(dataframe, id=False):
    if id:
        return dataframe.groupby(['Invoice', "StockCode"])['Quantity'].sum().unstack().fillna(0). \
            applymap(lambda x: 1 if x > 0 else 0)
    else:
        return dataframe.groupby(['Invoice', 'Description'])['Quantity'].sum().unstack().fillna(0). \
            applymap(lambda x: 1 if x > 0 else 0)

de_inv_pro_df = create_invoice_product_df(df_de)
de_inv_pro_df.head()

de_inv_pro_df = create_invoice_product_df(df_de, id=True)


############################################
# Birliktelik Kurallarının Çıkarılması
############################################


frequent_itemsets = apriori(de_inv_pro_df, min_support=0.01, use_colnames=True)
frequent_itemsets.sort_values("support", ascending=False).head(50)

rules = association_rules(frequent_itemsets, metric="support", min_threshold=0.01)
rules.sort_values("support", ascending=False).head()

rules.sort_values("lift", ascending=False).head(500)

# Soru: Çıktıdaki POST vs nedir?


################################################################################################
# Görev 3: ID'leri verilen ürünlerin isimleri nelerdir?
################################################################################################

# Get description from stockcode
def check_id(dataframe, stock_code):
    product_name = dataframe[dataframe["StockCode"] == stock_code][["Description"]].values[0].tolist()
    print(product_name)

# Error: IndexError: index 0 is out of bounds for axis 0 with size 0
check_id(df_de, 21987)
check_id(df_de, 23235)
check_id(df_de, 22747)

################################################################################################
# Görev 4: Sepetteki kullanıcılar için ürün önerisi yapınız.
################################################################################################


product_id = 22326
check_id(df, product_id)
sorted_rules = rules.sort_values("lift", ascending=False)

recommendation_list = []

for i, product in enumerate(sorted_rules["antecedents"]):
    for j in list(product):
        if j == product_id:
            recommendation_list.append(list(sorted_rules.iloc[i]["consequents"])[0])


recommendation_list[0:3]


################################################################################################
# Görev 5: Önerilen ürünlerin isimleri nelerdir?
################################################################################################

for product in recommendation_list[0:3]:
    check_id(df, product)