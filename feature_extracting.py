import pandas as pd
from nltk.stem.snowball import SnowballStemmer
import re
import numpy as np

# input all data sets
df_train = pd.read_csv("train.csv", encoding="ISO-8859-1")
df_test = pd.read_csv("test.csv", encoding="ISO-8859-1")
df_product_desc = pd.read_csv("product_descriptions.csv")
df_product_attribute = pd.read_csv("attributes.csv")
# extract brands for each product
df_brand = df_product_attribute\
    [df_product_attribute.name == "MFG Brand Name"]\
    [['product_uid', 'value']].rename(columns={'value': 'brand'})
# extract product id and attribute values
df_product_attribute_selected \
    = df_product_attribute[['product_uid', 'value']]\
    .rename(columns={'value': 'attributes'})


# Notice that str() cannot be omitted here because text can
# be float and that may cause trouble because join don't connect
# float.
# In addition, using df_attribute_temp.value.agg is safer because
# it gives more detailed error information
df_product_attribute_agg = df_product_attribute_selected \
    .groupby('product_uid')\
    .agg(lambda ls: " ".join([str(text) for text in ls]))
df_product_attribute = pd.DataFrame(
        df_product_attribute_agg.attributes.str.split(' ', 1).tolist(),
        columns=['product_uid', 'attributes'])

train_num = df_train.shape[0]

stemmer = SnowballStemmer('english')


def is_ascii(s):
    return all(ord(c) < 128 for c in s)


def stem_text(s):
    if type(s) in {int, float}:
        s = str(s)
    re.sub(r'[^\x00-\x7F]+', ' ', s)
    if is_ascii(s):
        return " ".join([stemmer.stem(word) for word in
                        s.lower().split()])
    else:
        return " "


# find in str2 occurrences of each word in str1
# example: str1 = "good job", str2 = "good good job"
# it gets 2(for good) and 1(for job), they their sum
# is 3
def find_occurrences(str1, str2):
    return sum(str2.count(word) for word in str1.split())


# merge different tables
df_all = pd.concat((df_train, df_test), axis=0, ignore_index=True)
df_all = pd.merge(df_all, df_product_desc, how='left',
                  on='product_uid')
# print df_all.info()
# print df_product_attribute_agg.info()
df_all = pd.merge(df_all, df_product_attribute, how='left',
                  on='product_uid')
df_all = pd.merge(df_all, df_brand, how='left', on='product_uid')

df_all['brand'] = df_all['brand'].map(lambda s: stem_text(s))

df_all['search_term'] = df_all['search_term'] \
    .map(lambda s: stem_text(s))
# print df_all.info()

# print df_all
# # save 1st part of the processed df
# df_file = open('first', 'w')
#
# df_all.to_csv(df_file)
# df_file.close()

df_all['product_title'] = df_all['product_title'] \
    .map(lambda s: stem_text(s))

# # save 2nd part of the processed df
# df_file = open('second', 'w')
# df_all.to_csv(df_file)
# df_file.close()

df_all['product_description'] = \
    df_all['product_description'].map(lambda s: stem_text(s))

# # save 3rd part of the processed df
# df_file = open('third', 'w')
# df_all.to_csv(df_file)
# df_file.close()

df_all['attributes'] = df_all['attributes'] \
    .map(lambda s: stem_text(s))

# # save 4th part of the processed df
# df_file = open('fourth', 'w')
# df_all.to_csv(df_file)
# df_file.close()



# save the whole df
# df_file = open('df_file_fifth', 'w')
df_all.to_pickle('df_all')
# df_file.close()

