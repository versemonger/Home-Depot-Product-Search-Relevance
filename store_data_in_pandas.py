"""
Adapted from https://www.kaggle.com/wenxuanchen/home-depot-product-search-relevance/sklearn-random-forest
This script basically extracts numerical features from text data.
The numerical features are basically the occurrences of the search
term in each column of the tuple.
"""
import pandas as pd
from nltk.stem.snowball import SnowballStemmer
import numpy as np
import sys


def remove_non_ascii(s):
    """
    Remove non ascii characters so that text processing functions
    function correctly on the string.
    :param s: a string that may contain non ascii characters.
    :return: a string whose non ascii characters have been removed
    """
    return "".join([char for char in s if 0 <= ord(char) < 128])


def stem_text(s):
    """
    stem the text.
    :param s: s
    :return: stemmed text.
    """
    if type(s) in {int, float}:
        return str(s)
    # the remove method is a destructive method
    # which returns a new string...
    # so you cannot leverage it without assignment.
    s = remove_non_ascii(s)
    return " ".join([stemmer.stem(word) for word in
                     s.lower().split()])


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
    = df_product_attribute[['product_uid', 'value']] \
    .rename(columns={'value': 'attributes'})

# Notice that str() cannot be omitted here because text can
# be float and that may cause trouble because join don't connect
# float.
# In addition, using df_attribute_temp.value.agg is safer because
# it gives more detailed error information
df_product_attribute_agg = df_product_attribute_selected \
    .groupby('product_uid') \
    .agg(lambda ls: " ".join([str(text) for text in ls]))
df_product_attribute \
    = pd.DataFrame({
        'product_uid': df_product_attribute_agg['attributes']
        .keys(),
        'attributes': df_product_attribute_agg['attributes']
        .get_values()})

train_num = df_train.shape[0]

stemmer = SnowballStemmer('english')

# merge different tables.
df_all = pd.concat((df_train, df_test), axis=0, ignore_index=True)
df_all = pd.merge(df_all, df_product_desc, how='left',
                  on='product_uid')
df_all = pd.merge(df_all, df_product_attribute, how='left',
                  on='product_uid')
df_all = pd.merge(df_all, df_brand, how='left', on='product_uid')

# store merged text
df_all.to_pickle('df_all_before_stem')

# stem all text fields.
df_all['brand'] = df_all['brand'].map(lambda s: stem_text(s))

df_all['search_term'] = df_all['search_term'] \
    .map(lambda s: stem_text(s))

df_all['product_title'] = df_all['product_title'] \
    .map(lambda s: stem_text(s))

df_all['product_description'] = \
    df_all['product_description'].map(lambda s: stem_text(s))

df_all['attributes'] = df_all['attributes'] \
    .map(lambda s: stem_text(s))

# save the whole df
df_all.to_pickle('df_all')
