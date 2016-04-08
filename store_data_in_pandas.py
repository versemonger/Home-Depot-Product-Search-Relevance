"""
Adapted from https://www.kaggle.com/wenxuanchen/home-depot-product-search-relevance/sklearn-random-forest
This script basically extracts numerical features from text data.
The numerical features are basically the occurrences of the search
term in each column of the tuple.
"""
import pandas as pd
from nltk.stem.porter import PorterStemmer
import requests
import re
import time
from random import randint
import enchant


dic = enchant.Dict("en_US")
START_SPELL_CHECK = \
    "<span class=\"spell\">Showing results for</span>"
END_SPELL_CHECK = \
    "<br><span class=\"spell_orig\">Search instead for"

HTML_Codes = (
    ("'", '&#39;'),
    ('"', '&quot;'),
    ('>', '&gt;'),
    ('<', '&lt;'),
    ('&', '&amp;'),
)


def spell_check(s):
    q = '+'.join(s.split())
    # relax and don't let google be angry
    time.sleep(randint(0, 2) / 50.)
    r = requests.get("https://www.google.com/search?q=" + q)
    content = r.text
    start = content.find(START_SPELL_CHECK)
    if start > -1:
        start += len(START_SPELL_CHECK)
        end = content.find(END_SPELL_CHECK)
        search = content[start:end]
        search = re.sub(r'<[^>]+>', '', search)
        for code in HTML_Codes:
            search = search.replace(code[1], code[0])
        search = search[1:]
        print s, '--->', search
    else:
        search = s
    return search


# from nltk.stem.snowball import SnowballStemmer

# stemmer = SnowballStemmer('english')
stemmer = PorterStemmer()
strNum = {'zero': 0, 'one': 1, 'two': 2, 'three': 3, 'four': 4,
          'five': 5, 'six': 6, 'seven': 7, 'eight': 8, 'nine': 9}


def remove_non_ascii(s):
    """
    Remove non ascii characters so that text processing functions
    function correctly on the string.
    :param s: a string that may contain non ascii characters.
    :return: a string whose non ascii characters have been removed
    """
    return "".join([char for char in s if 0 <= ord(char) < 128])


def stem_text(s, is_search_term):
    """
    stem the text.
    :param is_search_term: if true, spell check s.
    :param s: a string of text
    :return: stemmed text string
    """
    if type(s) in {int, float}:
        return str(s)
    s = remove_non_ascii(s)

    # s = unicodedata \
    #     .normalize('NFD', unicode(s)).encode('ascii', 'ignore')
    # Split words with a.A

    # This is necessary, otherwise you will get into trouble when
    # you make coalesce and split in extract_feature.py
    s = s.replace('\t', ' ')
    s = re.sub(r"(\w)\.([A-Z])", r"\1 \2", s)
    s = s.lower()
    s = s.replace(",", " ")
    s = s.replace("  ", " ")
    s = s.replace("$", " ")
    s = s.replace("?", " ")
    s = s.replace("-", " ")
    s = s.replace("//", "/")
    s = s.replace("..", " ")
    s = s.replace(" / ", " ")
    s = s.replace(" \\ ", " ")
    s = s.replace(".", " ")
    s = s.replace("&", " ")
    s = re.sub(r"(^\.|/)", r"", s)
    s = re.sub(r"(\.|/)$", r"", s)
    s = re.sub(r"([0-9])([a-z])", r"\1 \2", s)
    s = re.sub(r"([a-z])([0-9])", r"\1 \2", s)
    s = s.replace(" x ", " xbi ")
    s = re.sub(r"([a-z])( *)\.( *)([a-z])", r"\1 \4", s)
    s = re.sub(r"([a-z])( *)/( *)([a-z])", r"\1 \4", s)
    s = s.replace("*", " xbi ")
    s = s.replace(" by ", " xbi ")
    s = re.sub(r"([0-9])( *)\.( *)([0-9])", r"\1.\4", s)
    s = re.sub(r"([0-9]+)( *)(inches|inch|in|')\.?", r"\1in. ", s)
    s = re.sub(r"([0-9]+)( *)(foot|feet|ft|'')\.?", r"\1ft. ", s)
    s = re.sub(r"([0-9]+)( *)(pounds|pound|lbs|lb)\.?", r"\1lb. ",
               s)
    s = re.sub(r"([0-9]+)( *)(square|sq) ?\.?(feet|foot|ft)\.?",
               r"\1sq.ft. ", s)
    s = re.sub(r"([0-9]+)( *)(cubic|cu) ?\.?(feet|foot|ft)\.?",
               r"\1cu.ft. ", s)
    s = re.sub(r"([0-9]+)( *)(gallons|gallon|gal)\.?", r"\1gal. ",
               s)
    s = re.sub(r"([0-9]+)( *)(ounces|ounce|oz)\.?", r"\1oz. ", s)
    s = re.sub(r"([0-9]+)( *)(centimeters|cm)\.?", r"\1cm. ", s)
    s = re.sub(r"([0-9]+)( *)(milimeters|mm)\.?", r"\1mm. ", s)

    s = re.sub(r"([0-9]+)( *)(degrees|degree)\.?", r"\1deg. ", s)
    s = s.replace(" v ", " volts ")
    s = re.sub(r"([0-9]+)( *)(volts|volt)\.?", r"\1volt. ", s)
    s = re.sub(r"([0-9]+)( *)(watts|watt)\.?", r"\1watt. ", s)
    s = re.sub(r"([0-9]+)( *)(amperes|ampere|amps|amp)\.?",
               r"\1amp. ", s)
    s = s.replace("  ", " ")
    s = s.replace(" . ", " ")
    s = " ".join([str(strNum[z])
                  if z in strNum else z for z in s.split(" ")])

    s = s.lower()
    s = s.replace("toliet", "toilet")
    s = s.replace("airconditioner", "air conditioner")
    s = s.replace("vinal", "vinyl")
    s = s.replace("vynal", "vinyl")
    s = s.replace("skill", "skil")
    s = s.replace("snowbl", "snow bl")
    s = s.replace("plexigla", "plexi gla")
    s = s.replace("rustoleum", "rust oleum")
    s = s.replace("whirpool", "whirlpool")
    s = s.replace("whirlpoolga", "whirlpool ga")
    s = s.replace("whirlpoolstainless", "whirlpool stainless")
    if is_search_term:
        s = spell_check(s)
    s = " ".join([stemmer.stem(z) for z in s.split(" ")])
    return s


def main():
    # input all data sets
    df_train = pd.read_csv("train.csv", encoding="ISO-8859-1")
    df_test = pd.read_csv("test.csv", encoding="ISO-8859-1")
    df_product_desc = pd.read_csv("product_descriptions.csv")
    df_product_attribute = pd.read_csv("attributes.csv")

    # extract brands for each product
    df_brand = df_product_attribute \
        [df_product_attribute.name == "MFG Brand Name"] \
        [['product_uid', 'value']].rename(
            columns={'value': 'brand'})

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

    # merge different tables.
    df_all = pd.concat((df_train, df_test), axis=0,
                       ignore_index=True)
    df_all = pd.merge(df_all, df_product_desc, how='left',
                      on='product_uid')
    df_all = pd.merge(df_all, df_product_attribute, how='left',
                      on='product_uid')
    df_all = pd.merge(df_all, df_brand, how='left',
                      on='product_uid')

    # store merged text
    df_all.to_pickle('df_all_before_stem')

    # stem all text fields.
    df_all['brand'] = df_all['brand']\
        .map(lambda s: stem_text(s, False))

    df_all['search_term'] = df_all['search_term'] \
        .map(lambda s: stem_text(s, True))

    df_all['product_title'] = df_all['product_title'] \
        .map(lambda s: stem_text(s, False))

    df_all['product_description'] = \
        df_all['product_description']\
            .map(lambda s: stem_text(s, False))

    df_all['attributes'] = df_all['attributes'] \
        .map(lambda s: stem_text(s, False))

    # save the whole df
    df_all.to_pickle('df_all')


if __name__ == '__main__':
    main()
