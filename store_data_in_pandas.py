"""
Adapted from https://www.kaggle.com/wenxuanchen/home-depot-product-search-relevance/sklearn-random-forest
This script basically extracts numerical features from text data.
The numerical features are basically the occurrences of the search
term in each column of the tuple.
"""
import pandas as pd
import sys
# from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
import re
from nltk.stem.snowball import SnowballStemmer

# load spell checking dictionary
spell_check_dict = open('spellCheckingDict', 'r')
spell_check_dict_ = {}
for line in spell_check_dict:
    spell_check_dict_ \
        .update(eval('{' + line.strip().strip(',') + '}'))
spell_check_dict.close()
print "Spell check dictionary loaded."

stemmer = SnowballStemmer('english')
stopwords_set = set(stopwords.words())
# stemmer = PorterStemmer()

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
    if is_search_term and s in spell_check_dict_:
        print s, '-->',
        s = spell_check_dict_[s]
        print s
    if type(s) in {int, float}:
        return str(s)
    s = remove_non_ascii(s)

    # s = unicodedata \
    #     .normalize('NFD', unicode(s)).encode('ascii', 'ignore')
    # Split words with a.A

    # This is necessary, otherwise you will get into trouble when
    # you make coalesce and split in extract_feature.py
    # replace  \t\n\r\f\v with ' '
    s = re.sub(r'\s', ' ', s)

    # replace characters like ,$?-()[]{} with spaces
    s = re.sub(r'[,&!\$\?\-\(\)\[\]{}:<>]', " ", s)

    # remove . between abbreviations.
    s = re.sub(r'([A-Z])\.(?=[A-Z. ])', r"\1", s)

    # remove dot in something like cat..dog
    s = re.sub(r'\.\.+', ' ', s)

    # Remove all dots except dot in numbers.
    s = s.replace(".", " . ")
    # remove spaces between numbers, dot and numbers
    s = re.sub(r"([0-9]) *\. *([0-9])", r"\1.\2", s)
    s = s.replace(' . ', ' ')

    # replace numbers including , as pure number
    s = re.sub(r'([0-9]),([0-9])', r'\1\2', s)

    # add space after ending .
    # compared to \w\.[A-Z] this doesn't change something like
    # U.S.A
    # Drawback: remove ending . of in. ft. and so on.
    # s = re.sub(r"([a-z0-9])\.([A-Z])", r"\1 \2", s)

    # fix something like: cleaningSeat securityStandard and so on.
    s = re.sub(r'([a-z0-9])([A-Z])', r'\1 \2', s)
    s = re.sub(r'([0-9])([a-z])', r'\1 \2', s)
    s = s.lower()
    s = re.sub(r'([a-z])([0-9])', r'\1 \2', s)

    # remove urls
    s = re.sub(r'https? ?[^ ]+', '', s)

    # This modifies a special case where // is used instead of /
    # like 1//2 should be 1/2
    s = s.replace("//", "/")

    # this is not associated with number , so it could be ignored
    # There is no cases like 1 / 2 in product description,
    # and very likely there will be no cases like this in other
    # files.
    s = s.replace(' / ', ' ')

    # remove adjacent spaces
    s = re.sub(r' +', ' ', s)

    # replace one two and so on with 1 2 and so on
    s = " ".join([str(strNum[z])
                  if z in strNum else z for z in s.split(" ")])

    s = re.sub(r'([a-z]) *[/\\\\] *([a-z])', r'\1 \2', s)
    s = re.sub(r'([0-9])\\\\([0-9])', r'\1/\2', s)

    s = re.sub(r"/$", r"", s)

    # separate adjacent number and characters
    s = re.sub(r"([0-9])([a-z])", r"\1 \2", s)
    s = re.sub(r"([a-z])([0-9])", r"\1 \2", s)

    # replace all forms of multiplication by xbi
    s = s.replace(" x ", " xbi ")
    s = re.sub(r'([0-9])[x\*]([0-9])', r'\1 xbi \2', s)
    s = s.replace("*", " xbi ")
    s = s.replace(" by ", " xbi ")

    # remove adjacent spaces
    s = re.sub(r' +', r' ', s)

    # Transform units
    s = re.sub(r"([0-9]+) ?(inches|inch|in|')\.?", r"\1 in. ", s)
    s = re.sub(r"([0-9]+) ?(foot|feet|ft|'')\.?", r"\1 ft. ", s)
    s = re.sub(r"([0-9]+) ?(pounds|pound|lbs|lb)\.?", r"\1 lb. ",
               s)
    s = re.sub(r"([0-9]+) ?(square|sq)\.? ?(feet|foot|ft)\.?",
               r"\1 sq.ft. ", s)
    s = re.sub(r"([0-9]+) ?(cubic|cu)\.? ?(feet|foot|ft)\.?",
               r"\1 cu.ft. ", s)
    s = re.sub(r"([0-9]+) ?(gallons|gallon|gal)\.?", r"\1 gal. ",
               s)
    s = re.sub(r"([0-9]+) ?(ounces|ounce|oz)\.?", r"\1 oz. ", s)
    s = re.sub(r"([0-9]+) ?(centimeters|cm)\.?", r"\1 cm. ", s)
    s = re.sub(r"([0-9]+) ?(milimeters|mm)\.?", r"\1 mm. ", s)

    s = re.sub(r"([0-9]+) ?(degrees|degree)\.?", r"\1 deg. ", s)
    s = re.sub(r' v(\. | |$)', ' volts ', s)
    s = re.sub(r"([0-9]+) ?(volts|volt) *\.?", r"\1 volt. ", s)
    s = re.sub(r"([0-9]+) ?(watts|watt)\.?", r"\1 watt. ", s)
    s = re.sub(r"([0-9]+) ?(amperes|ampere|amps|amp)\.?",
               r"\1 amp. ", s)

    s = re.sub(r"(air *conditioner|a *c)", "ac", s)
    s = s.replace("toliet", "toilet")
    s = s.replace("vinal", "vinyl")
    s = s.replace("vynal", "vinyl")
    s = s.replace("skill", "skil")
    s = s.replace("snowbl", "snow bl")
    s = s.replace("plexigla", "plexi gla")
    s = s.replace("rustoleum", "rust oleum")
    s = s.replace("whirpool", "whirlpool")
    s = s.replace("whirlpoolga", "whirlpool ga")
    s = s.replace("whirlpoolstainless", "whirlpool stainless")
    s = re.sub(r' +', r' ', s)
    s = " ".join([stemmer.stem(z) for z in s.split(" ") if
                  z not in stopwords_set])
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

    # extract color for each product
    df_color = df_product_attribute \
        [df_product_attribute.name == 'Color Family' |
         df_product_attribute.name == 'Color/Finish'] \
        [['product_uid', 'value']].rename(
            columns={'value': 'color'})

    # extract material for each product
    df_material = df_product_attribute \
        [df_product_attribute.name == "Material"] \
        [['product_uid', 'value']].rename(
            columns={'value': 'material'})

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

    # aggregate color values
    df_color_agg = df_color \
        .groupby('product_uid') \
        .agg(lambda ls: " ".join([str(text) for text in ls]))
    df_color \
        = pd.DataFrame({
        'product_uid': df_color_agg['color']
            .keys(),
        'attributes': df_color_agg['color']
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
    df_all = pd.merge(df_all, df_color, how='left',
                      on='product_uid')
    df_all = pd.merge(df_all, df_material, how='left',
                      on='product_uid')

    # store merged text
    df_all.to_pickle('df_all_before_stem')

    # stem all text fields.
    df_all['brand'] = df_all['brand'] \
        .map(lambda s: stem_text(s, False))
    print 'Brand info processed.'
    df_all['search_term'] = df_all['search_term'] \
        .map(lambda s: stem_text(s, True))
    print 'Search_term processed.'
    df_all['product_title'] = df_all['product_title'] \
        .map(lambda s: stem_text(s, False))
    print 'title info processed'
    df_all['product_description'] = \
        df_all['product_description'] \
        .map(lambda s: stem_text(s, False))
    print 'description processed'
    df_all['attributes'] = df_all['attributes'] \
        .map(lambda s: stem_text(s, False))
    print 'attributes processed'
    df_all['color'] = df_all['color'] \
        .map(lambda s: stem_text(s, False))
    print 'color processed'
    df_all['material'] = df_all['material'] \
        .map(lambda s: stem_text(s, False))
    print 'material processed'

    # save the whole df
    df_all.to_pickle('df_all')


if __name__ == '__main__':
    main()
