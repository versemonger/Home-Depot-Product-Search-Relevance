import pandas as pd


def find_occurrences(str1, str2):
    """
    find in str2 occurrences of each word in str1
    example: str1 = "good job", str2 = "good good job"
    it gets 2(for good) and 1(for job), they their sum
    is 3
    :param str1:
    :param str2:
    :return:
    """
    return sum(str2.count(word) for word in str1.split())


df_train = pd.read_csv("train.csv", encoding="ISO-8859-1")
train_num = df_train.shape[0]
df_train = None

df_all = pd.read_pickle('df_all')
df_all['product_info'] \
    = df_all['search_term'] + "\t" + df_all['product_title']\
    + "\t" + df_all['product_description'] + "\t"\
    + df_all['attributes'] + "\t" + df_all['brand']

df_all['word_in_title'] = df_all['product_info']\
    .map(lambda x: find_occurrences(x.split('\t')[0],
                                    x.split('\t')[1]))
df_all['word_in_description'] = df_all['product_info']\
    .map(lambda x: find_occurrences(x.split('\t')[0],
                                    x.split('\t')[2]))
df_all['word_in_description'] = df_all['product_info']\
    .map(lambda x: find_occurrences(x.split('\t')[0],
                                    x.split('\t')[2]))
df_all = df_all.drop(['search_term','product_title','product_description'\
,'product_info'],axis=1)

