# MESMA COISA QUE NO ipynb (Notebook)

import math
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import pandas as pd
import plotly
import plotly.express as px
import requests
import urllib.parse
import uuid
import xml.etree.ElementTree as ET
from collections import OrderedDict
from sklearn.cluster import MiniBatchKMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from tqdm import notebook as tqdm

# plotly.offline.init_notebook_mode(connected=False)


# const
BASEURL_INFO = 'https://eutils.ncbi.nlm.nih.gov/entrez/eutils/einfo.fcgi'
BASEURL_SRCH = 'https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi'
BASEURL_FTCH = 'https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi'

# parameters
SOURCE_DB    = 'pubmed'
TERM         = 'cancer'
DATE_TYPE    = 'pdat'       # Type of date used to limit a search. The allowed values vary between Entrez databases, but common values are 'mdat' (modification date), 'pdat' (publication date) and 'edat' (Entrez date). Generally an Entrez database will have only two allowed values for datetype.
MIN_DATE     = '2018/01/01' # yyyy/mm/dd
MAX_DATE     = '2019/12/31' # yyyy/mm/dd
SEP          = '|'
BATCH_NUM    = 1000

# seed
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)


'''
make query function

base_url: base_url
params: parameter dictionary
        ex) {key1: value1, key2: value2}
'''
def mkquery(base_url, params):
    base_url += '?'
    for key, value in zip(params.keys(), params.values()):
        base_url += '{key}={value}&'.format(key=key, value=value)
    url = base_url[0:len(base_url) - 1]
    print('request url is: ' + url)
    return url

'''
getXmlFromURL
(mkquery wrapper)

base_url: base_url
params: parameter dictionary
        ex) {key1: value1, key2: value2}
'''
def getXmlFromURL(base_url, params):
    response = requests.get(mkquery(base_url, params))
    return ET.fromstring(response.text)

'''
getTextFromNode

root: Xml root node
path: XPath
fill: fill na string
mode: 0 = text, 1 = attribute
attrib: attribute name
'''
def getTextFromNode(root, path, fill='', mode=0, attrib='attribute'):
    if (root.find(path) == None):
        return fill
    else:
        if mode == 0:
            return root.find(path).text
        if mode == 1:
            return root.find(path).get(attrib)
    

# example
rootXml = getXmlFromURL(BASEURL_INFO, {'db': SOURCE_DB})


# Info API
rootXml = getXmlFromURL(BASEURL_INFO, {'db': SOURCE_DB})
print(rootXml.find('DbInfo').find('Count').text)
print(rootXml.find('DbInfo').find('LastUpdate').text)


# get xml
rootXml = getXmlFromURL(BASEURL_SRCH, {
    'db': SOURCE_DB,
    'term': TERM,
    'usehistory': 'y',
    'datetype': DATE_TYPE,
    'mindate': MIN_DATE,
    'maxdate': MAX_DATE})


# get querykey and webenv
Count = rootXml.find('Count').text
QueryKey = rootXml.find('QueryKey').text
WebEnv = urllib.parse.quote(rootXml.find('WebEnv').text)

print('total Count: ', Count)
print('QueryKey   : ', QueryKey)
print('WebEnv     : ', WebEnv)


articleDics = []
authorArticleDics = []
authorAffiliationDics = []

def pushData(rootXml):
    for article in rootXml.iter('PubmedArticle'):
        # get article info
        articleDic = {
            'PMID'                    : getTextFromNode(article, 'MedlineCitation/PMID', ''),
            'JournalTitle'            : getTextFromNode(article, 'MedlineCitation/Article/Journal/Title', ''),
            'Title'                   : getTextFromNode(article, 'MedlineCitation/Article/ArticleTitle', ''),
            'doi'                     : getTextFromNode(article, 'MedlineCitation/Article/ELocationID[@EIdType="doi"]', ''),
            'Abstract'                : getTextFromNode(article, 'MedlineCitation/Article/Abstract/AbstractText', ''),
        #    if you want to get data in flat(denormalized), uncomment below. but it's difficult to use for analytics.
        #    'Authors'                 : SEP.join([author.find('ForeName').text + ' ' +  author.find('LastName').text if author.find('CollectiveName') == None else author.find('CollectiveName').text for author in article.findall('MedlineCitation/Article/AuthorList/')]),
        #    'AuthorIdentifiers'       : SEP.join([getTextFromNode(author, 'Identifier', 'None') for author in article.findall('MedlineCitation/Article/AuthorList/')]),
        #    'AuthorIdentifierSources' : SEP.join([getTextFromNode(author, 'Identifier', 'None', 1, 'Source') for author in article.findall('MedlineCitation/Article/AuthorList/')]),
            'Language'                : getTextFromNode(article, 'MedlineCitation/Article/Language', ''),
            'Year_A'                  : getTextFromNode(article, 'MedlineCitation/Article/ArticleDate/Year', ''),
            'Month_A'                 : getTextFromNode(article, 'MedlineCitation/Article/ArticleDate/Month', ''),
            'Day_A'                   : getTextFromNode(article, 'MedlineCitation/Article/ArticleDate/Day', ''),
            'Year_PM'                 : getTextFromNode(article, 'PubmedData/History/PubMedPubDate[@PubStatus="pubmed"]/Year', ''),
            'Month_PM'                : getTextFromNode(article, 'PubmedData/History/PubMedPubDate[@PubStatus="pubmed"]/Month', ''),
            'Day_PM'                  : getTextFromNode(article, 'PubmedData/History/PubMedPubDate[@PubStatus="pubmed"]/Day', ''),
            'Status'                  : getTextFromNode(article, './PubmedData/PublicationStatus', ''),
            'MeSH'                    : SEP.join([getTextFromNode(mesh, 'DescriptorName') for mesh in article.findall('MedlineCitation/MeshHeadingList/')]),
            'MeSH_UI'                 : SEP.join([getTextFromNode(mesh, 'DescriptorName', '', 1, 'UI') for mesh in article.findall('MedlineCitation/MeshHeadingList/')]),
            'Keyword'                 : SEP.join([keyword.text if keyword.text != None else ''  for keyword in article.findall('MedlineCitation/KeywordList/')])
        }
        articleDics.append(OrderedDict(articleDic))

        if article.find('MedlineCitation/MeshHeadingList/MeshHeading/') != None:
            tmp = article

        # get author info
        for author in article.findall('MedlineCitation/Article/AuthorList/'):

            # publish author ID
            # * It's only random id. not use for identify author. if you want to identify author, you can use identifier.
            authorId = str(uuid.uuid4())

            # author article
            authorArticleDic = {
                'authorId'         : authorId,
                'PMID'             : getTextFromNode(article, 'MedlineCitation/PMID', ''),
                'name'             : getTextFromNode(author, 'ForeName') + ' ' +  getTextFromNode(author,'LastName') if author.find('CollectiveName') == None else author.find('CollectiveName').text,
                'identifier'       : getTextFromNode(author, 'Identifier', '') ,
                'identifierSource' : getTextFromNode(author, 'Identifier', '', 1, 'Source')
            }
            authorArticleDics.append(OrderedDict(authorArticleDic))

            # author affiliation(author: affiliation = 1 : n)
            if author.find('./AffiliationInfo') != None:
                for affiliation in author.findall('./AffiliationInfo'):
                    authorAffiliationDic = {
                        'authorId'          : authorId,
                        'affiliation'       : getTextFromNode(affiliation, 'Affiliation', '') ,
                    }
                    authorAffiliationDics.append(OrderedDict(authorAffiliationDic))



# ceil
iterCount = math.ceil(int(Count) / BATCH_NUM)

# get all data
for i in tqdm.tqdm(range(iterCount)):
    rootXml = getXmlFromURL(BASEURL_FTCH, {
        'db': SOURCE_DB,
        'query_key': QueryKey,
        'WebEnv': WebEnv,
        'retstart': i * BATCH_NUM,
        'retmax': BATCH_NUM,
        'retmode': 'xml'})
    
    pushData(rootXml)


# article
df_article = pd.DataFrame(articleDics)
df_article.head(10)


# author article
df_author = pd.DataFrame(authorArticleDics)
df_author.head(10)


# author affiliation
df_affiliation = pd.DataFrame(authorAffiliationDics)
df_affiliation.head(10)


df_article.to_csv('pubmed_article.csv')
df_author.to_csv('pubmed_author.csv')
df_affiliation.to_csv('pubmed_affiliation.csv')


# reload
df_article = pd.read_csv('pubmed_article.csv', index_col=0)
df_author = pd.read_csv('pubmed_author.csv', index_col=0)
df_affiliation = pd.read_csv('pubmed_affiliation.csv', index_col=0)


print('number of article: ', len(df_article))


# Categorical Feature
for catCol in ['Language', 'Status']:
    fig = px.bar(df_article[catCol].value_counts().reset_index(), x='index', y=catCol, height=400, width=800, title="{}'s unique count".format(catCol), labels={catCol: 'Count', 'index': catCol})
    fig.show()
    

# concat
# df_article['ArticlePublishDate'] = df_article['Year_A'].fillna(0).astype(int).astype(str).str.zfill(4) + '-' +  df_article['Month_A'].fillna(0).astype(int).astype(str).str.zfill(2) + '-' +  df_article['Day_A'].fillna(0).astype(int).astype(str).str.zfill(2)
df_article['ArticlePublishDate'] = df_article['Year_A'].fillna(0).astype(int).astype(str).str.zfill(4) + '-' +  df_article['Month_A'].fillna(0).astype(int).astype(str).str.zfill(2)
df_article['PubMedPublishDate'] = df_article['Year_PM'].fillna(0).astype(int).astype(str).str.zfill(4) + '-' +  df_article['Month_PM'].fillna(0).astype(int).astype(str).str.zfill(2)


for catCol in ['ArticlePublishDate', 'PubMedPublishDate']:
    fig = px.line(df_article[df_article[catCol] != '0000-00'][catCol].value_counts().reset_index().sort_values('index'), x='index', y=catCol, height=600, width=1200, title="{}'s distribution".format(catCol), labels={catCol: 'Publish Count', 'index': catCol})
    fig.show()


df_article[df_article['Language'] != 'eng'].head(5)


# False: not nan, True: is nan, values are percent
pd.merge(df_article, df_author, on='PMID', how='left').isnull().apply(lambda col: col.value_counts(), axis=0).fillna(0).astype(np.float).apply(lambda col: col/col.sum(), axis=0)


for catCol in ['authorId']:
    fig = px.bar(df_author.groupby('PMID', as_index=False).count()[catCol].value_counts().reset_index(), x='index', y=catCol, height=600, width=1600, title="How many authors in each articles?".format(catCol), labels={catCol: 'Count', 'index': catCol})
    fig.show()  


# False: not nan, True: is nan, values are percent
pd.DataFrame(pd.DataFrame(df_article['MeSH'].fillna('').astype(str) + df_article['Keyword'].fillna('').astype(str))[0] == '').apply(lambda col: col.value_counts(), axis=0).fillna(0).astype(np.float).apply(lambda col: col/col.sum(), axis=0)


df_article['allText'] = df_article['Title'].fillna('') + df_article['Abstract'].fillna('') + df_article['MeSH'].fillna('') + df_article['Keyword'].fillna('')


tfidf = TfidfVectorizer(
    min_df = 5,
    max_df = 0.95,
    max_features = 8000,
    stop_words = 'english'
)

tfidf.fit(df_article.allText)
text = tfidf.fit_transform(df_article.allText)


"""
Finding Optimal Clusters¶
https://www.kaggle.com/jbencina/clustering-documents-with-tfidf-and-kmeans
"""
def find_optimal_clusters(data, max_k):
    iters = range(10, max_k+1, 10)
    
    sse = []
    for k in iters:
        sse.append(MiniBatchKMeans(n_clusters=k, init_size=1024, batch_size=2048, random_state=20).fit(data).inertia_)
        print('Fit {} clusters'.format(k))
        
    f, ax = plt.subplots(1, 1)
    ax.plot(iters, sse, marker='o')
    ax.set_xlabel('Cluster Centers')
    ax.set_xticks(iters)
    ax.set_xticklabels(iters)
    ax.set_ylabel('SSE')
    ax.set_title('SSE by Cluster Center Plot')
    
find_optimal_clusters(text, 100)


# Clustring
clusters = MiniBatchKMeans(n_clusters=80, init_size=1024, batch_size=2048, random_state=20).fit_predict(text)


print(text.shape)
print(len(clusters))


# random sampling size
RANDOM_SAMPLING_SIZE = 3000

# random sampling
random_idx = np.random.choice(range(text.shape[0]), size=RANDOM_SAMPLING_SIZE, replace=False)

# t-sne(with random sampling pca)
tsne = TSNE(random_state=RANDOM_STATE).fit_transform(PCA(n_components=50, random_state=RANDOM_STATE).fit_transform(text[random_idx,:].todense()))

# random sampling df
df_article_tsne = df_article.iloc[random_idx]

# horizontal concat
df_article_tsne = df_article.iloc[random_idx].copy()
df_article_tsne['tsne_x'] = tsne[:, 0]
df_article_tsne['tsne_y'] = tsne[:, 1]
df_article_tsne['cluster'] = clusters[random_idx]
df_article_tsne.head()


# scatter visualize
fig = px.scatter(
    df_article_tsne, 
    x="tsne_x", 
    y="tsne_y", 
    color="cluster",
    height=1200,
    # size='petal_length', 
    color_continuous_scale=px.colors.sequential.Plasma,
    hover_data=['Title', 'PMID']
)
fig.update_layout(
    showlegend=False
) 
fig.show()


def get_top_keywords(data, clusters, labels, n_terms):
    df = pd.DataFrame(data.todense()).groupby(clusters).mean()
    clusterTexts = []
    
    for i,r in df.iterrows():
        top_keywords = ','.join([labels[t] for t in np.argsort(r)[-n_terms:]])
        clusterTexts.append(top_keywords)
#         print('\nCluster {}'.format(i))
#         print(top_keywords)
    return clusterTexts

clusterTexts = get_top_keywords(text, clusters, tfidf.get_feature_names(), 10)


df_article['cluster'] = clusters
df_cluster = pd.DataFrame(df_article.groupby('cluster', as_index=False).count().sort_values('cluster')['PMID'].values, columns=['num_articles'])
df_cluster['keywords'] = clusterTexts
df_cluster.sort_values('num_articles')