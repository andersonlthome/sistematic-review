#C:Windows> py -m pip install entrezpy --user

import urllib.request
import urllib.parse
import json
import codecs

def OpenFile(fname):
    open(fname, "w").close()
    return open(fname, "a", encoding='utf-8')


print('Beginning EMBASE response download with urllib...')
con_file = open("config.json")#file with your APIkeys
config = json.load(con_file)
con_file.close()

EMBASEapiKey = config['ELSapikey'] #Obtain from https://dev.elsevier.com
ETMapiKey = config['ETMapikey'] #Obtain from https://dev.elsevier.com
token = config['insttoken'] #Obtain from mailto:integrationsupport@elsevier.com 

SearchQuery = '("acute generalized exanthematous pustulosis":ti,ab) AND ([conference abstract]/lim OR [conference review]/lim)'
PageSize = 100 #controls number of records downloaded in one get request 

baseURL = 'https://api.elsevier.com/content/embase/article/'
params = {'query': SearchQuery, 'count':PageSize}
headers = {'X-ELS-APIKey':EMBASEapiKey,'X-ELS-Insttoken':token}

data = urllib.parse.urlencode(params).encode('ascii')

file_result = OpenFile("EMBASEresults.json")
file_abstracts = OpenFile("EMBASEabstracts.txt")

luiCounter = set()
req = urllib.request.Request(baseURL, data, headers)
with urllib.request.urlopen(req) as response:
    the_page = response.read()
    result = json.loads(the_page.decode('utf-8'))
    HitCount = result['header']['hits']

    print('Downloading %d articles' % HitCount)

    for page in range(1,HitCount,PageSize):
        ls = [x for x in [article['itemInfo']['itemIdList']['lui'] for article in result['results']]] #extracts lui identifiers
        luiCounter.update(ls)
        for article in result['results']:
            file_result.write(json.dumps(article,indent=1, sort_keys=True)+'\n')
            head = article['head']
            itemIdList = dict(article['itemInfo'])['itemIdList']
            doi = str()
            pui= str()
            try:
                doi = itemIdList['doi']
            except KeyError:
                pui = itemIdList['pui']

            abstracts = dict(head['abstracts'])
            abstract = abstracts['abstracts'][0]['paras'][0]
            if len(doi) > 0:
                file_abstracts.write("doi:"+doi+"\n"+abstract+'\n')
            else:
                file_abstracts.write("pui:"+pui+"\n"+abstract+'\n')

file_result.close()
file_abstracts.close()
print('Finished download!')