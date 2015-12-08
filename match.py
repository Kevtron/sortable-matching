#!/usr/bin/env python

import json
import re
import math

def main():
    productTFs, productCorpus = prepareProducts()
    listingTFs, listingCorpus, listingData = prepareListings()
    fullCorpus = productCorpus + listingCorpus
    fullIDFs = idf(fullCorpus)
    productTFIDF = tfidf(productTFs, fullIDFs)
    listingTFIDF = tfidf(listingTFs, fullIDFs)
    invertedProductIndexDict = invertToDict(productTFIDF)
    invertedListingIndexDict = invertToDict(listingTFIDF)
    commonTokenMapping = findCommonTokens(invertedProductIndexDict, invertedListingIndexDict)
    results = match(productTFIDF, listingTFIDF, commonTokenMapping)
    writeResults(results, listingData)

def tokenize(string):
    """ A simple implementation of input string tokenization
    Args:
        string (str): input string
    Returns:
        list: a list of tokens
    Possible extensions: remove stopwords. 
    """
    split_regex = r'[\W_]+'
    return filter(None, re.split(split_regex, string.lower()))

def prepareProducts():
    corpus =[] 
    productData = []
    productBag = {}
    for line in open("./products.txt",'r'):
        productData.append(json.loads(line))
    for product in productData:
        productAttributes = []
        for k,v in product.items():
            if not k == 'announced-date': # announced date doesn't add much information to the listing comparison
                tokens = tokenize(v)
                productAttributes += tokens
        corpus += list(set(productAttributes))
        corpus.append(product['product_name'].lower())
        productBag[product['product_name'].lower()] = tf(productAttributes) 
    return productBag, corpus

def prepareListings():
    corpus=[]
    listingData = []
    listingBag = {}
    listingIndex=0
    for line in open("./listings.txt",'r'):
        listingData.append([listingIndex, json.loads(line)])
        listingIndex+=1
    for listing in listingData:
        listingAttributes = []
        for k,v in listing[1].items():
            if not k == 'currency' or not k == 'price': # price and currency don't tell us much about the product
                tokens = tokenize(v)
                listingAttributes += tokens
        corpus += list(set(listingAttributes))
        listingBag[listing[0]] = tf(listingAttributes) 
    return listingBag, corpus, listingData

def writeResults(results, listingData):
    '''Write file with results
    Args: 
        Results (dict) : mapping { product : [listingID,cossine similarity score] }
        ListingData (list) : ordered listings [[listingID, listingObject]...]
    Returns
        Results.txt on disk { "product_name": product, "listing": [listingObject(s)]}.
    '''
    f=open('results.txt', 'w')
    for product in results.keys():  
        listing = results[product][0]
        f.write(json.dumps( {"product_name": product, "listing": [listingData[listing][1]] })) 
        f.write("\n")
    return

def match(productTFIDF, listingTFIDF, commonTokenMapping):
    '''Match products to listing usings the cossim similarity. Each product matches at most one listing. 0.25 threshold could be adjusted, but seems to be the tipping point between losing true positives and accumulating a lot of false positives
    Args: 
        commonTokenMapping (dict) : mapping {(product, listingID) : [commonTokens]}
        productTFIDF (dict) : mapping { product : {token : TDIDF weight}}
        listingTDIF (dict) : mapping { listingID : {token : TDIDF weight}}
    Returns: 
        results (dict) : mapping { product : [listingID,cossine similarity score] }
    '''
    results = {}
    for index, commonTokens in commonTokenMapping.items():
        product, listing = index   
        productVector = productTFIDF[product]
        listingVector = listingTFIDF[listing]
        cossimScore = cossim(productVector, listingVector, commonTokens)
        if results.get(product, [0,0])[1] < cossimScore and cossimScore > 0.25:
            results[product]=[listing, cossimScore]   
        else:
            continue
    return results


def findCommonTokens(invertedProductIndexDict, invertedListingIndexDict):
    '''Find common tokens between two mappings
    Args:
        invertedProductIndexDict (dict): input mapping {token: product}
        invertedListingIndexDict (dict): input mapping {token: listingID}
    Returns:
        commonTokenMapping (dict) : mapping {(product, listingID), [commonTokens]}
    '''
    commonTokenMapping = {}
    for token, products in invertedProductIndexDict.items():
        if token in invertedListingIndexDict:
            listings = invertedListingIndexDict[token]
            keyGen = ((product, listing) for product in products for listing in listings)
            for key in keyGen:
                commonTokenMapping[key] =  commonTokenMapping.get(key, []) + [token]
    return commonTokenMapping

def tf(tokens):
    """ Compute TF
    Args:
        tokens (list): input list of tokens
    Returns:
        Dictionary: (token, TF value)
    """
    numTokens = len(tokens)
    tokenMapping = {}
    for token in tokens:
        tokenMapping[token] = tokenMapping.get(token,0) + 1./numTokens
    return tokenMapping

def idf(corpus):
    """ Compute IDF
    Args:
        corpus (list): input corpus
    Returns:
        Dictionary: (token, IDF value)
    """
    N = float(len(corpus))
    uniqueTokens = list(set(corpus))
    tokenMapping = {}
    for token in corpus:
        tokenMapping[token] = tokenMapping.get(token,0) + 1.
    idfs = {k: N * 1./v for k, v in tokenMapping.items()}
    return idfs

def tfidf(tfs, idfs):
    """ Compute TFIDF
    Args:
        tfs (dictionary): token to tf mapping
        idfs (dictionary): token to idf mapping
    Returns:
        Dictionary: (token, tfidf value)
    """
    tfIdfDict = {}
    for k, v in tfs.items():
        for token in v.keys():
            print k, v[token], idfs[token]
        tfIdfDict[k] = { token: v[token]*idfs[token] for token in v.keys()}
    return tfIdfDict

def dotprod(a, b, common):
    """ Compute modified dot product with pre-calculated intersection
    Args:
        a (dictionary): first dictionary of record to value
        b (dictionary): second dictionary of record to value
        common (list): list of common values 
    Returns:
        dotProd: result of the dot product with the two input dictionaries
    """
    return sum(a[k]*b[k] for k in common)

def norm(a):
    """ Compute square root of the dot product
    Args:
        a (dictionary): a dictionary of record to value
    Returns:
        norm: a dictionary of tokens to its TF values
    """
    return math.sqrt(sum(a[k]*a[k] for k in a))

def cossim(a, b, common):
    """ Compute cosine similarity
    Args:
        a (dictionary): first dictionary of record to value
        b (dictionary): second dictionary of record to value
        common (list): list of common values 
    Returns:
        cossim: dot product of two dictionaries divided by the norm of the first dictionary and
                then by the norm of the second dictionary
    """
    return dotprod(a,b,common)/(norm(a)*norm(b))

def invertToDict(record):
    """ Invert {ID, {tokens:TFIDF}} to a dict of {token: [ID]}
    Args:
        record: mapping of IDs to token:TDIDF 
    Returns:
        pairs: inverse mapping of token to IDs containing tokens
    """
    pairs = {}
    for entry in record:
        ID=entry
        tokens=record[ID].keys()
        for token in tokens:
            pairs[token] = pairs.get(token, []) + [ID] 
    return (pairs)

if __name__=="__main__":
    main()
