#!/usr/bin/env python
import match 
import pytest 
import unittest

@pytest.mark.parametrize("description, string, tokens",[
    ('Tokenize should handle text', 'A quick brown fox jumps over the lazy dog.', ['a','quick','brown','fox','jumps','over','the','lazy','dog']),
    ('Tokenize should handle empty string', ' ', []),
    ('Tokenize should handle puntuations and lowercase', '!!!!123A/456_B/789C.123A', ['123a','456', 'b','789c','123a']),
    ('Duplicates should be removed' , 'fox fox' , ['fox', 'fox']),
])
def test_tokenize(description, string, tokens):
    assert match.tokenize(string) == tokens

@pytest.mark.parametrize("description, tokens, tfscores",[
    ('Test sentence, no duplicates', ['one', 'two', 'three','four','five', 'six'], {'one': 0.16666666666666666, 'two': 0.16666666666666666, \
                             'three': 0.16666666666666666, 'four': 0.16666666666666666, \
                             'five': 0.16666666666666666, 'six': 0.16666666666666666}),
    ('Test sentence, with duplicates', ['one', 'one', 'two'], {'one': 0.6666666666666666, 'two': 0.3333333333333333}),
])
def test_calculate_tf(description, tokens, tfscores):
    assert match.tf(tokens) == tfscores

@pytest.mark.parametrize("description, corpus, idfscores",[
    ('Test sentence, no duplicates', ['one', 'two', 'three','four','five', 'six'], {'one': 6.0, 'two': 6.0, \
                             'three': 6.0, 'four': 6.0, \
                             'five': 6.0, 'six': 6.0}),
    ('Test sentence, with duplicates', ['one', 'one', 'two'], {'one': 1.5, 'two': 3.0}),
])
def test_calculate_idf(description, corpus, idfscores):
    assert match.idf(corpus) == idfscores

@pytest.mark.parametrize("description, tfscores, idfscores, expectedtfidfs" ,[
            ('Tf subset of idfs', {'input' : {'one': 6.0, 'two': 5.0}}, {'one' : 7.0, 'two' : 4.0, 'three': 6.0}, \
                                             {'input' : {'one': 42.0, 'two': 20.0}}),
            ('Tf same set as idfs', {'input' : {'one': 6.0, 'two': 5.0}, 'otherinput': {'three' : 3.0}}, {'one' : 7.0, 'two' : 4.0, 'three': 6.0}, \
                                             {'input' : {'one': 42.0, 'two': 20.0}, 'otherinput' : {'three' : 18.0}}),
])
def test_calculate_tfidf(description, tfscores, idfscores, expectedtfidfs):
    tfidfs = match.tfidf(tfscores, idfscores)
    assert tfidfs.keys() == expectedtfidfs.keys()
    for key in tfidfs.keys():
        assert tfidfs[key] == expectedtfidfs[key]  

@pytest.mark.parametrize("description, vec1, vec2, common, expected",[
            ('Same components', {'A' : 2, 'B' : 3, 'C' : 4 }, {'A' : 1 , 'B' : 2, 'C' : 3}, ['A', 'B', 'C'], 20),
            ('different components', {'A' : 3 ,'B' : 2, 'C' : 1}, {'A' : 2, 'D' : 3, 'E' : 4}, ['A'], 6),
])
def test_dot_prod(description, vec1, vec2, common, expected):
    assert match.dotprod(vec1, vec2, common) == expected

@pytest.mark.parametrize("description, vector, norm",[
            ("three non-zero components", {'foo': 2, 'bar': 3, 'baz': 5 }, 6.16441400297),
            ("two non-zero components", {'foo': 1, 'bar': 0, 'baz': 20 }, 20.0249843945),
])
def test_norm(description, vector, norm):
    assert match.norm(vector) - norm < 0.0000001
    
@pytest.mark.parametrize("description, vec1, vec2, common, expected",[
            ('Same components', {'A' : 1, 'B' : 2, 'C' : 3 }, {'A' : 1 , 'B' : 2, 'C' : 3}, ['A', 'B', 'C'], 1),
            ('different components', {'A' : 3 ,'B' : 2, 'C' : 1}, {'A' : 2, 'D' : 3, 'E' : 4}, ['A'],  0.2977750),
])
def test_cossim(description, vec1, vec2, common, expected):
    assert match.cossim(vec1, vec2, common) - expected < 0.0000001

@pytest.mark.parametrize("description, inputDict, outputDict",[
            ("no overlapping tokens", {'foo': {'footken' : 1}, 'bar': {'bartoken' : 1}, 'baz': {'baztoken': 1} }, \
                                        {'footken' : ['foo'], 'bartoken' : ['bar'], 'baztoken' : ['baz']}),
            ("sharing common tokens", {'foo': {'spam' : 1, 'eggs' : 1 }, 'bar': {'spam' : 1}, 'baz': {'eggs' : 1} }, \
                                        {'spam' : ['foo', 'bar'], 'eggs' : ['foo', 'baz']} ),
])
def test_invert_to_dict(description, inputDict, outputDict):
    invertedInput = match.invertToDict(inputDict)
    assert sorted(invertedInput.keys()) == sorted(outputDict.keys())
    for key in invertedInput.keys():
        assert sorted(invertedInput[key]) == sorted(outputDict[key])

@pytest.mark.parametrize("description, inputA, inputB, outputDict",[
            ("Only one token per key pair", {'token1': ['key1'], 'token2': ['key2'], 'token3': ['key3'] }, \
                                            {'token1': ['keyA'], 'token2': ['keyB'], 'token3': ['keyC'] }, \
                                            {('key1', 'keyA') : ['token1'], ('key2', 'keyB') : ['token2'], ('key3', 'keyC') : ['token3']}),\
            ("Multiple key pairs per token", {'token1': ['key1'], 'token2': ['key2'], 'token3': ['key3'] }, \
                                            {'token1': ['keyA'], 'token2': ['keyB', 'keyA'], 'token3': ['keyC'] }, \
                                            {('key1', 'keyA') : ['token1'], ('key2', 'keyB') : ['token2'], \
                                            ('key3', 'keyC') : ['token3'], ('key2', 'keyA') : ['token2']}), \
            ("Multiple tokens per key pair", {'token1': ['key1'], 'token2': ['key2'], 'token3': ['key1'] }, \
                                            {'token1': ['keyA'], 'token2': ['keyB'], 'token3': ['keyA'] }, \
                                            {('key1', 'keyA') : ['token1', 'token3'], ('key2', 'keyB') : ['token2']}),
])
def test_find_common_tokens(description, inputA, inputB, outputDict):
    commonTokenDict = match.findCommonTokens(inputA, inputB)
    assert commonTokenDict.keys() == outputDict.keys()
    for key in commonTokenDict.keys():
        assert sorted(commonTokenDict[key]) == sorted(outputDict[key])

@pytest.mark.parametrize("description, vec1, vec2, common, expected",[
            ('Same components', {'product' : {'A' : 1, 'B' : 2, 'C' : 3 }}, {'listing': {'A' : 1 , 'B' : 2, 'C' : 3}}, \
                                { ('product', 'listing'): ['A', 'B', 'C']}, { 'product' : ['listing', 1.0]}),
            ('different components', {'product': {'A' : 4 ,'B' : 2, 'C' : 5}}, {'listing': {'A' : 2, 'D' : 3, 'E' : 4}}, \
                                 { ('product', 'listing'):['A'] },  {}),
])
def test_match(description, vec1, vec2, common, expected):
    assert match.match(vec1, vec2, common) == expected
