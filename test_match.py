#!/usr/bin/env python
import match 
import pytest #as Test
import unittest

#print match.dotprod(1,2,3)

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
def test_calculate_tf(description, corpus, idfscores):
    assert match.idf(corpus) == idfscores

def test_calculate_tfidf():
    pass

def test_dot_prod():
    pass

@pytest.mark.parametrize("description, vector, norm",[
            ("three non-zero components", {'foo': 2, 'bar': 3, 'baz': 5 }, 6.16441400297),
            ("two non-zero components", {'foo': 1, 'bar': 0, 'baz': 20 }, 20.0249843945),
])
def test_norm(description, vector, norm):
    assert match.norm(vector) - norm < 0.0000001
    
def test_cossim():
    pass

def test_invert_to_dict():
    pass

def test_find_common_tokens():
    pass

def test_match():
    pass
