import sys
import os
test_path = os.path.dirname(__file__)
sys.path.append(os.path.abspath(test_path+'/../'))
import theseus
import data



def test_theseus_node_smoke():
    thn = theseus.Node(data.documents)
    assert thn.counter.most_common(2) == [('Python', 4), ('R', 4)]

def test_get_frequency():
    thn = theseus.Node(data.documents)
    assert thn.get_frequencies(2) == {'Python': 0.05970149253731343, 'R': 0.05970149253731343}

def test_compare_doc_to_background():
    background = theseus.Node(data.documents)
    first = theseus.Node(data.documents[:1])

    assert first.get_frequencies(2) == {'Big Data': 0.14285714285714285, 'Hadoop': 0.14285714285714285}

def test_assert_xy_table():
    background = theseus.Node(data.documents)
    first = theseus.Node(data.documents[:1])

    x, y, keys = first.create_xy_table(background)
    lenkeys = len(keys)
    assert len(x) == lenkeys
    assert len(y) == lenkeys
    assert sorted(keys[:5]) == sorted(['Big Data', 'HBase', 'Hadoop', 'Java', 'Spark'])
    assert x[:3] ==  [0.14285714285714285, 0.14285714285714285, 0.14285714285714285]
    assert y[:3] == [0.029850746268656716, 0.04477611940298507, 0.04477611940298507]


data_root = 'tests/data/spam_example/'
spam = data.load_per_line_file(data_root+'spam.txt')
node_spam = theseus.Node(spam)

easy_ham = data.load_per_line_file(data_root+'easy_ham.txt')
hard_ham = data.load_per_line_file(data_root+'hard_ham.txt')
ham = easy_ham + hard_ham
node_ham = theseus.Node(ham)

def test_load_single_line_data():

    assert len(spam) == 498
    assert isinstance(spam, list)
    assert isinstance(spam[0], list)
    assert isinstance(spam[0][0], str)

    assert len(ham) == (2741 + 283)
    assert isinstance(spam, list)
    assert isinstance(spam[0], list)
    assert isinstance(spam[0][0], str)

def test_ratio_001_narrow_specific():
    ratio = 0.01
    assert node_spam.create_profile(node_ham, ratio=ratio)[:5] == ['free', 'rates', 'home', '&', 'systemworks']
    assert node_ham.create_profile(node_spam, ratio=ratio)[:5] == ['[satalk]', '(was', '[ilug]', 'bliss', 'selling']

def test_ratio_010_wider_but_still_distinct():
    ratio = 0.1
    assert node_spam.create_profile(node_ham, ratio=ratio)[:5] == ['your', 'free', 'rates', 'home', '&']
    assert node_ham.create_profile(node_spam, ratio=ratio)[:5] == ['[satalk]', '(was', '[ilug]', 'bliss', 'selling']

def test_ratio_035_distinct_but_pronouns_bleeding_in():
    ratio = 0.35
    assert node_spam.create_profile(node_ham, ratio=ratio)[:5] == ['your', 'you', '-', 'free', 'at']
    assert node_ham.create_profile(node_spam, ratio=ratio)[:5] == ['re:', '[satalk]', 'new', '(was', '[ilug]']

def test_ratio_001_semantic_meaning_lost_all_pronouns():
    ratio = 1.5
    assert node_spam.create_profile(node_ham, ratio=ratio)[:5] == ['the', 'your', 'of', 'for', 'to']
    assert node_ham.create_profile(node_spam, ratio=ratio)[:5] == ['the', 'for', 'of', 'to', 'in']

def test_visualize():
    background = theseus.Node(data.documents)
    first = theseus.Node(data.documents[:1], name='tests/output/theseus')

    first.visualize(background, magnification=3.0, viz=False)

def test_predict():
    """The graph shows the cutoff used, or 'angle' first.  Then the list that follows
        is the number of sentences from target group which are considered classified True
        for the cutoff ranging from 0 to 12
        At depth 0 everything matches since the cutoff is 0, meaning even no words matches.
    """
    node_spam.depth = 100
    # spam node versus ham node, looking at spam sentences
    tests = [ # number of required for a 'hit' ->
        [0.2, [498, 309, 164, 82, 42, 19, 3, 1, 1, 1, 1, 0]],
        [0.3, [498, 312, 169, 84, 45, 19, 7, 1, 1, 1, 1, 1]],
        [0.4, [498, 326, 188, 103, 49, 23, 7, 1, 1, 1, 1, 1]],
        [0.5, [498, 326, 188, 103, 49, 23, 7, 1, 1, 1, 1, 1]],
        [0.6, [498, 332, 196, 106, 52, 24, 8, 1, 1, 1, 1, 1]],
        [0.7, [498, 351, 205, 121, 58, 27, 8, 1, 1, 1, 1, 1]],
        [0.8, [498, 356, 216, 128, 68, 29, 9, 1, 1, 1, 1, 1]],
        [0.9, [498, 357, 222, 129, 72, 29, 9, 1, 1, 1, 1, 1]],
        [1.0, [498, 380, 250, 157, 87, 31, 15, 4, 1, 1, 1, 1]],
        [1.1, [498, 382, 256, 159, 92, 33, 15, 4, 1, 1, 1, 1]],
    ]
    for ratio, hit_list in tests:
        node_spam.create_profile(node_ham, ratio=ratio)
        hits = []
        for cutoff in range(12):
            node_spam.cutoff = cutoff
            hits.append(sum([1 for sentence in spam if node_spam.predict(sentence)]))
        assert hits == hit_list

    # spam node versis ham node, looking at ham sentences
    tests2 = [
        [0.2, [3024, 292, 31, 3, 1, 0, 0, 0, 0, 0, 0, 0]],
        [0.3, [3024, 316, 35, 3, 1, 0, 0, 0, 0, 0, 0, 0]],
        [0.4, [3024, 411, 54, 6, 1, 0, 0, 0, 0, 0, 0, 0]],
        [0.5, [3024, 411, 54, 6, 1, 0, 0, 0, 0, 0, 0, 0]],
        [0.6, [3024, 483, 77, 9, 1, 0, 0, 0, 0, 0, 0, 0]],
        [0.7, [3024, 784, 145, 15, 3, 0, 0, 0, 0, 0, 0, 0]],
        [0.8, [3024, 865, 215, 31, 3, 1, 0, 0, 0, 0, 0, 0]],
        [0.9, [3024, 903, 241, 37, 5, 1, 0, 0, 0, 0, 0, 0]],
        [1.0, [3024, 1241, 510, 131, 29, 6, 3, 0, 0, 0, 0, 0]],
        [1.1, [3024, 1304, 531, 141, 33, 9, 3, 0, 0, 0, 0, 0]],
    ]
    for ratio, hit_list in tests2:
        node_spam.create_profile(node_ham, ratio=ratio)
        hits = []
        for cutoff in range(12):
            node_spam.cutoff = cutoff
            hits.append(sum([1 for sentence in ham if node_spam.predict(sentence)]))
        assert hits == hit_list

def test_not_uniqify():
    """with no uniqify flag the xy tables are created as a fraction of per group.
        i.e.
            a a a a a a a a a a
            b b b b b
            c
        would yeild 10a's, 5b's and 1 c.  For  total of 16 words.
        Thus the a ratio is 10/16 = 0.625
    """
    data_root = 'tests/data/uniquify/'
    background = 'background.txt'
    back = theseus.Node()
    back.load_file(data_root+background)
    abc = theseus.Node()
    abc.load_file(data_root+'abc.txt')

    x, y, keys = abc.create_xy_table(back, ratio=0.1)
    assert round(x[0], 3)    == 0.625
    assert round(y[0], 3)    == 0.018
    assert keys[0]           == 'a'

def test_uniqify():
    """with no uniqify flag the xy tables are created as a fraction of per group.
        i.e.
            a a a a a a a a a a
            b b b b b
            c
        would yeild 10a's, 5b's and 1 c.  For  total of 16 words.
        Thus the a ratio is 10/16 = 0.625
    """
    data_root = 'tests/data/uniquify/'
    background = 'background.txt'
    back = theseus.Node()
    back.load_file(data_root+background, uniquify=True)
    abc = theseus.Node()
    abc.load_file(data_root+'abc.txt', uniquify=True)

    x, y, keys = abc.create_xy_table(back, ratio=0.35)
    for idx, point in enumerate(zip(x,y,keys)):
        print(idx, point)

    assert round(x[0], 3)    == 0.333
    assert round(y[0], 3)    == 0.1
    assert keys[0]           == 'a'
