import os
import sys
import codecs
import numpy
import array as arr
from decimal import *
from tkinter import *
import tkinter.messagebox
from sklearn.metrics import confusion_matrix, accuracy_score

tag_list = set()
tag_list2 = set()
tag_count = {}
number_tag = {}
word_set = set()
word_set2 = set()
x = 0
z = 0
x2 = 0
z2 = 0
expect = arr.array('i')
predict = arr.array('i')


def parse_traindata():
    fin = "hi_test_tagged.txt"
    output_file = "hmmmodel.txt"
    wordtag_list = []
    try:
        input_file = codecs.open(fin, mode='r', encoding="utf-8")
        lines = input_file.readlines()
        for line in lines:
            line = line.strip('\n')
            # returns a list of all the words in the string
            data = line.split(" ")
            # appends a passed obj into the existing list.
            wordtag_list.append(data)

        input_file.close()
        #print(wordtag_list)
        return wordtag_list

    except IOError:
        fo = codecs.open(output_file, mode='w', encoding="utf-8")
        fo.write("File not found: {}".format(fin))
        fo.close()
        sys.exit()


def parse_traindata2():
    fin = "hmmmoutput.txt"
    # output_file = "hmmmodel.txt"
    wordtag_list2 = []

    input_file = codecs.open(fin, mode='r', encoding="utf-8")
    lines = input_file.readlines()
    for line in lines:
            line = line.strip('\n')
            # returns a list of all the words in the string
            data = line.split(" ")
            # appends a passed obj into the existing list.
            wordtag_list2.append(data)

    input_file.close()
    return wordtag_list2


def transition_count():
    global tag_list
    global word_set
    global number_tag
    global x
    global z
    global u
    global expect
    train_data = parse_traindata()
    transition_dict = {}
    global tag_count
    for value in train_data:
        previous = "start"
        for data in value:
            i = data[::-1]
            word = data[:-i.find("/") - 1]
            word_set.add(word.lower())
            data = data.split("/")
            tag = data[-1]
            tag_list.add(tag)
            if tag in tag_count:
                tag_count[tag] += 1
            else:
                tag_count[tag] = 1
                x += 1
                number_tag[tag] = x
            if tag in number_tag:
                z = number_tag.get(tag)
                expect.append(z)
            if (previous + "~tag~" + tag) in transition_dict:
                transition_dict[previous + "~tag~" + tag] += 1
                previous = tag
            else:
                transition_dict[previous + "~tag~" + tag] = 1
                previous = tag
    print(len(expect))
    print(expect)
    return transition_dict


def transition_count2():
    global tag_list2
    global word_set2
    global predict
    global x2
    global z2
    train_data = parse_traindata2()
    global tag_count
    for value in train_data:
        previous = "start"
        for data in value:
            i = data[::-1]
            word = data[:-i.find("/") - 1]
            word_set2.add(word.lower())
            data = data.split("/")
            tag = data[-1]
            tag_list2.add(tag)
            if tag in tag_count:
                tag_count[tag] += 1
            else:
                tag_count[tag] = 1
                x2 += 1
                number_tag[tag] = x2
            if tag in number_tag:
                z2 = number_tag.get(tag)
                predict.append(z2)
    print(len(predict))
    print(predict)


def transition_probability():
    count_dict = transition_count()
    prob_dict = {}
    for key in count_dict:
        den = 0
        val = key.split("~tag~")[0]
        for key_2 in count_dict:
            if key_2.split("~tag~")[0] == val:
                den += count_dict[key_2]
        prob_dict[key] = Decimal(count_dict[key])/den
    return prob_dict


def transition_smoothing():
    transition_prob = transition_probability()
    for tag in tag_list:
        if "start" + tag not in transition_prob:
            transition_prob[("start" + "~tag~" + tag)] = Decimal(1) / Decimal(len(word_set) + tag_count[tag])
    for tag1 in tag_list:
        for tag2 in tag_list:
            if (tag1+"~tag~"+tag2) not in transition_prob:
                transition_prob[(tag1+"~tag~"+tag2)] = Decimal(1)/Decimal(len(word_set) + tag_count[tag1])
    return transition_prob


def emission_count():
    train_data = parse_traindata()
    count_word = {}
    for value in train_data:
        for data in value:
            i = data[::-1]
            word = data[:-i.find("/") - 1]
            tag = data.split("/")[-1]
            if word.lower() + "/" + tag in count_word:
                count_word[word.lower() + "/" + tag] += 1
            else:
                count_word[word.lower() + "/" + tag] = 1
    return count_word


def emission_probability():
    global tag_count
    word_count = emission_count()
    emission_prob_dict = {}
    for key in word_count:
        emission_prob_dict[key] = Decimal(word_count[key])/tag_count[key.split("/")[-1]]
    return emission_prob_dict


def main():
        global tag_count
        transition_model = transition_smoothing()
        emission_model = emission_probability()

        fout = codecs.open("hmmmodel.txt", mode='w', encoding="utf-8")
        for key, value in transition_model.items():
            fout.write('%s:%s\n' % (key, value))

        fout.write('Emission Model\n')
        for key, value in emission_model.items():
            fout.write('%s:%s\n' % (key, value))
        os.system("python hmmdecode.py")
        transition_count2()
        results = confusion_matrix(numpy.array(expect), numpy.array(predict))
        print(results)
        print(number_tag)
        q = accuracy_score(numpy.array(expect), numpy.array(predict))
        q = q*100
        j = str(q)
        x1 = "ACCURACY OBTAINED  = "
        x7 = x1 + j
        tkinter.messagebox.showinfo(" Run Successfully ", x7)
        print("done")


if __name__ == '__main__':
    main()
