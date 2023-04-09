import os
import sys
import argparse
import math
import numpy as np

# Set of all possible tags
TAGS = ["AJ0", "AJC", "AJS", "AT0", "AV0", "AVP", "AVQ", "CJC", "CJS", "CJT", "CRD",
        "DPS", "DT0", "DTQ", "EX0", "ITJ", "NN0", "NN1", "NN2", "NP0", "ORD", "PNI",
        "PNP", "PNQ", "PNX", "POS", "PRF", "PRP", "PUL", "PUN", "PUQ", "PUR", "TO0",
        "UNC", 'VBB', 'VBD', 'VBG', 'VBI', 'VBN', 'VBZ', 'VDB', 'VDD', 'VDG', 'VDI',
        'VDN', 'VDZ', 'VHB', 'VHD', 'VHG', 'VHI', 'VHN', 'VHZ', 'VM0', 'VVB', 'VVD',
        'VVG', 'VVI', 'VVN', 'VVZ', 'XX0', 'ZZ0', 'AJ0-AV0', 'AJ0-VVN', 'AJ0-VVD',
        'AJ0-NN1', 'AJ0-VVG', 'AVP-PRP', 'AVQ-CJS', 'CJS-PRP', 'CJT-DT0', 'CRD-PNI', 'NN1-NP0', 'NN1-VVB',
        'NN1-VVG', 'NN2-VVZ', 'VVD-VVN', 'AV0-AJ0', 'VVN-AJ0', 'VVD-AJ0', 'NN1-AJ0', 'VVG-AJ0', 'PRP-AVP',
        'CJS-AVQ', 'PRP-CJS', 'DT0-CJT', 'PNI-CRD', 'NP0-NN1', 'VVB-NN1', 'VVG-NN1', 'VVZ-NN2', 'VVN-VVD']

# mark of end of sentence
EOS = ['.', '!', '?', ';']

PUL = ['(', '[']

PUN = ['.', '!', '?', ';', ',', ':', '-']

PUQ = ['"']

PUR = [')', ']']

class HMM:
    def __init__(self, tag_set, train_set):
        self.tag_set = tag_set
        self.train_set = train_set
        self.tag_len = len(tag_set)
        self.train_len = len(train_set)

        # convert b/w tag and index
        self.tag2idx = {tag: i for i, tag in enumerate(tag_set)} # 'AJ0': 0, 'AJC': 1, 'AJS': 2,
        self.idx2tag = {i: tag for i, tag in enumerate(tag_set)} # 0: 'AJ0', 1: 'AJC'
        
        # init initial, transition prob matrix
        self.ip = None
        self.tp = None

        # init observation prob dict
        self.tag_word_dict = [{} for _ in range(self.tag_len)]

        # frequency counting
        self.freq_tag = None
        self.most_freq = int()

    def get_index(self, tag):
        return self.tag2idx[tag]
    
    def get_tag(self, index):
        return self.idx2tag[index]

    def init_prob_calc(self):
        tag_count = np.ones((self.tag_len,))
        freq_count = np.zeros((self.tag_len,))
        read_next = True

        # at eos, record next tag to tag_count
        for i in range(self.train_len):
            temp_word = self.train_set[i][0]
            temp_tag = self.train_set[i][1]
            freq_count[self.get_index(temp_tag)] += 1
            if (read_next):
                tag_count[self.get_index(temp_tag)] += 1
                read_next = False
            if (temp_word in EOS):
                read_next = True

        # normalize vector and set to init_prob
        self.ip = tag_count / np.sum(tag_count)
        self.freq_tag = freq_count

        self.freq_tag[self.get_index('PUN')] = 0
        self.freq_tag[self.get_index('PUR')] = 0
        self.freq_tag[self.get_index('PUL')] = 0
        self.freq_tag[self.get_index('PUQ')] = 0
        self.most_freq = np.argmax(self.freq_tag)

    def trans_prob_calc(self):
        result = np.ones((self.tag_len, self.tag_len)) # (from tag, to tag)
        curr, prev = '', ''

        # access curr tag -> record -> make curr to prev -> loop
        for i in range(self.train_len):
            curr = self.train_set[i][1]
            if (prev):
                result[self.get_index(prev)][self.get_index(curr)] += 1
            prev = curr
            
        # normalize vector and set to trans_prob
        # add eps to avoid divide by zero
        eps = np.finfo(result.dtype).eps
        arr_norm = np.linalg.norm(result, axis=1, keepdims=True)
        self.tp = result / np.maximum(arr_norm, eps)

    def observ_prob_calc(self):
        # access curr tag & word -> record to dict -> loop to next
        for i in range(self.train_len):
            idx = self.get_index(self.train_set[i][1])
            word = self.train_set[i][0].lower()

            # find word in dict, if yes increment, if no add key
            if word in self.tag_word_dict[idx]:
                self.tag_word_dict[idx][word] += 1
            else:
                self.tag_word_dict[idx][word] = 1
            
        # normalize dict and set to observ_prob
        # skip dict with no value
        for i in range(self.tag_len):
            tot = sum(self.tag_word_dict[i].values())
            if (tot == 0):
                continue
            for key in self.tag_word_dict[i]:
                self.tag_word_dict[i][key] /= tot

    def train_init(self):
        # Calculate initial prob
        self.init_prob_calc()

        # Calculate trans prob
        self.trans_prob_calc()

        # Calculate observ prob
        self.observ_prob_calc()
    
    def training(self, words):
        words_len = len(words)
        seen = False

        # initialization
        delta = np.zeros((self.tag_len, words_len))
        for i in range(self.tag_len):
            if (words[0].lower() in self.tag_word_dict[i]):
                delta[i, 0] = self.ip[i] * self.tag_word_dict[i][words[0].lower()]
                if (delta[i, 0] != 0):
                    seen = True
            else:
                delta[i, 0] = 0

        # if never seen word exist, directly use init prob
        if (seen is False):
            delta[:, 0] = self.ip

        # normalization
        delta[:, 0] /= np.sum(delta[:, 0])

        # recursion
        sos = False # start of sentence 
        
        for t in range(1, words_len):
            seen3 = False
            seen2 = True
            enter_sos = False

            # if beginning of sentence, use init_prob
            if (sos):
                # hardcode symbols
                if (words[t] in PUL):
                    delta[self.get_index('PUL'), t] = 1
                elif (words[t] in PUN):
                    delta[self.get_index('PUN'), t] = 1
                elif (words[t] in PUQ):
                    delta[self.get_index('PUQ'), t] = 1
                elif (words[t] in PUR):
                    delta[self.get_index('PUR'), t] = 1
                else:
                    enter_sos = True
                    for i in range(self.tag_len):
                        if (words[t].lower() in self.tag_word_dict[i]):
                            delta[i, t] = self.ip[i] * self.tag_word_dict[i][words[t].lower()]
                            if (delta[i, t] != 0):
                                seen3 = True
                        else:
                            delta[i, t] = 0

                    # if never seen word exist, directly use init prob
                    if (seen3 is False):
                        delta[:, t] = self.ip

                    # normalization
                    delta[:, t] /= np.sum(delta[:, t])

            # update start of sentence status
            if (words[t] in EOS):
                sos = True
            else:
                sos = False
            
            # leaving sos, goto next loop
            if (enter_sos):
                continue

            # hardcode symbols
            if (words[t] in PUL):
                delta[self.get_index('PUL'), t] = 1
            elif (words[t] in PUN):
                delta[self.get_index('PUN'), t] = 1
            elif (words[t] in PUQ):
                delta[self.get_index('PUQ'), t] = 1
            elif (words[t] in PUR):
                delta[self.get_index('PUR'), t] = 1

            # if not symbol, but words
            else:  
                seen2 = False
                for s in range(self.tag_len):
                    if (words[t].lower() in self.tag_word_dict[s]):
                        seen2 = True
                        delta[s, t] = np.max(delta[:, t-1] * self.tp[:, s]) * self.tag_word_dict[s][words[t].lower()]
                    else:
                        delta[s, t] = 0

                # if never seen word exist, directly use trans prob
                if (seen2 is False):
                    best_idx_from_prev = np.argmax(delta[:, t-1])
                    delta[:, t] = self.tp[best_idx_from_prev]
                
            # normalization - add eps to avoid divide by zero
            eps = np.finfo(delta[:, t].dtype).eps
            arr_norm = np.linalg.norm(delta[:, t], axis=0, keepdims=True)
            delta[:, t] /= np.maximum(arr_norm, eps)

            # delta[:, t] /= np.sum(delta[:, t])
        return delta
    
    def prediction(self, words, delta):
        words_len = len(words)
        # backtrack
        states = np.zeros(words_len, dtype=np.int32)
        states[words_len-1] = np.argmax(delta[:, words_len-1])
        for t in range(words_len-2, -1, -1):
            # hardcode symbols
            if (words[t] in PUL):
                states[t] = self.get_index('PUL')
            elif (words[t] in PUN):
                states[t] = self.get_index('PUN')
            elif (words[t] in PUQ):
                states[t] = self.get_index('PUQ')
            elif (words[t] in PUR):
                states[t] = self.get_index('PUR')
            # handle other words, no PUNs should be predicted
            else:
                states[t] = np.argmax(delta[:, t] * self.tp[:, states[t+1]])
                while self.get_tag(states[t]) in ['PUL', 'PUN', 'PUQ', 'PUR']:
                    delta[states[t], t] = 0
                    states[t] = np.argmax(delta[:, t] * self.tp[:, states[t+1]])
                # if argmax still zero, choose the most frequent
                if delta[states[t], t] == 0:
                    states[t] = self.most_freq

        tags = np.empty(words_len, dtype=object)
        for k in range(words_len):
            tags[k] = self.get_tag(states[k])
        
        # make 2d array using predicted tags and given test words
        output = np.stack((words, tags), axis=1)
        return output

# read train data from file list to generate list type
def read_train_data(file):
    output = []
    for item in file:
        with open(item, 'r') as f:
            lines = f.readlines()
            data = [line.strip().split(' : ') for line in lines]
            output.extend(data)
    return output

# read test data to generate list type
def read_test_data(file):
    with open(file, 'r') as f:
        lines = f.readlines()
        data = [line.strip() for line in lines]
    return data

# write data list to the output file as expected format
def write_data(file, data_dict):
    with open(file, 'w') as f:
        for word, tag in data_dict:
            f.write(word + ' : ' + tag + '\n')

def check_accuracy(list1, list2):
    a = [lst[1] for lst in list1]
    b = [lst[1] for lst in list2]

    count_same = 0
    count_diff = 0
    
    # Count the number of True and False values
    for i in range(min(len(a), len(b))):
        if a[i] == b[i]:
            count_same += 1
        else:
            count_diff += 1
            # print("Line ", i+1, " Expected: ", a[i], ", Got ", b[i])

    print("Accuracy: ", count_same / (count_diff + count_same))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--trainingfiles",
        action="append",
        nargs="+",
        required=True,
        help="The training files."
    )
    parser.add_argument(
        "--testfile",
        type=str,
        required=True,
        help="One test file."
    )
    parser.add_argument(
        "--outputfile",
        type=str,
        required=True,
        help="The output file."
    )
    args = parser.parse_args()

    training_list = args.trainingfiles[0]

    # Output function
    train_data = read_train_data(training_list)
    test_data = read_test_data(args.testfile)

    tagger = HMM(TAGS, train_data)
    tagger.train_init()
    delta = tagger.training(test_data)
    predicted_tags = tagger.prediction(test_data, delta)
    write_data(args.outputfile, predicted_tags)

    correct_ans = read_train_data(["train1.txt"])
    check_accuracy(correct_ans, predicted_tags)
