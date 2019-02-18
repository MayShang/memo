import tensorflow as tf
import os
import zipfile
import collections


local_filename = os.path.join('/tmp', 'text8.zip')

def read_data(filename):
    # if filename.endswith('zip') or filename.endswith('gz'):
    if zipfile.is_zipfile(filename):
        with zipfile.ZipFile(filename) as f:
            data = tf.compat.as_str(f.read(f.namelist()[0])).split()
    else:
        with open(filename, 'r') as f:
            data = tf.compat.as_str(f.read()).split()
    return data

def build_dataset(words, n_words):
    count = [['UNK', -1]]
    count.extend(collections.Counter(words).most_common(n_words - 1))
    dictionary = dict()

    for word, _ in count:
        dictionary[word] = len(dictionary)

    data = list()
    unk_count = 0
    for word in words:
        index = dictionary.get(word, 0)
        if index == 0:
            unk_count += 1
        data.append(index)
    count[0][1] = unk_count
    reversed_dictionary = dict(zip(dictionary.values(), dictionary.keys()))

    return data, count, dictionary, reversed_dictionary 

def refresh():
    print('refresh')

def main():
    print('hello')

if __name__ == '__main__':
    tf.app.run()
