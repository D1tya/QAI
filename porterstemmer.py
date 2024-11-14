from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
ps = PorterStemmer()
sample = ['program', 'programmer', 'programming', 'programs', 'programmers']

for s in sample:
    print(f"{s} : {ps.stem(s)}")


sentence = "Programmers program with different programming languages"
words = word_tokenize(sentence)
print(words)

for w in words:
    print(f"{w} : {ps.stem(w)}")

import io
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
stop_words = set(stopwords.words('english'))
file1 = open("./data/sample.txt")

line = file1.read()

words = line.split()

print(f"Before Removing Stop Words : \n{words}\nLength : {len(words)}")
for r in words:
    if not r in stop_words:
        appendFile = open('./data/output_filteredtext.txt','a')
        appendFile.write(" "+r)
        appendFile.close()

file2 = open("./data/output_filteredtext.txt")

line2 = file2.read()

words = line2.split()

print(f"After Removing Stop Words : \n{words}\nLength : {len(words)}")