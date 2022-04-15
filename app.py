#import files
from flask import Flask, render_template, request
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory, StopWordRemover, ArrayDictionary
import string
import re
from nltk.corpus import wordnet as wn
from flask_cors import CORS, cross_origin
import logging

class BoyerMoore(object):
    """ Encapsulates pattern and associated Boyer-Moore preprocessing. """

    def __init__(self, p, alphabet='ABCDEFGHIJKLMNOPQRSTUVWXYZ'):
        # Create map from alphabet characters to integers
        self.amap = {alphabet[i]: i for i in range(len(alphabet))}
        # Make bad character rule table
        self.bad_char = dense_bad_char_tab(p, self.amap)
        # Create good suffix rule table
        _, self.big_l, self.small_l_prime = good_suffix_table(p)

    def bad_character_rule(self, i, c):
        """ Return # skips given by bad character rule at offset i """
        assert c in self.amap
        assert i < len(self.bad_char)
        ci = self.amap[c]
        return i - (self.bad_char[i][ci]-1)

    def good_suffix_rule(self, i):
        """ Given a mismatch at offset i, return amount to shift
            as determined by (weak) good suffix rule. """
        length = len(self.big_l)
        assert i < length
        if i == length - 1:
            return 0
        i += 1  # i points to leftmost matching position of P
        if self.big_l[i] > 0:
            return length - self.big_l[i]
        return length - self.small_l_prime[i]

    def match_skip(self):
        """ Return amount to shift in case where P matches T """
        return len(self.small_l_prime) - self.small_l_prime[1]

# End Class Boyermoore

def z_array(s):
    """ Use Z algorithm (Gusfield theorem 1.4.1) to preprocess s """
    assert len(s) > 1
    z = [len(s)] + [0] * (len(s)-1)

    # Initial comparison of s[1:] with prefix
    for i in range(1, len(s)):
        if s[i] == s[i-1]:
            z[1] += 1
        else:
            break

    r, l = 0, 0
    if z[1] > 0:
        r, l = z[1], 1

    for k in range(2, len(s)):
        assert z[k] == 0
        if k > r:
            # Case 1
            for i in range(k, len(s)):
                if s[i] == s[i-k]:
                    z[k] += 1
                else:
                    break
            r, l = k + z[k] - 1, k
        else:
            # Case 2
            # Calculate length of beta
            nbeta = r - k + 1
            zkp = z[k - l]
            if nbeta > zkp:
                # Case 2a: zkp wins
                z[k] = zkp
            else:
                # Case 2b: Compare characters just past r
                nmatch = 0
                for i in range(r+1, len(s)):
                    if s[i] == s[i - k]:
                        nmatch += 1
                    else:
                        break
                l, r = k, r + nmatch
                z[k] = r - k + 1
    return z

def n_array(s):
    """ Compile the N array (Gusfield theorem 2.2.2) from the Z array """
    return z_array(s[::-1])[::-1]

def big_l_prime_array(p, n):
    """ Compile L' array (Gusfield theorem 2.2.2) using p and N array.
        L'[i] = largest index j less than n such that N[j] = |P[i:]| """
    lp = [0] * len(p)
    for j in range(len(p)-1):
        i = len(p) - n[j]
        if i < len(p):
            lp[i] = j + 1
    return lp

def big_l_array(p, lp):
    """ Compile L array (Gusfield theorem 2.2.2) using p and L' array.
        L[i] = largest index j less than n such that N[j] >= |P[i:]| """
    l = [0] * len(p)
    l[1] = lp[1]
    for i in range(2, len(p)):
        l[i] = max(l[i-1], lp[i])
    return l

def small_l_prime_array(n):
    """ Compile lp' array (Gusfield theorem 2.2.4) using N array. """
    small_lp = [0] * len(n)
    for i in range(len(n)):
        if n[i] == i+1:  # prefix matching a suffix
            small_lp[len(n)-i-1] = i+1
    for i in range(len(n)-2, -1, -1):  # "smear" them out to the left
        if small_lp[i] == 0:
            small_lp[i] = small_lp[i+1]
    return small_lp

def good_suffix_table(p):
    """ Return tables needed to apply good suffix rule. """
    n = n_array(p)
    lp = big_l_prime_array(p, n)
    return lp, big_l_array(p, lp), small_l_prime_array(n)

def good_suffix_mismatch(i, big_l_prime, small_l_prime):
    """ Given a mismatch at offset i, and given L/L' and l' arrays,
        return amount to shift as determined by good suffix rule. """
    length = len(big_l_prime)
    assert i < length
    if i == length - 1:
        return 0
    i += 1  # i points to leftmost matching position of P
    if big_l_prime[i] > 0:
        return length - big_l_prime[i]
    return length - small_l_prime[i]

def good_suffix_match(small_l_prime):
    """ Given a full match of P to T, return amount to shift as
        determined by good suffix rule. """
    return len(small_l_prime) - small_l_prime[1]

def dense_bad_char_tab(p, amap):
    """ Given pattern string and list with ordered alphabet characters, create
        and return a dense bad character table.  Table is indexed by offset
        then by character. """
    tab = []
    nxt = [0] * len(amap)
    for i in range(0, len(p)):
        c = p[i]
        assert c in amap
        tab.append(nxt[:])
        nxt[amap[c]] = i+1
    return tab

# p = inputan, p_bm = preprocessing, t = text/pertanyaan
def boyer_moore(p, p_bm, t):
    """ Do Boyer-Moore matching """
    # i adalah untuk angka increment
    i = 0
    # occurrences adalah hasil matching, nanti akan ada tandanya. [1,2,3]
    occurrences = []
    while i < len(t) - len(p) + 1:
        # shift untuk increment per loop
        shift = 1
        # mismatched adalah sebagai tanda jika tidak ada yg sama
        mismatched = False
        # len(p)-1 adalah panjang pattern dikurangi 1 #-1 kedua adalah jika angka sudah dibawah 0 maka berhenti #-1 ketiga adalah nilai per loop nya berarti dikurangi 1
        for j in range(len(p)-1, -1, -1):
            if p[j] != t[i+j]:# jika pattern huruf ke-j tidak sama dengan text huruf ke-i+j
                # coba bad char rule
                skip_bc = p_bm.bad_character_rule(j, t[i+j])
                # coba good suffix rule
                skip_gs = p_bm.good_suffix_rule(j)
                # mencari angka / index terbesar dari bc/gs rule
                shift = max(shift, skip_bc, skip_gs)
                # huruf nya tidak cocok = true
                mismatched = True
                break
        # jika matched
        if not mismatched:
            # tambahkan angka index ke occurrences
            occurrences.append(i)
            # skip character
            skip_gs = p_bm.match_skip()
            # cari index yg terbesar
            shift = max(shift, skip_gs)
        # incrementing (0+1)
        i += shift
    # me return hasil matching
    return occurrences

app = Flask(__name__)
# untuk fix error cors
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'

# the stopword by Sastrawi
stop_factory = StopWordRemoverFactory().get_stop_words()

# fitur tambah kata stopword
more_stopword = []

# proses merging kata stopword sastrawi dengan kata yg kita tambah
data = stop_factory + more_stopword

dictionary = ArrayDictionary(data)

stopword = StopWordRemover(dictionary)
# end stopword

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/get")
@cross_origin()
def get_bot_response():
    # menghapus kata-kata yg tidak penting menggunakan stopword by Sastrawi
    pattern = stopword.remove(request.args.get('msg').lower()) # "pattern" - thing we search for

    # menghilangkan simbol simbol di pertanyaan yg user input seperti tanda tanya, dll
    pattern = re.sub("[!@#$`'’~%^&*()-_=+.,;:|}{?/><]", '', pattern)

    # jika user menginput hanya dua huruf atau kurang, maka langsung munculkan "Aku Tidak Mengerti"
    if len(pattern) <= 2:
        return str('Aku tidak mengerti')

    # mengambil data pertanyaan dan jawaban di file txt question answer dan memisahkan antara pertanyaan dengan jawaban
    questionAnswer = open("question-answer.txt", "r", encoding="utf8").read().replace("\n", "|").split('|')

    # membuat objek boyermoore
    p_bm = BoyerMoore(pattern, alphabet='abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ ')

    # penyamaan string dari user input, kepada tiap tiap pertanyaan yg ada di txt questionAnswer
    for index, item in enumerate(questionAnswer):
        # menghilangkan simbol simbol di pertanyaan yg ada di file txt seperti tanda tanya, dll
        item = stopword.remove(re.sub("[!@#$`'’~%^&*()-_=+.,;:|}{?/><]", '', item.lower()))
        # pertanyaan index nya ganjil, jawaban index nya genap. maka ketika index nya ganjil langsung di skip, karena tidak mungkin menyamakan dengan jawaban
        if index % 2 != 0 and index != 0:
            continue
        # trial and error penyamaan string
        try:
            # proses penyamaan menggunakan bc rule/ gs rule
            result = boyer_moore(pattern, p_bm, item)
            # jika ternyata pertanyaan user input dan pertanyaan di txt ada yg sama, maka tampilkan jawaban. jawaban adalah index+1 posisinya
            if len(result) > 0:
                return str(questionAnswer[index+1])
        except AssertionError:
            # jika tidak ada yg sama, maka coba gunakan sinonim dari wordnet. untuk kemudian disamakan kembali dengan pertanyaan yg ada di txt
            for i, word in enumerate(pattern.split()):
                # membuat patokan kata original yg user input
                wordAsal = word
                # trial and error
                try:
                    # cari sinonim ke wordnet, kata yg user input
                    arraySynonym = wn.synsets(word, lang='ind')[0].lemma_names('ind')
                    # muncul banyak sinonim nya, lalu coba replace kata original nya dengan kata per sinonim. lalu lakukan penyamaan dengan pertanyaan yg ada di txt
                    for j, synonym in enumerate(arraySynonym):
                        # me replace kata ori dengan sinonim
                        pattern = pattern.replace(word, synonym)
                        # kata sinonim nya
                        word = synonym
                        # trial and error
                        try:
                            # membuat objek boyermoore
                            result = boyer_moore(pattern, p_bm, item)
                            # jika ternyata pertanyaan user input dan pertanyaan di txt ada yg sama, maka tampilkan jawaban. jawaban adalah index+1 posisinya
                            if len(result) > 0:
                                return str(questionAnswer[index+1])
                        except AssertionError:
                            # jika ketika sudah direplace sinonim, tidak ada kesamaan dengan pertanyaan. maka lanjut ke sinonim berikutnya
                            continue
                except IndexError:
                    continue
                # replace kembali kata sinonim ke kata ori nya
                pattern = pattern.replace(word, wordAsal)
    # jika tidak ada kesamaan sama sekali, seperti contoh ngetik kata asal "lskajflaskdjflaskdjf"
    return str('Aku Tidak Mengerti')

log = logging.getLogger('werkzeug')
log.disabled = True

# proses start aplikasi
if __name__ == "__main__":
    app.run()
