import string
import re
# import nltk
# from nltk.stem.porter import PorterStemmer
# from nltk.corpus import stopwords
# nltk.download("stopwords")
#
# stemmer = PorterStemmer()
# stop = stopwords.words('english')

def preprocess(text):
    return text.translate(str.maketrans("","", string.punctuation)).lower()
def tokenize(text):
    MY_CONSONANTS = r"က-အ"

    VIRAMA = r"\u1039"

    ASAT = r"\u103A"

    ENGLISH = r"A-Za-z0-9"

    OTHER = r"ဣဤဥဦဧဩဪဿ၌၍၏၀-၉၊။!-/:-@\[-`{-~\s"

    # Myanmar syllable boundary pattern
    SYLLABLE_BREAK_PATTERN = re.compile(
        rf"((?<!{VIRAMA})"
        rf"[{MY_CONSONANTS}]"
        rf"(?![{ASAT}{VIRAMA}])"
        rf"|[{ENGLISH}{OTHER}])"
    )
    segmented = SYLLABLE_BREAK_PATTERN.sub(r"|\1", text)

    tokens = segmented.split("|")

    tokens = [token for token in tokens if token and token not in [' ','။','၊',"သည်","မှာ","ကို","၏","နှင့်","က","မှာ","တော့","လည်း"]]

    return tokens
