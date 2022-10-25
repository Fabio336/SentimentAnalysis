import re
import nltk
import spacy
import unicodedata
import unidecode

from bs4 import BeautifulSoup
from contractions import CONTRACTION_MAP
from nltk.tokenize.toktok import ToktokTokenizer
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import sent_tokenize, word_tokenize

tokenizer = ToktokTokenizer()
porter = PorterStemmer()
lemmatizer = WordNetLemmatizer()
stopword_list = nltk.corpus.stopwords.words('english')
nlp = spacy.load('en_core_web_sm')

def remove_html_tags(text):
    """
    Remove html tags from text
    
    Parameters
    ----------
    text : str
        Text to normalize
    """
    # Put your code
    cleaner = re.compile('<.*?>')
    text = re.sub(cleaner, '', text)
    return text


def stem_text(text):
    """
    Return a crude heuristic process that chops off the ends of words
    in the hope of reduce inflectional forms of wordsand sometimes
    derivationally related forms of a word to a common base form.
    
    Parameters
    ----------
    text : str
        Text to normalize
    """
    # Put your code
    # Word-list for stemming
    token_words=tokenizer.tokenize(text)
    stem_sentence=[]
    for i in token_words:
        word=i.lower()
        stem_word=porter.stem(word)
        stem_sentence.append(stem_word)
    text = " ".join(stem_sentence)
    return text


def lemmatize_text(text):
    """
    Considers the context and converts the word to its meaningful base form.

    Parameters
    ----------
    text : str
        Text to normalize
    """
    # Put your code
    doc = nlp(text)
    lemma_sentence = []
    for token in doc:
        lemma_sentence.append(token.lemma_)
    
    text = ' '.join(map(str,lemma_sentence))
    return text


def expand_contractions(text, contraction_mapping=CONTRACTION_MAP):
    """
    Expand contractions

    Parameters
    ----------
    text : str
        Text to normalize
    """
    for contractions,base in contraction_mapping.items():
        text=text.replace(contractions, base)
    return text


def remove_accented_chars(text):
    """
    This function deletes the accents from setences

    Parameters
    ----------
    text : str
        Text to normalize
    """
    # Put your code
    text = ''.join(c for c in unicodedata.normalize('NFD', text)
                  if unicodedata.category(c) != 'Mn')
    return text


def remove_special_chars(text, remove_digits=False):
    text = re.sub(r"[^A-Za-z ]+",'', text)
    if remove_digits:
      text = re.sub(r'[^a-zA-Z0-9 ]+', '', text)
    return text


def remove_stopwords(text,is_lower_case=True, stopwords=stopword_list):
    """
    This function removes stopwords from sentence

    Parameters
    ----------
    text : str
        Text to normalize
    """
    # Put your code
    if is_lower_case:
        text=text.lower()

    token_words=tokenizer.tokenize(text)
    cleaned_sentence=[]
    for token in token_words:
        if token not in stopwords:
            cleaned_sentence.append(token)
            
    text = " ".join(cleaned_sentence)
    return text


def remove_extra_new_lines(text):
    # Put your code
    """
    This function deletes extra new lines

    Parameters
    ----------
    text : str
        Text to normalize
    """
    text=re.sub('\n+', ' ', text)
    return text
    
def remove_extra_whitespace(text):
    """
    This function removes extra whitespaces

    Parameters
    ----------
    text : str
        Text to normalize
    """
    # Put your code
    token_words=tokenizer.tokenize(text)
    cleaned_sentence=[]
    for i in token_words:
        cleaned_sentence.append(i)
    text = " ".join(cleaned_sentence)
    return text

def normalize_corpus(
    corpus,
    html_stripping=True,
    contraction_expansion=True,
    accented_char_removal=True,
    text_lower_case=True,
    text_stemming=False,
    text_lemmatization=False,
    special_char_removal=True,
    remove_digits=True,
    stopword_removal=True,
    stopwords=stopword_list
):
    
    normalized_corpus = []

    # Normalize each doc in the corpus
    for doc in corpus:
        # Remove HTML
        if html_stripping:
            doc = remove_html_tags(doc)
            
        # Remove extra newlines
        doc = remove_extra_new_lines(doc)
        
        # Remove accented chars
        if accented_char_removal:
            doc = remove_accented_chars(doc)
            
        # Expand contractions    
        if contraction_expansion:
            doc = expand_contractions(doc)
            
        # Lemmatize text
        if text_lemmatization:
            doc = lemmatize_text(doc)
            
        # Stemming text
        if text_stemming and not text_lemmatization:
            doc = stem_text(doc)
            
        # Remove special chars and\or digits    
        if special_char_removal:
            doc = remove_special_chars(
                doc,
                remove_digits=remove_digits
            )  

        # Remove extra whitespace
        doc = remove_extra_whitespace(doc)

         # Lowercase the text    
        if text_lower_case:
            doc = doc.lower()

        # Remove stopwords
        if stopword_removal:
            doc = remove_stopwords(
                doc,
                is_lower_case=text_lower_case,
                stopwords=stopwords
            )

        # Remove extra whitespace
        doc = remove_extra_whitespace(doc)
        doc = doc.strip()
            
        normalized_corpus.append(doc)
        
    return normalized_corpus

def normalize_review(review: str) -> str:
    return normalize_corpus([review], stopword_list)[0]