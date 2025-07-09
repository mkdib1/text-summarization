import re
from typing import List
from nltk.corpus import stopwords
from spacy.language import Language

from gensim import corpora, models # 'gensim' is specialised in topic modeling
from blanc import BlancHelp

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import normalize

import numpy as np
import networkx as nx

from matplotlib import pyplot as plt
from textwrap import wrap
from pprint import pprint


def preprocess_article(nlp: Language, text: str, 
                       remove_stopwords: bool = False, stop_words=None) -> List[str]:
    """
    Preprocesses a news article for further NLP tasks such as topic modeling or summarization.

    Cleaning and preprocessing steps:
    - Removes multiple spaces and strips into a single space.
    - Normalizes quotation marks (e.g., “ ” ’).
    - Removes all non-alphanumeric but a set of punctuation (.,;:- and symbols as %, £, $).
    - Uses spaCy to segment the text into sentences and tokenize/lemmatize each sentence.
    - Removes stopwords from lemmatized tokens.

    Parameters:
    - nlp: A spaCy `Language` object used for tokenization and sentence segmentation.
    - text: The raw input text to be cleaned and processed.
    - remove_stopwords: Whether to exclude stopwords from the lemmatized output.
    - stop_words: A collection (e.g., set or list) of stopwords to filter out, used if `remove_stopwords=True`.

    Returns:
    - processed_sentences: A list of preprocessed, lemmatized sentences (as strings).
    - original_sentences: A list of the original sentences extracted from the text.
    - doc: The spaCy `Doc` object for original index sentences retrieval.
    """
    text = re.sub(r'\s+', ' ', text)  # collapse multiple spaces
    text = re.sub(r'[“”’]', " ", text)  # normalize quotes
    text = re.sub(r'[^a-zA-Z0-9%£$.,;:\-\s]', '', text)  # keep alphanum + some punctuation

    doc = None
    doc = nlp(text)

    original_sentences = [sent.text.strip() for sent in doc.sents]
    processed_sentences = []

    for sent in doc.sents:
        words = []

        for token in sent:
            if token.is_punct and token.text != ".":  # keep "."
                continue
            lemma = token.lemma_.lower()
            if remove_stopwords and lemma in stop_words:
                continue

            if lemma.strip() == "":
                continue
            words.append(lemma)

        if words:
            processed_sentences.append(" ".join(words))

    return processed_sentences, original_sentences, doc



def extract_pos_features(doc, stop_words):
    """
    Analyzes sentences through extracting part-of-speech (POS) features from a spaCy Doc object to aid in scoring for summarization.

    Features extracted:
    - Whether each sentence contains a proper noun that functions as the subject (or passive subject).
    - Whether each sentence starts with an adverb, a discourse marker (e.g., "however", "but", etc.), 
      or a stopword.

    Parameters:
    - doc: A spaCy `Doc` object representing the parsed text.
    - stop_words: A set or list of stopwords used to check if a sentence starts with one.

    Returns:
    - has_propn_subj: A list of booleans indicating if each sentence contains a proper noun subject.
    - starts_with_adv: A list of booleans indicating if each sentence begins with an adverb, a discourse connective, or a stopword.
    """
    has_propn_subj = []
    starts_with_adv = []
    
    for sent in doc.sents:
        has_propn = any(token.dep_ in ("nsubj", "nsubjpass") and token.pos_ == "PROPN" for token in sent) # check if the sentence contains a proper noun which is also the subject (or passive subject)
        has_propn_subj.append(has_propn)

        first_token = sent[0]
        starts_adv = (first_token.pos_ == "ADV" 
                      or first_token.text.lower() in {"however", "but", "yet", "although"} 
                      or first_token.text.lower() in stop_words
                     )
        starts_with_adv.append(starts_adv)
    
    return has_propn_subj, starts_with_adv



def mmr(doc_embeddings, query_embedding, lambda_param=None, top_n=5, has_propn_subj=None, starts_with_adv=None,
        bonus_weight=None, penalty_weight=None):
    """
    Applies the Maximal Marginal Relevance (MMR) algorithm to select an informative and not-redundant subset of sentences.
    
    Parameters:
    - doc_embeddings: np.ndarray of shape (n_sentences, embedding_dim), the embeddings of the candidate sentences.
    - query_embedding: np.ndarray of shape (embedding_dim,), the embedding representing the query/topic to which doc_embeddings is compared with.
    - lambda_param: float (0 ≤ λ ≤ 1), balancing relevance (λ) and diversity (1−λ). Required.
    - top_n: int, number of sentences to select.
    - has_propn_subj: Optional list of bools, indicating whether each sentence has a proper noun subject.
    - starts_with_adv: Optional list of bools, indicating whether each sentence starts with an adverb or stopword.
    - bonus_weight: float, score increase if the sentence contains a proper noun subject.
    - penalty_weight: float, score reduction if the sentence starts with an adverb, a discourse connective, or a stopword.

    Returns:
    - selected: A list of indices of the selected sentences in order of selection.
    """
    selected = []
    candidates = list(range(doc_embeddings.shape[0]))
    sim_to_query = cosine_similarity(doc_embeddings, query_embedding.reshape(1, -1)).flatten()
    for _ in range(top_n):
        mmr_scores = []
        for idx in candidates:
            sim_to_selected = 0
            if selected:

                selected_embeddings = doc_embeddings[selected]
                sim_values = cosine_similarity(doc_embeddings[idx].reshape(1, -1), selected_embeddings)[0]
                sim_to_selected = max(sim_values)
            weight = 1.0
            
            # Rewards if the sentence contains a proper noun subject
            if has_propn_subj is not None and has_propn_subj[idx]:
                weight += bonus_weight
            
            # Penalizes if the sentence starts with an adverb or stopword
            if starts_with_adv is not None and starts_with_adv[idx]:
                weight -= penalty_weight
            
            # Compute weighted MMR score
            mmr_score = weight * (lambda_param * sim_to_query[idx] - (1 - lambda_param) * sim_to_selected)
            mmr_scores.append((idx, mmr_score))

        # Select the candidate with the highest MMR score
        selected_idx = max(mmr_scores, key=lambda x: x[1])[0]
        selected.append(selected_idx)
        candidates.remove(selected_idx)

    return selected



def pager_summarizer(nlp: Language = None, text: str = None, num_sentences: int = None,
                     remove_stopwords=False, stop_words=None, use_mmr: bool = False,
                     lambda_param=None, bonus_weight: float = None, penalty_weight: float = None,
                     plot_graph: bool = False, pagerank_top_k: int = 20,
                     embedding_model=None):
    """
    Generates an extractive summary of a news article using PageRank.

    Parameters:
    - nlp: A spaCy language model used for tokenization.
    - row: A row from a DataFrame containing the article text (expects a `content` or `news` field).
    - num_s: Desired number of sentences in the summary.
    - pg: Custom integer value chosen as the top-k rank sentences to be extracted by PageRank.
    - use_mmr: If True, Maximum Marginal Relevance is performed.
    - lmbd_p: Lambda parameter used by MMR.
    - stop_words: List or set of stopwords to remove.
    - bonus_weight: Weight factor to reward original sentences containing prop nouns as subjects.
    - penalty_weight: Weight factor to penalize original sentences starting with an adverb, a discourse marker or a stopword.

    Returns:
    - A string containing the summary, or an empty string in case of error.
    """
    
    # Preprocessing
    processed_sentences, original_sentences, doc = preprocess_article(
        nlp=nlp, text=text, remove_stopwords=remove_stopwords, stop_words=stop_words
    )

    num_sentences = max(1, min(num_sentences, len(original_sentences))) # max number of sentences set
    has_propn_subj, starts_with_adv = extract_pos_features(doc, stop_words) # POS tag

    if embedding_model is not None: # sentence transf for embedding representation
        doc_embeddings = embedding_model.encode(processed_sentences, convert_to_numpy=True)
    else: # tf-idf for text representation
        vectorizer = TfidfVectorizer(strip_accents='unicode', max_df=0.8) 
        tfidf_matrix = vectorizer.fit_transform(processed_sentences)
        doc_embeddings = tfidf_matrix.toarray()


    similarity_matrix = cosine_similarity(doc_embeddings) # similarity matrix using cosine similarity
    np.fill_diagonal(similarity_matrix, 0)
    nx_graph = nx.from_numpy_array(similarity_matrix) # graph building
    pagerank_scores = nx.pagerank(nx_graph, max_iter=500) # pagerank computing

    if plot_graph: # graph plot
        node_colors = [pagerank_scores[i] for i in range(len(pagerank_scores))]
        pos = nx.spring_layout(nx_graph, seed=42)
        nx.draw(nx_graph, pos=pos, node_color=node_colors, cmap=plt.cm.Blues, with_labels=True)
        plt.show()

    if not use_mmr: # rank with 'pagerank_top_k' parameter
        ranked = sorted(((pagerank_scores[i], i, s) for i, s in enumerate(original_sentences)), reverse=True)
        selected = sorted(ranked[:num_sentences], key=lambda x: x[1])
        summary = ' '.join([s for _, _, s in selected])
        return summary

    # use_mmr == True
    total_sentences = len(original_sentences) # rescaling sentences and position after preprocessing 
    if pagerank_top_k is None or pagerank_top_k >= total_sentences: # takes all the sentences
        sub_embeddings = doc_embeddings
        top_k_indices = list(range(total_sentences))
    else:
        top_k = min(pagerank_top_k, total_sentences) # takes only top-k sentences
        ranked_indices = [i for i in pagerank_scores.keys()]
        ranked_indices = sorted(ranked_indices, key=lambda i: pagerank_scores[i], reverse=True)

        top_k_indices = ranked_indices[:top_k]
        sub_embeddings = doc_embeddings[top_k_indices]

    query = np.mean(sub_embeddings, axis=0) # query as the average of the sentence embeddings
    scaler = MinMaxScaler() 
    query = scaler.fit_transform(query.reshape(-1, 1)).flatten() # scaled and flatten

    selected_local = mmr(
        sub_embeddings,
        query,
        top_n=num_sentences,
        lambda_param=lambda_param,
        bonus_weight=bonus_weight,
        penalty_weight=penalty_weight,
        has_propn_subj=[has_propn_subj[i] for i in top_k_indices],
        starts_with_adv=[starts_with_adv[i] for i in top_k_indices]
    )

    selected_global = [top_k_indices[i] for i in selected_local]
    selected_global.sort()

    summary = ' '.join([original_sentences[i] for i in selected_global])
    return summary


 
def lda_summarizer(nlp, text, num_sentences=None, num_topics=3, remove_stopwords=False, stop_words=None,
                   bonus_weight=0.6, penalty_weight=0.3):
    """
    Generates an extractive summary of a news article using LDA-based topic modeling.

    Parameters:
    - nlp: A spaCy language model used for tokenization.
    - row: A row from a DataFrame containing the article text (expects a `content` or `news` field).
    - num_s: Desired number of sentences in the summary.
    - nt: Number of topics to consider in the LDA model.
    - stop_words: List or set of stopwords to remove.
    - bonus_weight: Weight factor to reward sentences containing prop nouns as subjects.
    - penalty_weight: Weight factor to penalize sentences starting with an adverb, a discourse marker or a stopword.

    Returns:
    - A string containing the summary, or an empty string in case of error.
    """
    if num_sentences is None:
        raise ValueError("The number of top-n sentences to be extracted must be defined.")

    # preprocessing
    processed_sentences, original_sentences, doc = preprocess_article(nlp, text,
                                                  remove_stopwords=remove_stopwords,
                                                  stop_words=stop_words)

    num_sentences = max(1, min(num_sentences, int(len(original_sentences) * 0.6))) # max number of sentences set
    has_propn_subj, starts_with_adv = extract_pos_features(doc, stop_words) # POS tag
    tokenized_sentences = [sentence.split() for sentence in processed_sentences] # BoW tokenization

    # corpus for LDA
    dictionary = corpora.Dictionary(tokenized_sentences)
    corpus = [dictionary.doc2bow(sent) for sent in tokenized_sentences]

    # LDA perform
    lda_model = models.LdaModel(corpus=corpus,
                                id2word=dictionary,
                                num_topics=num_topics,
                                random_state=42,
                                alpha='auto',
                                eta='auto',
                                passes=10,
                                iterations=100,
                                per_word_topics=True,
                                minimum_probability=0.02,
                                update_every=0)

    # score computing
    sentence_scores = []
    for i, bow in enumerate(corpus):
        topic_dist = lda_model.get_document_topics(bow)
        base_score = sum(prob for _, prob in topic_dist)
        weight = 1.0
        if has_propn_subj[i]:
            weight += bonus_weight
        if starts_with_adv[i]:
            weight -= penalty_weight

        final_score = weight * base_score
        sentence_scores.append((final_score, i, original_sentences[i]))

    ranked = sorted(sentence_scores, reverse=True)
    selected = sorted(ranked[:num_sentences], key=lambda x: x[1])
    summary = ' '.join([s[2] for s in selected])
    return summary



def print_summary(text):
    """
    Text summary by wrapping lines to a fixed width.

    Parameters:
    - text: A string representing the text to be printed.

    Behavior:
    - Wraps the text to lines of 100 characters.
    - Prints the formatted text line by line.
    """
    print('\n'.join(wrap(text, 100)))
    


def blanc_scoring(docs: list, summaries: list, blanc_init: BlancHelp):
    """
    Computes BLANC scores for a list of document-summary pairs using the BLANC evaluator.

    Parameters:
    - docs: List of original documents.
    - summaries: List of generated summaries corresponding to the documents.
    - blanc_init: An initialized `BlancHelp` object for BLANC evaluation.

    Returns:
    - scores: A list of BLANC scores for each document-summary pair.
    """
    scores = blanc_init.eval_pairs(docs=docs, summaries=summaries)
    return scores


def blanc_score_unit(docs: list, summaries: list, blanc_init: BlancHelp):
    """
    Computes and prints the average and standard deviation of BLANC scores for a set of summaries by calling
    `blanc_scoring` function.

    Parameters:
    - docs: List of original documents.
    - summaries: List of corresponding summaries to evaluate.
    - blanc_init: An initialized `BlancHelp` object for BLANC scoring.

    Returns:
    - mean_score: The average BLANC score across all evaluations.
    - std_score: The standard deviation of the valid BLANC scores.
    """
    scores = blanc_scoring(docs=docs, summaries=summaries, blanc_init=blanc_init)
    valid_scores = [s for s in scores if s is not None]
    mean_score = np.mean(valid_scores)
    std_score = np.std(valid_scores)

    print(f"🔎 BLANC average: {mean_score:.3f}")
    print(f"📉 BLANC std: {std_score:.3f}")
    return mean_score, std_score

