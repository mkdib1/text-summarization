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


def preprocess_article(nlp: Language, text: str, remove_stopwords: bool = False, stop_words=None) -> List[str]:
    
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
    Estrae due feature boolean per ciascuna frase:
    - has_propn_subj: True se c'è un soggetto PROPN
    - starts_with_adv: True se la frase inizia con ADV o parole tipo "however", ecc.
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
    selected = []
    candidates = list(range(doc_embeddings.shape[0]))
    sim_to_query = cosine_similarity(doc_embeddings, query_embedding.reshape(1, -1)).flatten()
    for _ in range(top_n):
        mmr_scores = []
        for idx in candidates:
            sim_to_selected = 0
            if selected:
                # sim_to_selected = max(cosine_similarity(
                #     doc_embeddings[idx].reshape(1, -1),
                #     doc_embeddings[selected])[0]
                # )
                selected_embeddings = doc_embeddings[selected]
                sim_values = cosine_similarity(doc_embeddings[idx].reshape(1, -1), selected_embeddings)[0]
                # print(sim_values)
                sim_to_selected = max(sim_values)
            weight = 1.0
            
            # Aggiungi bonus se la frase ha nome proprio come soggetto
            if has_propn_subj is not None and has_propn_subj[idx]:
                weight += bonus_weight
            
            # Penalizza se inizia con avverbio o particella
            if starts_with_adv is not None and starts_with_adv[idx]:
                weight -= penalty_weight
            
            # Calcola MMR pesata
            mmr_score = weight * (lambda_param * sim_to_query[idx] - (1 - lambda_param) * sim_to_selected)
            mmr_scores.append((idx, mmr_score))

        # print(mmr_scores)
        selected_idx = max(mmr_scores, key=lambda x: x[1])[0]
        selected.append(selected_idx)
        candidates.remove(selected_idx)

    return selected

'''
def pager_summarizer(nlp:Language=None, text:str=None, num_sentences:int=None, remove_stopwords=False, 
                     stop_words=None, use_mmr:bool=False, lambda_param=None, 
                     bonus_weight:float=None, penalty_weight:float=None,plot_graph:bool=False):
    
    ## Preprocessing: ottieni frasi preprocessate
    processed_sentences, original_sentences, doc = preprocess_article(nlp=nlp, text=text, remove_stopwords=remove_stopwords, stop_words=stop_words)
    # Ottieni frasi originali con Spacy per output naturale
    # original_sentences = [sent.text.strip() for sent in doc.sents]

    # if len(original_sentences) <= num_sentences:
    #     return original_sentences  # se poche frasi, restituisci tutte
    
    num_sentences = max(1, min(num_sentences, int(len(original_sentences) * 0.6)))

    
    # Calcolo features per pesatura (propn subj e inizio con avv)
    has_propn_subj, starts_with_adv = extract_pos_features(doc, stop_words)


    # Tf-Idf sulla lista preprocessata
    vectorizer = TfidfVectorizer(strip_accents='unicode')
    tfidf_matrix = vectorizer.fit_transform(processed_sentences)
    # print("- - -TF-IDF MATRIX - - -")
    # print(tfidf_matrix)
    print("- - -RANGE of VALUES TF-IDF MATRIX - - -")
    print(f"max: {np.max(tfidf_matrix)}, min: {np.min(tfidf_matrix)}")
    print("- - - SHAPE TF-IDF MATRIX - - -")
    print(tfidf_matrix.shape)
    if use_mmr:
        if lambda_param is None:
            raise ValueError("Lambda param must be defined when using MMR.")
        query = np.asarray(np.mean(tfidf_matrix.toarray(), axis=0))
        scaler = MinMaxScaler()
        query = scaler.fit_transform(query.reshape(-1, 1)).flatten()
        # print("- - - QUERY TO PASS TO MMR FUNC - - -")
        # print(query)
        print("- - - SHAPE QUERY - - -")
        print(query.shape)
        print("- - -RANGE of VALUES QUERY - - -")
        print(f"max: {np.max(query)}, min: {np.min(query)}")
        selected_indices = mmr(tfidf_matrix, 
                              query, 
                              top_n=num_sentences, 
                              lambda_param=lambda_param,
                              bonus_weight=bonus_weight,
                              penalty_weight=penalty_weight,
                              has_propn_subj=has_propn_subj,
                              starts_with_adv=starts_with_adv)
        # restituisci frasi originali corrispondenti agli indici selezionati
        selected_indices_sorted = sorted(selected_indices)  # ordina per posizione nel testo
        summary = ' '.join([original_sentences[i] for i in selected_indices_sorted])
    else:
        # Similarity matrix e Pagerank
        similarity_matrix = (tfidf_matrix * tfidf_matrix.T).toarray()
        np.fill_diagonal(similarity_matrix, 0)
        nx_graph = nx.from_numpy_array(similarity_matrix)
        scores = nx.pagerank(nx_graph)

        if plot_graph: # optional
            node_colors = [scores[i] for i in range(len(scores))]
            pos = nx.spring_layout(nx_graph, seed=42)
            nx.draw(nx_graph,
                    pos=pos,
                    node_color=node_colors,
                    cmap=plt.cm.Blues,
                    with_labels=True)
            plt.show()

        ranked = sorted(((scores[i], i, s) for i, s in enumerate(original_sentences)), reverse=True)
        selected = sorted(ranked[:num_sentences], key=lambda x: x[1])  # ordina per indice originale

        summary = ' '.join([s for _, _, s in selected])

    return summary

'''

def pager_summarizer(nlp: Language = None, text: str = None, num_sentences: int = None,
                     remove_stopwords=False, stop_words=None, use_mmr: bool = False,
                     lambda_param=None, bonus_weight: float = None, penalty_weight: float = None,
                     plot_graph: bool = False, pagerank_top_k: int = 20,
                     embedding_model=None):
    """
    Pipeline di summarizzazione ibrida:
    - Se use_mmr==False: usa solo PageRank
    - Se use_mmr==True e pagerank_top_k è un int < numero frasi: seleziona top_k con PageRank, poi MMR su queste
    - Se use_mmr==True e pagerank_top_k è None o >= numero frasi: applica MMR su tutte le frasi senza PageRank
    - embedding_model: modello SentenceTransformer per calcolare sentence embeddings (vettori densi).
      Se None, si usa TF-IDF (vettori sparsi).
    """

    # Preprocessing
    processed_sentences, original_sentences, doc = preprocess_article(
        nlp=nlp, text=text, remove_stopwords=remove_stopwords, stop_words=stop_words
    )

    num_sentences = max(1, min(num_sentences, len(original_sentences)))

    # Features POS per bonus/penalità
    has_propn_subj, starts_with_adv = extract_pos_features(doc, stop_words)

    # Calcolo embeddings (TF-IDF o sentence embeddings)
    if embedding_model is not None:
        # Usare sentence embeddings
        doc_embeddings = embedding_model.encode(processed_sentences, convert_to_numpy=True)
    else:
        # Usare TF-IDF
        vectorizer = TfidfVectorizer(strip_accents='unicode', max_df=0.8)
        tfidf_matrix = vectorizer.fit_transform(processed_sentences)
        doc_embeddings = tfidf_matrix.toarray()

    # Matrice di similarità per PageRank (coseno)
    # Calcolo matrice similarità coseno tra embeddings
    similarity_matrix = cosine_similarity(doc_embeddings)
    np.fill_diagonal(similarity_matrix, 0)
    nx_graph = nx.from_numpy_array(similarity_matrix)
    pagerank_scores = nx.pagerank(nx_graph, max_iter=500)

    if plot_graph:
        node_colors = [pagerank_scores[i] for i in range(len(pagerank_scores))]
        pos = nx.spring_layout(nx_graph, seed=42)
        nx.draw(nx_graph, pos=pos, node_color=node_colors, cmap=plt.cm.Blues, with_labels=True)
        plt.show()

    if not use_mmr:
        # Solo PageRank: prendo top num_sentences
        ranked = sorted(((pagerank_scores[i], i, s) for i, s in enumerate(original_sentences)), reverse=True)
        selected = sorted(ranked[:num_sentences], key=lambda x: x[1])
        summary = ' '.join([s for _, _, s in selected])
        return summary

    # use_mmr == True
    total_sentences = len(original_sentences)
    if pagerank_top_k is None or pagerank_top_k >= total_sentences:
        # MMR su tutte le frasi senza filtro PageRank
        sub_embeddings = doc_embeddings
        top_k_indices = list(range(total_sentences))
    else:
        # MMR solo su top-k frasi PageRank
        top_k = min(pagerank_top_k, total_sentences)
        #  ranked_indices = [i for _, i, _ in sorted(((pagerank_scores[i], i, s) for i, s in enumerate(original_sentences)), reverse=True)]
        ranked_indices = [i for i in pagerank_scores.keys()]
        ranked_indices = sorted(ranked_indices, key=lambda i: pagerank_scores[i], reverse=True)

        top_k_indices = ranked_indices[:top_k]
        sub_embeddings = doc_embeddings[top_k_indices]

    # Query: media vettori delle frasi (top-k o tutte)
    query = np.mean(sub_embeddings, axis=0)
    scaler = MinMaxScaler()
    query = scaler.fit_transform(query.reshape(-1, 1)).flatten()

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

    # Mappo gli indici locali MMR a quelli globali
    selected_global = [top_k_indices[i] for i in selected_local]
    selected_global.sort()

    summary = ' '.join([original_sentences[i] for i in selected_global])
    return summary


 
def lda_summarizer(nlp, text, num_sentences=None, num_topics=3, remove_stopwords=False, stop_words=None,
                   bonus_weight=0.6, penalty_weight=0.3):
    if num_sentences is None:
        raise ValueError("The number of top-n sentences to be extracted must be defined.")

    # Preprocessing: ottieni frasi preprocessate e Doc per POS tagging
    processed_sentences, original_sentences, doc = preprocess_article(nlp, text,
                                                  remove_stopwords=remove_stopwords,
                                                  stop_words=stop_words)

    # Frasi originali per l'output finale
    num_sentences = max(1, min(num_sentences, int(len(original_sentences) * 0.6)))

    # Features POS
    has_propn_subj, starts_with_adv = extract_pos_features(doc, stop_words)

    # Tokenizza frasi preprocessate per BoW
    tokenized_sentences = [sentence.split() for sentence in processed_sentences]

    # Dizionario e corpus per LDA
    dictionary = corpora.Dictionary(tokenized_sentences)
    corpus = [dictionary.doc2bow(sent) for sent in tokenized_sentences]

    # Modello LDA
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

    # Calcolo score
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
        # print("- - - FINAL SCORE - - -")
        # print(final_score)
        sentence_scores.append((final_score, i, original_sentences[i]))

    ranked = sorted(sentence_scores, reverse=True)
    selected = sorted(ranked[:num_sentences], key=lambda x: x[1])
    summary = ' '.join([s[2] for s in selected])
    return summary



def print_summary(text):
    print('\n'.join(wrap(text, 100)))
    


def blanc_scoring(docs: list, summaries: list, blanc_init: BlancHelp):
    scores = blanc_init.eval_pairs(docs=docs, summaries=summaries)
    return scores

def blanc_score_unit(docs: list, summaries: list, blanc_init: BlancHelp):
    scores = blanc_scoring(docs=docs, summaries=summaries, blanc_init=blanc_init)
    valid_scores = [s for s in scores if s is not None]
    mean_score = np.mean(valid_scores)
    std_score = np.std(valid_scores)

    print(f"🔎 BLANC medio: {mean_score:.3f}")
    print(f"📉 Deviazione standard: {std_score:.3f}")
    return mean_score, std_score

