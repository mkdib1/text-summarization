import pandas as pd
import os
import nltk
from nltk.corpus import stopwords
import spacy
from sentence_transformers import SentenceTransformer
model = SentenceTransformer('all-MiniLM-L6-v2')


from util import preprocess_article, lda_summarizer, pager_summarizer, blanc_scoring, blanc_score_unit
from blanc import BlancHelp

from dotenv import load_dotenv
load_dotenv()


bbc_path = os.getenv("BBC_DIR")

if __name__ == "__main__":
    
    nltk.download('stopwords')
    nltk.download('punkt_tab')
    
    stop_words = set(stopwords.words('english')) # stop_words list
    added_stopwords = {
        "however", "yet", "although", "though", "even though", "nevertheless", "nonetheless",
        "still", "despite", "in spite of", "whereas", "alternatively", "instead", "regardless",
        "notwithstanding", "albeit", "conversely", "be that as it may", "even so", "that said",
        "even if", "except", "rather", "apart from", "despite that", "then again", "in contrast",
        "after all"
    }
    
    all_stopwords = set(stop_words).union(added_stopwords)
    nlp = spacy.load("en_core_web_sm") # spacy.nlp load
    
    # data read
    df_bbc =  pd.read_csv(bbc_path, sep='\t')
    
    print("--- DF RANDOM SAMPLING ---")
    df_sample = df_bbc.sample(frac=0.05, random_state=125) # extract randomly 1% of rows, in order to make further tuning in reasonable time
    print(f"--- extracted {len(df_sample)} out of {len(df_bbc)} rows ---")

    # lists init. to save results
    model_name = []
    avg_score = []
    avg_std = []

    
    print("- - - LDA SUMMARIZER - - -")
    
    # hyperparameter lists init. for tuning
    num_sent = [2, 5] # number of sentences
    num_topic = [2, 4] # number of topics
    bonus_weights = [0.0, 0.6] # rewarding weights for sentences which begin with PROPN as subject of the sentence
    penalty_weights = [0.0, 0.6] # penalizing weights for sentences which begin with ADV or stop_words
    
    for num_s in num_sent:
        for num_t in num_topic:
            for bonu_w in bonus_weights:
                for penalty_w in penalty_weights:
                    blanc = None
                    blanc = BlancHelp() # blanc init.
                    
                    # summary extraction
                    summaries = [
                        lda_summarizer(
                            nlp=nlp,
                            text=text,
                            num_sentences=num_s,
                            num_topics=num_t,
                            remove_stopwords=True,
                            stop_words=all_stopwords,
                            bonus_weight=bonu_w,
                            penalty_weight=penalty_w
                        )
                        for text in df_sample['content']
                    ]

                    # blanc scores computing
                    score, std = blanc_score_unit(
                        docs=df_sample['content'].tolist(),
                        summaries=summaries,
                        blanc_init=blanc
                    )

                    # results appending
                    model_name.append(f'lda_ns{num_s}_nt{num_t}_bw{bonu_w}_pw{penalty_w}')
                    avg_score.append(score)
                    avg_std.append(std)
    
    print("- - - PAGE RANK SUMMARIZER WITH MMR - - -")
    
    
    # lambda parameter of Maximum Marginal Relevance
    lamb_list = [0.4, 0.8]
    pgs = [None, 10]
    models = [None, model]
    use_mmr = [True, False]
    
    i = 0
    for num_s in num_sent:
        for lamb in lamb_list:
            for bonu_w in bonus_weights:
                for penalty_w in penalty_weights:
                    for pg in pgs:
                        for um in use_mmr:
                            for mod in models:
                                i += 1
                                print(f" - - - RUN {i} di 128 - - -")
                                blanc = None
                                blanc = BlancHelp() # blanc init.
                                
                                # summary extraciton
                                summaries = [
                                    pager_summarizer(
                                        nlp=nlp,
                                        text=text,
                                        num_sentences=num_s,
                                        remove_stopwords=True,
                                        stop_words=all_stopwords,
                                        use_mmr=True,
                                        lambda_param=lamb,
                                        bonus_weight=bonu_w,
                                        penalty_weight=penalty_w,
                                        embedding_model=mod
                                    )
                                    for text in df_sample['content']
                                ]

                                # blanc scores computing
                                score, std = blanc_score_unit(
                                    docs=df_sample['content'].tolist(),
                                    summaries=summaries,
                                    blanc_init=blanc
                                )

                                # results appending
                                model_name.append(f'pr_ns{num_s}_l{lamb}_bw{bonu_w}_pw{penalty_w}_pg{pg}_um{um}_mod{mod}')
                                avg_score.append(score)
                                avg_std.append(std)
                    
    blanc_score_df = pd.DataFrame({
    'model': model_name,
    'avg_blanc_score': avg_score,
    'avg_std': avg_std
                    })

    print(blanc_score_df)
    blanc_score_df.to_csv('/Data/results/blanc_scores.csv', sep=";", index=False)
    


    
    





