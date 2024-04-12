import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import joblib
import nltk ,string ,re
from scipy.sparse import hstack

st.set_option('deprecation.showPyplotGlobalUse', False)
sw_stopwords=["akasema","alikuwa","alisema","baada","basi","bila","cha","chini","hadi",
              "hapo","hata","hivyo","hiyo","huku","huo","ili","ilikuwa","juu","kama","karibu",
              "katika","kila","kima","kisha","kubwa","kutoka","kuwa","kwa","kwamba","kwenda","kwenye","la","lakini",
              "mara","mdogo","mimi","mkubwa","mmoja","moja","muda","mwenye","na","naye","ndani","ng","ni","nini",
              "nonkungu","pamoja","pia","sana","sasa","sauti","tafadhali","tena","tu","vile","wa",
              "wakati","wake","walikuwa","wao","watu","wengine","wote","ya","yake","yangu","yao","yeye","yule","za",
              "zaidi","zake","na","ya","wa","kwa","ni","za","katika","la","kuwa","kama","kwamba","cha","hiyo","lakini","yake","hata","wakati",
              "hivyo","sasa","wake","au","watu","hii","zaidi","vya","huo","tu","kwenye","si","pia","ili","moja","kila","baada","ambao","ambayo","yao","wao","kuna",
              "hilo","kutoka","kubwa","pamoja","bila","huu","hayo","sana","ndani","mkuu","hizo","kufanya","wengi","hadi","mmoja","hili","juu","kwanza","wetu","kuhusu",
              "baadhi","wote","yetu","hivi","kweli","mara","wengine","nini","ndiyo","zao","kati","hao","hapa","kutokana","muda","habari","ambaye","wenye","nyingine","hakuna",
              "tena","hatua","bado","nafasi","basi","kabisa","hicho","nje","huyo","vile","yote","mkubwa","alikuwa","zote","leo","haya","huko","kutoa","mwa","kiasi","hasa","nyingi","kabla","wale","chini","gani","hapo","lazima","mwingine","bali","huku","zake","ilikuwa",
              "tofauti","kupata","mbalimbali","pale","kusema","badala","wazi","yeye","alisema","hawa",
              "ndio","hizi","tayari","wala","muhimu","ile","mpya","ambazo","dhidi","kwenda","sisi","kwani",
              "jinsi","binafsi","kutumia","mbili","mbali","kuu","mengine","mbele","namna","mengi","upande","na","lakini","ingawa"
              "ingawaje","kwa","sababu","hadi","hata","kama","ambapo","ambamo","ambako","ambacho","ambao","ambaye","ilhali","ya","yake","yao","yangu","yetu","yenu","vya","vyao","vyake","vyangu",
                "vyenu","vyetu","yako","yao","hizo","yenu","mimi","sisi","wewe","nyinyi","yeye","wao","nao","nasi","nanyi","ni","alikuwa","atakuwa","hii","hizi","zile",
                "ile","hivi","vile","za","zake","zao","zenu","kwenye","katika","kwa","kwao","kwenu","kwetu","dhidi","kati","miongoni","katikati","wakati","kabla","baada",
                "baadaye","nje","tena","mbali","halafu","hapa","pale","mara","mara","yoyote","wowote","chochote","vyovyote","yeyote","lolote","mwenye","mwenyewe","lenyewe",
                "lenye","wote","lote","vyote","nyote","kila","zaidi","hapana","ndiyo","au","ama","ama","sio","siye","tu","budi","nyingi","nyingine","wengine","mwingine",
                "zingine","lingine","kingine","chote","sasa","basi","bila","cha","chini","hapo","pale","huku","kule","humu","hivyo","hivyohivyo","vivyo","palepale","fauka",
                "hiyo","hiyohiyo","zile","zilezile","hao","haohao","huku","hukuhuku","humuhumu","huko","hukohuko","huo","huohuo","hili","hilihili","ilikuwa","juu","karibu",
                "kila","kima","kisha","kutoka","kwenda","kubwa","ndogo","kwamba","kuwa","la","lao","lo","mara","na",
                "mdogo","mkubwa","ng’o","pia","aidha","vile","vilevile","kadhalika","halikadhalika","ni","sana","pamoja","pamoja","tafadhali","tena",
                "wa","wake","wao",
                "ya","yule","wale","zangu","nje","afanaleki","salale","oyee","yupi","ipi","lipi","ngapi","yetu","si","angali","wangali","loo","la","ohoo",
                "barabara","oyee",
                "ewaa","walahi","masalale","duu","toba","mh","kumbe","ala","ebo","haraka","pole","polepole","harakaharaka","hiyo","hivyo","vyovyote",
                "atakuwa","itakuwa","mtakuwa",
                "tutakuwa","labda","yumkini","haiyumkini","yapata","takribani","hususani","yawezekana","nani","juu""chini",
                "ndani","baadhi","kuliko","vile","mwa","kwa","hasha","hivyo","moja","kisha",
                "pili","kwanza","ili","je","jinsi","ila","ila","nini","hasa","huu","zako","mimi",
]
tokenizer = nltk.tokenize.RegexpTokenizer(r'\w+')

#load joblib models
stacking_model = joblib.load('stacking_model.pkl')
voting_model = joblib.load('voting_model.pkl')
tfidf = joblib.load('tfidf.pkl')

# Load the data
train = pd.read_csv('Train.csv')
model_performance = pd.read_csv('model_performance.csv')

def clean_text(text):
    '''Make text lowercase, remove text in square brackets,remove links,remove punctuation
    and remove words containing numbers.'''
    text = text.lower()
    text = re.sub(' ', '', text)
    text = re.sub('https?://\S+|www\.\S+', '', text)
    text = re.sub('<.*?>+', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    return text

def combine_text(list_of_text):
    '''Takes a list of text and combines them into one large chunk of text.'''
    combined_text = ' '.join(list_of_text)
    return combined_text

def remove_err(text):
    text = text.replace('â€™', "")
    text = text.replace('â€œ', "")
    text = text.replace('â€', "")
    text = text.replace('â€˜', "")
    text = text.replace('â€”', "")
    text = text.replace('â€“', "")
    text = text.replace('â€¢', "")
    text = text.replace('â€¦', "")
    text = re.sub(r'^â€˜(.*?)â€™$', '', text)
    text = re.sub(r'^â€œ(.*?)â€', '', text)
    return text

def preprocess_user_input(user_input):
    # Clean text
    cleaned_text = clean_text(user_input)
    cleaned_text = remove_err(cleaned_text)
    
    # Tokenization
    tokens = tokenizer.tokenize(cleaned_text)
    
    # Remove stopwords
    words = [w for w in tokens if w not in sw_stopwords]
    
    # Combine text
    combined_text = combine_text(words)
    
    return [combined_text]

# Function to display category distribution chart
def display_category_distribution():
    plt.figure(figsize=(10, 6))
    sns.countplot(train['category'])
    plt.title('Category Distribution')
    plt.xlabel('Category')
    plt.ylabel('Count')
    st.pyplot()

# Function to display model performance metrics and comparison
def display_model_performance():
    st.write(model_performance)
    st.line_chart(model_performance[['Accuracy', 'Log Loss']])

# Function to provide explanation on model performance
def explain_model_performance():
    st.write("Model Comparison and Explanation:")
    st.write("- Logistic Regression and CatBoostClassifier performed relatively well compared to other models, possibly due to their ability to handle text data effectively.")
    st.write("- Decision Tree Classifier had low accuracy and high log loss, indicating overfitting on the training data.")
    st.write("- Ensemble techniques like Voting Classifier and Stacking Classifier improved overall performance by combining predictions from multiple models.")

# Function to make prediction based on user input
def make_prediction(user_input, model_option):
    if model_option == 'StackingClassifier':
        prediction = stacking_model.predict([user_input])[0]
        probability = stacking_model.predict_proba([user_input])[0]
    elif model_option == 'VotingClassifier':
        prediction = voting_model.predict([user_input])[0]
        probability = voting_model.predict_proba([user_input])[0]
        
    return prediction, probability

    # Placeholder code for prediction
    #prediction = "Politics"
    #probability = {'Politics': 0.7, 'Sports': 0.2, 'Entertainment': 0.1}
    #return prediction, probability

# Streamlit application layout
def main():
    st.title("News Categorization Application")

    st.sidebar.title("Explore")

    page = st.sidebar.radio("Navigate", ["Home", "Model Performance", "Further Research"])

    if page == "Home":
        st.write("This application classifies Swahili news articles into different categories such as Politics, Sports, and Entertainment.")
        st.write("Let's get started by exploring the data and model performance!")
        st.write(train.head(5))
        
        st.subheader("Category Distribution")
        display_category_distribution()
        
        st.write("We had to map the categories to numerical values for the model to work. Here is the mapping: ")
        mapped_code = """
        category_mapping = {
                            "Kitaifa": 0,
                            "Biashara": 1,
                            "michezo": 2,
                            "Kimataifa": 3,
                            "Burudani": 4
                            }

        train["category"] = train.category.map(category_mapping)
                """
        st.code(mapped_code, language='python')

        st.write("We generated new features .Here is the code: ")
        feature_code = """
        #calculate word count for each text
        train['word_count'] = train['content'].apply(lambda x: len(x.split()))

        #Calculate average word length
        train['avg_word_length'] = train['content'].apply(lambda x: np.mean([len(word) for word in x.split()]))


        #Calculate word count for each text
        test['word_count'] = test['content'].apply(lambda x: len(x.split()))

        #Calculate average word length
        test['avg_word_length'] = test['content'].apply(lambda x: np.mean([len(word) for word in x.split()]))
                
        """
        st.code(feature_code, language='python')
        
        st.write("The following section outlines code we used to preprocess the text data: ")
        preprocess_code = """
        def clean_text(text):
            '''Make text lowercase, remove text in square brackets,remove links,remove punctuation
            and remove words containing numbers.'''
            text = text.lower()
            text = re.sub(' ', '', text)
            text = re.sub('https?://\S+|www\.\S+', '', text)
            text = re.sub('<.*?>+', '', text)
            text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
            return text


        # Applying the cleaning function to both test and training datasets
        train['content'] = train['content'].apply(lambda x: clean_text(x))
        test['content'] = test['content'].apply(lambda x: clean_text(x))
        
        
        def remove_err(text):
            text = text.replace('â€™', "")
            text = text.replace('â€œ', "")
            text = text.replace('â€', "")
            text = text.replace('â€˜', "")
            text = text.replace('â€”', "")
            text = text.replace('â€“', "")
            text = text.replace('â€¢', "")
            text = text.replace('â€¦', "")
            text = re.sub(r'^â€˜(.*?)â€™$', '', text)
            text = re.sub(r'^â€œ(.*?)â€', '', text)
            return text


        # Applying the cleaning function to both test and training datasets
        train['content'] = train['content'].apply(lambda x: remove_err(x))
        test['content'] = test['content'].apply(lambda x: remove_err(x))

        
        # Tokenizing the training and the test set
        tokenizer = nltk.tokenize.RegexpTokenizer(r'\w+')
        train['content'] = train['content'].apply(lambda x: tokenizer.tokenize(x))
        test['content'] = test['content'].apply(lambda x: tokenizer.tokenize(x))
        
        def remove_stopwords(text):
            words = [w for w in text if w not in sw_stopwords]
            return words

        train['content'] = train['content'].apply(lambda x : remove_stopwords(x))
        test['content'] = test['content'].apply(lambda x : remove_stopwords(x))
        
        """
        st.code(preprocess_code, language='python')
        
        st.write("We further used TF-IDF to vectorize the text data. Here is the code: ")
        tfidf_code = """
        tfidf = TfidfVectorizer(min_df=15, max_df=0.5, ngram_range=(1, 2),norm='l2',sublinear_tf=True)
        train_vectors = tfidf.fit_transform(train['content'])
        test_vectors = tfidf.transform(test["content"])
        
        
        # Concatenate TF-IDF vectors with new features
        X_train_combined = hstack((train_vectors, train[['word_count', 'avg_word_length']].values))
        X_test_combined = hstack((test_vectors, test[['word_count', 'avg_word_length']].values))
        """
        st.code(tfidf_code, language='python')
        
        st.write("With this preprocessing, we were able to train and evaluate different models to classify news articles into different categories. Let's explore the model performance next!")
    elif page == "Model Performance":
        st.write("We evaluated the performance of different models using accuracy and log loss metrics. Here are the results:")
        display_model_performance()
        explain_model_performance()
    elif page == "Further Research":
        st.write("We can further improve the model performance by exploring the following:")
        st.write("- Hyperparameter tuning to optimize model performance.")
        st.write("- Feature engineering to extract more meaningful features from the text data.")
        st.write("- Exploring advanced deep learning models like LSTM and BERT for text classification.")
        st.write("Feel free to explore the data, model performance, and make predictions using the sidebar!")
    else:
        st.write("Invalid page selection. Please select a valid page from the sidebar.")
        
    

if __name__ == "__main__":
    main()
