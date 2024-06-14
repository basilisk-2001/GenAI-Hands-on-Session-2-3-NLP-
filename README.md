# GenAI-Hands-on-Session-2-3-NLP-
This repository contains very intermediate hands on tasks that can be performed for easy understanding of NLP related topics

### 1. **Text Preprocessing Workflow**
   - **Activity:** Provide a raw text dataset and walk through the preprocessing steps.
   - **Tasks:** Tokenization, stop word removal, stemming, lemmatization, and vectorization (e.g., TF-IDF).
   - **Tools:** Use Python libraries such as NLTK, SpaCy, or Scikit-learn.
   - **Example Code:**
     ```python
     import nltk
     from nltk.corpus import stopwords
     from nltk.tokenize import word_tokenize
     from sklearn.feature_extraction.text import TfidfVectorizer

     nltk.download('punkt')
     nltk.download('stopwords')

     text = "Natural Language Processing with Python is very interesting."
     tokens = word_tokenize(text)
     tokens = [word for word in tokens if word.isalnum()]
     tokens = [word for word in tokens if word not in stopwords.words('english')]

     vectorizer = TfidfVectorizer()
     tfidf_matrix = vectorizer.fit_transform([' '.join(tokens)])
     print(tfidf_matrix.toarray())
     ```

### 2. **Sentiment Analysis**
   - **Activity:** Build a simple sentiment analysis model.
   - **Tasks:** Use a pre-trained sentiment analysis model or train a basic model using labeled data.
   - **Tools:** NLTK, TextBlob, or Hugging Face Transformers.
   - **Example Code:**
     ```python
     from textblob import TextBlob

     text = "Natural Language Processing is amazing!"
     blob = TextBlob(text)
     sentiment = blob.sentiment
     print(sentiment)
     ```

### 3. **Named Entity Recognition (NER)**
   - **Activity:** Implement a Named Entity Recognition task.
   - **Tasks:** Identify and classify named entities in a given text.
   - **Tools:** SpaCy.
   - **Example Code:**
     ```python
     import spacy

     nlp = spacy.load("en_core_web_sm")
     text = "Google was founded by Larry Page and Sergey Brin."
     doc = nlp(text)

     for ent in doc.ents:
         print(ent.text, ent.label_)
     ```

### 4. **Language Generation with GPT-4o**
   - **Activity:** Generate text using GPT-4o.
   - **Tasks:** Use the GPT-4o model to complete prompts, answer questions, or create new text.
   - **Tools:** OpenAI API.
   - **Example Code:**
     ```python
     import openai

     openai.api_key = 'your-api-key'

     response = openai.Completion.create(
         engine="gpt-4",
         prompt="Once upon a time",
         max_tokens=50
     )

     print(response.choices[0].text.strip())
     ```

### 5. **Comparison of GPT-3 and GPT-4o**
   - **Activity:** Compare the performance of GPT-3 and GPT-4o.
   - **Tasks:** Generate responses to the same prompt using both models and compare the outputs.
   - **Tools:** OpenAI API.
   - **Example Code:**
     ```python
     def generate_text(prompt, model):
         response = openai.Completion.create(
             engine=model,
             prompt=prompt,
             max_tokens=50
         )
         return response.choices[0].text.strip()

     prompt = "Explain the concept of Natural Language Processing."
     gpt3_text = generate_text(prompt, "gpt-3")
     gpt4_text = generate_text(prompt, "gpt-4")

     print("GPT-3 Output:", gpt3_text)
     print("GPT-4 Output:", gpt4_text)
     ```

### 6. **Topic Modeling**
   - **Activity:** Perform topic modeling on a corpus.
   - **Tasks:** Use Latent Dirichlet Allocation (LDA) to identify topics within a set of documents.
   - **Tools:** Gensim.
   - **Example Code:**
     ```python
     import gensim
     from gensim import corpora

     documents = [
         "Natural Language Processing is a field of Artificial Intelligence.",
         "Machine Learning is a subset of Artificial Intelligence.",
         "Deep Learning and Neural Networks are popular in AI."
     ]

     tokens = [doc.lower().split() for doc in documents]
     dictionary = corpora.Dictionary(tokens)
     corpus = [dictionary.doc2bow(token) for token in tokens]
     lda_model = gensim.models.ldamodel.LdaModel(corpus, num_topics=2, id2word=dictionary, passes=15)

     topics = lda_model.print_topics(num_words=4)
     for topic in topics:
         print(topic)
     ```

### 7. **Building a Simple Chatbot (Advanced-can be done if time permits)**
   - **Activity:** Create a basic rule-based chatbot.
   - **Tasks:** Implement a simple conversational agent using predefined rules.
   - **Tools:** NLTK or any rule-based framework.
   - **Example Code:**
     ```python
     from nltk.chat.util import Chat, reflections

     pairs = [
         (r'hi|hello', ['Hello!', 'Hi there!']),
         (r'what is your name?', ['I am a chatbot.', 'You can call me Chatbot.']),
         (r'how are you?', ['I am fine, thank you!', 'Doing well, how about you?']),
     ]

     chatbot = Chat(pairs, reflections)
     chatbot.converse()
     ```
