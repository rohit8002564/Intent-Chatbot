import os
import csv
import datetime
import streamlit as st
import json
import random
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# Load intents JSON file
with open('intents.json') as file:
    data = json.load(file)

# Preprocess data
tags = []
patterns = []
for intent in data['intents']:
    for pattern in intent['patterns']:
        patterns.append(pattern)
        tags.append(intent['tag'])

# Fit the vectorizer
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(patterns)

# Train the model
model = LogisticRegression()
model.fit(X, tags)

# Define chat function
def chat(user_input):
    input_vector = vectorizer.transform([user_input])
    prediction = model.predict(input_vector)[0]
    for intent in data['intents']:
        if intent['tag'] == prediction:
            return random.choice(intent['responses'])

counter = 0

def main():
    global counter
    st.title("Intents of Chatbot using NLP")
    
    # Create a sidebar menu with options
    menu = ["Home", "Conversation History", "About"]
    choice = st.sidebar.selectbox("Menu", menu)
    
    # Home Menu
    if choice == "Home":
        st.write("Welcome")
        if not os.path.exists('chat_log.csv'):
            with open('chat_log.csv', 'w', newline='', encoding='utf-8') as csvfile:
                csv_writer = csv.writer(csvfile)
                csv_writer.writerow(['User Input', 'Chatbot Response', 'Timestamp'])
                
        counter += 1
        user_input = st.text_input("You:", key=f"user_input_{counter}")
        
        if user_input:
            # Convert the user input to a string
            user_input_str = str(user_input)
            
            response = chat(user_input)
            st.text_area("Chatbot:", value=response, height=120, max_chars=None, key=f"chatbot_{counter}")
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
            with open('chat_log.csv', 'a', newline='', encoding='utf-8') as csvfile:
                csv_writer = csv.writer(csvfile)
                csv_writer.writerow([user_input_str, response, timestamp])
                
            if response.lower() in ['goodbye', 'bye']:
                st.write("Thank you for chatting with me. Have a great day!")
                st.stop()
                
    elif choice == "Conversation History":
        st.header("Conversation History")
        with open('chat_log.csv', 'r', encoding='utf-8') as csvfile:
            csv_reader = csv.reader(csvfile)
            next(csv_reader)
            for row in csv_reader:
                st.text(f"User: {row[0]}")
                st.text(f"Chatbot: {row[1]}")
                st.text(f"Timestamp: {row[2]}")
                st.markdown("------")
                
    elif choice == "About":
        st.write("The goal of this project is to create a chatbot that can understand and respond to user inputs using NLP techniques.")
        st.subheader("Project Overview:")
        st.write("""
        The project is divided into two parts:
        
        1. NLP techniques and Logistic Regression algorithm are used to train the chatbot on labeled data.
        2. Streamlit is used for building the chatbot interface.
        """)
        
        st.subheader("Dataset:")
        st.write("The dataset consists of various intents, patterns, and responses used to train the chatbot.")

if __name__ == '__main__':
    main()
