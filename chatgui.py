import nltk
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
import pickle
import numpy as np

from keras.models import load_model
model = load_model('chatbot_model.h5')
import json
import random
intents = json.loads(open('intents.json').read())
words = pickle.load(open('words.pkl','rb'))
classes = pickle.load(open('classes.pkl','rb'))


def clean_up_sentence(sentence):
    # tokenize the pattern - split words into array
    sentence_words = nltk.word_tokenize(sentence)
    # stem each word - create short form for word
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words

# return bag of words array: 0 or 1 for each word in the bag that exists in the sentence

def bow(sentence, words, show_details=True):
    # tokenize the pattern
    sentence_words = clean_up_sentence(sentence)
    # bag of words - matrix of N words, vocabulary matrix
    bag = [0]*len(words)  
    for s in sentence_words:
        for i,w in enumerate(words):
            if w == s: 
                # assign 1 if current word is in the vocabulary position
                bag[i] = 1
                if show_details:
                    print ("found in bag: %s" % w)
    return(np.array(bag))

def predict_class(sentence, model):
    # filter out predictions below a threshold
    p = bow(sentence, words,show_details=False)
    res = model.predict(np.array([p]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i,r] for i,r in enumerate(res) if r>ERROR_THRESHOLD]
    # sort by strength of probability
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({"intent": classes[r[0]], "probability": str(r[1])})
    return return_list

def getResponse(ints, intents_json):
    tag = ints[0]['intent']
    list_of_intents = intents_json['intents']
    for i in list_of_intents:
        if(i['tag']== tag):
            result = random.choice(i['responses'])
            break
    return result

def chatbot_response(msg):
    ints = predict_class(msg, model)
    res = getResponse(ints, intents)
    return res


import tkinter as tk
from tkinter import *
import time

def send():
    msg = EntryBox.get("1.0", 'end-1c').strip()
    EntryBox.delete("0.0", END)

    if msg != '':
        # User message
        ChatLog.config(state=NORMAL)
        ChatLog.insert(END, "You: " + msg + '\n\n', 'user-msg')
        ChatLog.config(state=DISABLED)
        ChatLog.yview(END)

        # Bot typing effect
        bot_msg = Label(ChatLog, text="Bot: ", wraplength=250, font=("Helvetica", 12), anchor='w', justify='left')
        bot_msg.pack()

        res = chatbot_response(msg)
        for char in res:
            bot_msg.config(text=bot_msg.cget("text") + char)
            ChatLog.yview(END)
            base.update_idletasks()  # Update the window
            time.sleep(0.02)  # Faster typing effect for the final response

        ChatLog.yview(END)

        # Remove the temporary Label widget used for typing effect
        bot_msg.destroy()

        # Display the final bot response
        ChatLog.config(state=NORMAL)
        ChatLog.insert(END, "Bot: " + res + '\n\n', 'bot-msg')
        ChatLog.config(state=DISABLED)
        ChatLog.yview(END)

base = Tk()
base.title("ChatBot")
base.geometry("500x600")
base.resizable(width=FALSE, height=FALSE)
base.configure(bg='#F0F0F0')  # Set background color

# Create Chat window
ChatLog = Text(base, bd=0, bg="#F0F0F0", height="10", width="60", font=("Helvetica", 14), wrap=WORD)
ChatLog.tag_configure('user-msg', foreground='#004080', justify='right')
ChatLog.tag_configure('bot-msg', foreground='#800040', justify='left')
ChatLog.config(state=DISABLED)

# Bind scrollbar to Chat window
scrollbar = Scrollbar(base, command=ChatLog.yview, cursor="heart")
ChatLog['yscrollcommand'] = scrollbar.set

# Create Button to send message
# Create Button to send message
SendButton = Button(base, font=("Helvetica", 14, 'bold'), text="Send", width="15", height=3,
                    bd=0, bg="#32de97", activebackground="#3c9d9b", fg='#ffffff', command=send)


# Create the box to enter message
EntryBox = Text(base, bd=0, bg="#EAEAEA", fg='#000000', width="35", height="4", font=("Helvetica", 14), wrap=WORD)
# Bind the <Return> key to the send function
EntryBox.bind("<Return>", lambda event=None: SendButton.invoke())

# Place all components on the screen
scrollbar.place(x=476, y=6, height=496)
ChatLog.place(x=6, y=6, height=496, width=470)
EntryBox.place(x=6, y=510, height=80, width=350)
SendButton.place(x=362, y=510, height=80, width=114)

base.mainloop()

base.mainloop()
