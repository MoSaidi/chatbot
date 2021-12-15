from tkinter import *
from tkinter import ttk
import time
import random
import json
import torch
from model import NeuralNet
from nltk_utils import bag_of_words, tokenize
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
with open('intents.json', 'r') as json_data:
    intents = json.load(json_data)
FILE = "data.pth"
data = torch.load(FILE)
input_size = data["input_size"]
hidden_size = data["hidden_size"]
output_size = data["output_size"]
all_words = data['all_words']
tags = data['tags']
model_state = data["model_state"]

model = NeuralNet(input_size, hidden_size, output_size).to(device)
model.load_state_dict(model_state)
model.eval()
c1 = '#263238'
c2 = '#faa21f'
c3 = '#1e282d'
c6 = '#577e75'

c4 = '#faa21f'
c5 = '#577e75'

c7 = '#1e282d'
c8 = '#faa21f'


def welcome_to_info():
    frame_welcome.pack_forget()
    frame_info.pack()

def info_to_chat():
    frame_info.pack_forget()
    frame_chat.pack()

def submit():
    global chat_raw
    chat_raw = entry.get('1.0', 'end-1c')
    entry.delete('1.0', END)
    chat=tokenize(chat_raw)
    X = bag_of_words(chat, all_words)
    X = X.reshape(1, X.shape[0])
    X = torch.from_numpy(X).to(device)
    output = model(X)
    _, predicted = torch.max(output, dim=1)
    tag = tags[predicted.item()]
    probs = torch.softmax(output, dim=1)
    prob = probs[0][predicted.item()]
    global label_request
    label_request = Label(frame_chats, text=chat_raw, bg=c4, fg=c7, justify=LEFT, wraplength=300,
                          font='Verdana 10 bold')
    label_request.pack(anchor='w')
    global answer
    if prob.item() > 0.75:
        for intent in intents['intents']:
            if tag == intent["tag"]:
                answer=random.choice(intent['responses'])
    else:
        answer="I don't understand........."
    get_response()

def get_response():
    global label_response
    label_response = Label(frame_chats, text=answer, bg=c5, fg=c8, justify=LEFT, wraplength=300, font='Verdana 10 bold')
    label_response.pack(anchor='e')

def refresh_screen () :

    for widget in frame_chats.winfo_children():
        widget.destroy()
    label_space = Label (frame_chats , bg = c1 ,  text = '')
    label_space.pack()


def chat_to_info():
    frame_chat.pack_forget()
    frame_info.pack()


def info_to_welcome():
    frame_info.pack_forget()
    frame_welcome.pack()


def info():
    global myname
    myname = entry_user.get('1.0', 'end-1c')

    global chatbot
    chatbot = entry_chat.get('1.0', 'end-1c')

    if myname == "" or chatbot == "":
        Label(frame_info, text="Fill both fields to proceed.", bg="red", fg="white", font='Verdana 11 bold').place(
            x=182, y=96)
        return

    entry_user.delete('1.0', END)
    entry_chat.delete('1.0', END)

    frame_info.pack_forget()
    frame_chat.pack()


root = Tk()

# ----------------------------------------------------------------------------------------------------

"""  images used in window  """

back = PhotoImage(file='arrow_behind.png')

front = PhotoImage(file='arrow_ahead.png')

exitt = PhotoImage(file='exit.png')

screen_1 = PhotoImage(file='image_5.png')

submit_img = PhotoImage(file='image_8.png')


# ---------------------------------------------------------------------------------------------------------------------

"""     WELCOME FRAME    """
"""    first frame containing time date and welcome messages """

frame_welcome = Frame(root, bg=c1, height='670', width='550')
frame_welcome.pack_propagate(0)
frame_welcome.pack()

welcome = Label(frame_welcome, text='Welcome', font="Vardana 40 bold", bg=c1, fg="white")
welcome.place(x=160, y=200)

welcome_chatbot = Label(frame_welcome, text='I am Chatbot ! ', font="Helvetica 15 bold italic", bg=c1, fg=c6)
welcome_chatbot.place(x=200, y=270)

pic_1 = Label(frame_welcome, image=screen_1)
pic_1.place(x=-2, y=357)

button_front = Button(frame_welcome, image=front, relief="flat", bg=c1, bd="3px solid black",
                      command=welcome_to_info).place(x=470, y=10)

# __________________________________________________________________

"""  time option  """


def clock():
    current = time.strftime("%H:%M:%S")
    label_time = Label(frame_welcome, bd=5, text=current, height=1, width=8, font='Ariel 11 bold', fg="white",
                       relief='groove', bg=c3)
    label_time.place(x=120, y=63)

    label_time.after(1000, clock)


button_time = Button(frame_welcome, text='Time', height=1, font='Vardana 10 bold', width=8, bg=c2, fg=c1, command=clock)
button_time.place(x=30, y=63)

# _____________________________________________________________________________

"""    date option   """


def date():
    try:
        date = time.strftime("%d %B , 20%y")
        label_date = Label(frame_welcome, bd=5, relief='groove', text=date, bg=c3, fg="white", height=1,
                           font='Ariel 11 bold')
        label_date.place(x=400, y=63)

        label_date.after(86400000, date)

    except AttributeError:
        print('')


button_date = Button(frame_welcome, text='Date', height=1, font='Vardana 10 bold', width=8, bg=c2, fg=c1, command=date)
button_date.place(x=310, y=63)

# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

"""     INFO FRAME   """
"""     frame of entering names    """

frame_info = Frame(root, bg=c1, height='670', width='550')
frame_info.pack_propagate(0)

spacer1 = Label(frame_info, bg=c1)
spacer1.pack()

spacer2 = Label(frame_info, bg=c1)
spacer2.pack()

label_sub = Label(frame_info, text="Enter Information", bg=c1, fg="white", font='Verdana 30 italic')
label_sub.pack()

user_name = Label(frame_info, text='Enter your name : ', bg=c1, fg=c2, font='Ariel 15')
user_name.place(x=80, y=130)

entry_user = Text(frame_info, bg=c6, fg="white", height='1', width='40', font='Ariel 15')
entry_user.focus()
entry_user.place(x=80, y=170)

chatbot_name = Label(frame_info, text='Give Chatbot a Name : ', bg=c1, fg=c2, font='Ariel 15')
chatbot_name.place(x=80, y=220)

entry_chat = Text(frame_info, bg=c6, fg="white", height='1', width='40', font='Ariel 15')
entry_chat.place(x=80, y=260)

button_1 = Button(frame_info, text='submit', font='Vardana 10 bold', bg=c2, fg=c1, command=info)
button_1.place(x=470, y=330)

button_back = Button(frame_info, image=back, relief="flat", bg=c1, command=info_to_welcome).place(x=10, y=10)

# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

"""     TOPIC FRAME   """
""""   frame for topic selection     """

# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

"""         CHAT FRAME   """
""""       main chat screen   """

frame_chat = Frame(root, bg=c1, height='670', width='550')
frame_chat.pack_propagate(0)

frame_top = Frame(frame_chat, bg=c3, height='100', width='550')
frame_top.pack()

label_topic = Label(frame_top, bg=c3, fg='white', font='Verdana 20 bold ')
label_topic.pack(pady='40')

frame_spacer = Frame(frame_top, bg=c2, height="10", width="550")
frame_spacer.pack()

bottom_frame = Frame(frame_chat, bg=c2, height='100', width='550')
bottom_frame.pack_propagate(0)
bottom_frame.pack(side=BOTTOM)

button = Button(bottom_frame, image=submit_img, relief="flat", font='Vardana 10 bold', bg=c3,command=submit )
button.place(x=410, y=27)

entry = Text(bottom_frame, bg=c3, fg=c6, height='5', width='45', font='Verdana 10')
entry.bind('<Return>')
entry.place(x=30, y=10)

frame_chats = Frame(frame_chat, bg=c1, height='450', width='500')
frame_chats.pack_propagate(0)
frame_chats.pack()
label_space = Label(frame_chats, bg=c1).pack()

button_refresh = Button(frame_chat, bg=c3, fg=c2, text='refresh', font='Vardana 10 bold',command=refresh_screen)
button_refresh.place(x=440, y=80)
button_back = Button(frame_chat, image=back, relief="flat", bg=c3, command=chat_to_info).place(x=10, y=10)
button_front = Button(frame_chat, image=exitt, relief="flat", bg=c3, command=root.destroy).place(x=440, y=10)

# -----------------------------------------------------------------------------------------------------------

root.mainloop()