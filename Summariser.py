#Author - Samuel Sharkey 18676848
#Summarisation artefact for Dissertation

from summarizer import Summarizer,TransformerSummarizer
from tkinter import *
from tkinter import filedialog, messagebox
import os
import json 
from transformers import T5Tokenizer, T5ForConditionalGeneration, T5Config
import torch
import time

backg = 'black'
window = Tk()
window.title('Summarization')
window.geometry("700x450")
window.resizable(width=False, height=False)
window['background']= 'black'

def open_txt():
    text_file = filedialog.askopenfilename(initialdir="/textfiles",
    title="Open Text File", filetype=(("Text Files", "*.txt"),))
    text_file = open(text_file, 'r',encoding='utf-8')
    body = text_file.read()

    return body

T = Label(window, text="Summarisation")
T.config(font=("courier",16),fg="white",bg=backg)
T.pack(pady=10)

my_text = Text(window, background='#101010', foreground="#39ff14", borderwidth=18, relief='sunken',width=65, height=12,wrap=WORD)
my_text.pack(pady=10)

variable = StringVar(window)
variable.set("Select A Model") # default value

w = OptionMenu(window, variable, "BERT", "GPT-2", "T5")
w.place(relx=0.5, rely=0.7, anchor=CENTER)

ratio = IntVar(window)
ratio.set("Select a Maximum length")

r = OptionMenu(window,ratio,"100","200","300","400","500","1000")
r.place(relx=0.5, rely=0.79, anchor=CENTER)

print(torch.cuda.is_available())

def ok(): 
    
    answer = messagebox.askyesno("Warning", "Contuining will download large files. Continue?")
    if answer == False:
        window.destroy()

    my_text.delete('1.0', END)
    body = open_txt()
    start_time = time.time()

    if variable.get() == "BERT":
        ratioSet = ratio.get()
        T.config(text="BERT")
        bert_model = Summarizer()
        output = ''.join(bert_model(body, min_length=10,max_length=ratioSet))
        print(output)
        print(time.time() - start_time)

    elif variable.get() == "GPT-2":
        ratioSet = ratio.get()
        T.config(text="GPT-2")
        GPT2_model = TransformerSummarizer(transformer_type="GPT2",transformer_model_key="gpt2-xl") #335m
        output = ''.join(GPT2_model(body, min_length=10,max_length=ratioSet))
        print(output)
        print(time.time() - start_time)

    elif variable.get() == "T5":
        ratio1 = ratio.get()
        ratioSet = int(ratio1)
        print("ratio", ratioSet)

        T.config(text="T5")
        model = T5ForConditionalGeneration.from_pretrained('t5-large')
        tokenizer = T5Tokenizer.from_pretrained('t5-large')
        device = torch.device('cpu')

        preprocess_text = body.strip().replace("\n","")
        t5_prepared_Text = "summarize: "+preprocess_text
        print ("original text preprocessed: \n", preprocess_text)

        tokenized_text = tokenizer.encode(t5_prepared_Text, return_tensors="pt")

        summary_ids = model.generate(tokenized_text,
                                            num_beams=4,
                                            no_repeat_ngram_size=2,
                                            min_length=90,
                                            max_length=ratioSet,
                                            early_stopping=True)

        output = tokenizer.decode(summary_ids[0], skip_special_tokens=True)

        print ("\n\nSummarized text: \n",output)
        print(time.time() - start_time)

    my_text.insert(END, output)

button = Button(window, text="OK", command=ok)
button.place(relx=0.5, rely=0.88, anchor=CENTER)

window.mainloop()

# @inproceedings{wolf-etal-2020-transformers,
#     title = "Transformers: State-of-the-Art Natural Language Processing",
#     author = "Thomas Wolf and Lysandre Debut and Victor Sanh and Julien Chaumond and Clement Delangue and Anthony Moi and Pierric Cistac and Tim Rault and Rémi Louf and Morgan Funtowicz and Joe Davison and Sam Shleifer and Patrick von Platen and Clara Ma and Yacine Jernite and Julien Plu and Canwen Xu and Teven Le Scao and Sylvain Gugger and Mariama Drame and Quentin Lhoest and Alexander M. Rush",
#     booktitle = "Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing: System Demonstrations",
#     month = oct,
#     year = "2020",
#     address = "Online",
#     publisher = "Association for Computational Linguistics",
#     url = "https://www.aclweb.org/anthology/2020.emnlp-demos.6",
#     pages = "38--45"
# }