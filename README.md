# NLP-Summarizer
Natural language processing summarizer using 3 state of the art Transformer models: BERT, GPT2, and T5

This project aimed to provide insight and explanations to current limitations on Natural Language Processing models by exploring the Transformer model, the latest state-of-the-art NLP solution, as well as discussing possible use cases for such tools in a domestic and workplace environment. An in-depth explanation of the architecture and the limitations it aims to solve was provided, as well as how it can be used to infer various tasks. Numerous use cases of NLP were also explored and how tools such as this can be extremely useful and have a massive impact on today’s society, both domestically and in the workplace. Three specific Transformer models were implemented using a GUI to evaluate their effectiveness. The final artefact provides a user with an interaction between the models for document summarisation tasks of variable output lengths.

<H1> Working Example </H1>

Following example created using another student's project introduction, original word count was ~1000. 

<H3> Initial GUI </h3>
<p align="center">
  <img src="https://user-images.githubusercontent.com/45834305/152804705-a229d8d4-9f6c-4c0b-85b2-6a22e89013b5.png">
</p>

<H3> After Summarization </h3>

<p align="center">
  <img src="https://user-images.githubusercontent.com/45834305/152805385-131fdc0e-95de-48ac-b0c3-cadab0eae9c0.png">
</p>

<H32> Getting Started </h2>

All code is ran using Python version 3.8.8  
The artefact to be operated in it's entirety requires ~20GB of available space for 
downloads of the pre-trained models.

```!pip install bert-extractive-summarizer
!pip install transformers
!pip install spacy==2.0.12
!pip install torch
!pip install tk
```

Runtime will be displayed as an output in console
