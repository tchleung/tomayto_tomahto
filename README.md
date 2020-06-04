# Tomayto, Tomahto
# A Spoken-Language Classfier
![cover_image](https://reasoningwithvolcanoes.files.wordpress.com/2019/09/aboutusbanner.png)

#### ![Click here](TBD) for a 3-minute summary presentation 

# Table of Contents
<!--ts-->
1. [Background and Motivation](#background-and-motivation)
2. [Data](#data) 
4. [Exploratory Analysis](#exploratory-analysis)
    * [KMeans](#kmeans)
    * [LDA](#lda)
    * [NMF](#nmf)
5. [Use Case](#use-case)
    * [Pre-Launch](#pre-launch)
    * [Launch](#launch)
6. [Conclusion](#conclusion)
7. [Future work](#future-work)
8. [Credits](#credits)
<!--te-->

## **Background and Motivation**
There are many different languages, dialects, and accents in the world. Someone from the American South speaks quite differently from a New Yorker and a British person ("Tomayto, Tomahto"), although the three groups of people are all speaking English.
<br/><br/>
Just like English, Mandarin Chinese also has many different variations. A person from Beijing can probably tell if he/she is speaking to someone from Shanghai, Hong Kong, or Taiwan, even though they are all speaking Chinese.
<br/><br/>
<img src="https://www.sinologyinstitute.com/sites/default/files/illustration_0.png" width=550>
<br/><br/>
I was born and raised in Hong Kong, where we were taught three languages in school (Cantonese, English and Mandarin Chinese). However, Mandarin was not a language that was widely used in our day-to-day lives. After I moved to the US, throughout college and in my adult life, I have made friends with a number of Mandarin-speakers from both Mainland China and Taiwan. My Mandarin has gotten better but it is still quite difficult for me to tell apart the accent sometimes.
<br/><br/>
As a data scientist, I have two objectives for this project:
1. Train a model to tell between a Chinese accent versus a Taiwanese accent
2. Use the model to guess the origin of the Mandarin-speakers around me

## **Data**
<img src="https://blog.mozilla.org/internetcitizen/files/2017/06/moz_blog_common-voice_1920_v02.png" width=550>
