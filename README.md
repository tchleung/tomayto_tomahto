# Tomayto, Tomahto
Language Classification - Audio Processing with Convolution Neural Network
<br/>
*By Tom Leung*

<p align="center">
<img src='https://reasoningwithvolcanoes.files.wordpress.com/2019/09/aboutusbanner.png'/>
<p/>

#### ![Click here](TBD) for a 3-minute summary presentation 

# Table of Contents
<!--ts-->
1. [Background and Motivation](#background-and-motivation)
2. [Data](#data) 
3. [EDA](#eda)
4. [CNN](#constructing-a-cnn)
    * [Pre-processing](#pre-processing)
    * [CNN Structure](#cnn-structure)
    * [Results](#results)
5. [Conclusion](#conclusion)
6. [Future work](#future-work)
7. [Credits](#credits)
<!--te-->

## **Background and Motivation**
There are many different languages, dialects, and accents in the world. Someone from the American South speaks quite differently from a New Yorker and a British person ("Tomayto, Tomahto"), although the three groups of people are all speaking English.
<br/><br/>
Just like English, Mandarin Chinese also has many different variations. A person from Beijing can probably tell if he/she is speaking to someone from Shanghai, Hong Kong, or Taiwan, even though they are all speaking Chinese.
<br/><br/>
<p align="center">
<img src="https://www.sinologyinstitute.com/sites/default/files/illustration_0.png" width=550>
<p/>
<br/><br/>
I was born and raised in Hong Kong, where we were taught three languages in school (Cantonese, English and Mandarin Chinese). However, Mandarin was not a language that was widely used in our day-to-day lives. After I moved to the US, throughout college and in my adult life, I have made friends with a number of Mandarin-speakers from both Mainland China and Taiwan. My Mandarin has gotten better but it is still quite difficult for me to tell apart the accent sometimes.
<br/><br/>
As a data scientist, I have two goals for this project:
1. Train a CNN to classify the audio clip between Mainland Chinese and Taiwanese
2. Use the model to guess the origin of the Mandarin-speakers around me

## **Data**
Data obtained from the [Mozilla Common Voice](https://voice.mozilla.org/en) project, a crowd-sourced database aimed to open source speech recognition. You can volunteer to record a short sentence of the language you choose, and you can also volunteer to validate the recordings of other users. I combined the Chinese (Taiwan) dataset with the Chinese (China) dataset.

## **EDA**
The dataset was not very balanced, Taiwan has ~3 times more clips than China.
<br/>
<p align="center">
    <img src="img/TBA"/>
<p/>
<br/><br/>
The dataset contained both validated and invalidated clips. To understand what it means, I volunteered to validate some Chinese audio clips, and came across several quiet recordings, trolls playing loud music, or people speaking a totally different language.

| Class | Validated | Not Validated |
|--------|----------------|------------|  
| Taiwan | 48968 | 21249 |
| China | 16898 | 2571 |

Volunteers who recorded their voice could also submit their age, gender, accent. Unfortunately, some of the metadata were missing. I was mostly concerned with gender balance. Taiwanese set has a good mix of men and women. In contrast, there were many more audio clips recorded by Chinese men than women.

| Class | Male | Female |
|--------|-----------|-----------|
| Taiwan | 22091 | 14367 |
| China | 10962 | 1851 |

Hence, I decided to curate my own balanced dataset. I randomly selected 1851 male and 1851 female audio clips from the validated pool of both Taiwan and China, then split them into train, test and hold-out set with a 8:1:1 ratio.

# **Constructing a CNN**
## **Pre-processing**
You may be wondering why covolutional neural network is chosen for this task. Let's first look at how we can visualize sound.

Most people have probably seen a waveform before. It is plotting amplitude over time (you can think of amplitude as loudness)
<br/>
<p align="center">
    <img src="img/TBA"/>
<p/>
<br/><br/>
What is missing from a waveform is the frequency (think of 'pitch'). This is where a spectrogram comes in. X axis is time, Y axis is frequency. Color denotes the amplitude.
<br/>
<p align="center">
    <img src="img/TBA"/>
<p/>
<br/><br/>
Another form of presentation is the Mel-frequency cepstral coefficients (MFCC)
<br/>
<p align="center">
    <img src="img/TBA"/>
<p/>
<br/><br/>
As you may suspect, audio recognition can actually be dealt with as an image recognition problem! So I turned all the audio clips into 128x256 shaped mel-scaled spectrograms, padded/trimmed the audio clips that are either too short or too long

### **Outline**
<br/>
<p align="center">
    <img src="img/TBA"/>
<p/>
<br/><br/>

## **CNN Structure**
I experimented with many different structures, the following structure with 3 sets of Conv-Pool-Dropout gave the highest accuracy

| Layers | Output Shape | # Parameter |
|--------|--------------|-------------|
| Conv2D | 128, 256, 32 | 832 |
| MaxPooling2D | 64, 256, 32 | 0 |
| Dropout | 64, 256, 32 | 0 |
| Conv2D | 64, 128, 64 | 51264 |
| MaxPooling2D | 32, 64, 64 | 0 |
| Dropout | 32, 64, 64 | 0 |
| Conv2D | 32, 64, 128 | 204928 |
| MaxPooling2D | 16, 32, 128 | 0 |
| Dropout | 16, 32, 128 | 0 |
| Flatten | 65536 | 0 |
| Dense | 256 | 16777472 |
| Dense | 2 | 514 |

Input shape: 5924 x 128 x 256

## **Results**

## **Conclusion**
<br/><br/>
One caveat. Due to the self-policing nature of the common voice project, there is no assurance that all the metadata are correct. I manually played a dozen samples and the voices did match the gender and the language, so it gives me a slight comfort that I have some good data.
<br/><br/>

## **Future Work**

## **Credit**


