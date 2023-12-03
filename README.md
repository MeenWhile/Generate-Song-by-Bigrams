# NLP-Generate-Song-by-Bigrams

## Objective
วัตถุประสงค์ของโปรเจ็คนี้คือการ Generate เพลง ด้วย bi-gram

## Simple Summary
step 1 : change all midi file to dataframe

step 2 : change dataframe to text

step 3 : generate new text by using bi-gram

step 4 : change new text to dataframe

step 5 : change new dataframe to midi file

## 0. Import Python Packages
เริ่มต้น เราได้ import library ที่จำเป็นในการวิเคราะห์ข้อมูลซึ่งประกอบด้วย
  1. random
  2. copy
  3. pandas
  4. mido
  5. nltk

```python
import random
import copy
import pandas as pd

import mido
from mido import MidiFile, MidiTrack, Message
import nltk
from nltk import bigrams, FreqDist, ConditionalFreqDist
```

## 1. Data Collection and Data Preparation
ต่อมา เราได้ทำการดึงเพลงเข้ามาใน python จากนั้นทำการ transpose เพลงเพื่อให้เพลงทั้งหมดอยู่ใน key เดียวกัน โดยในที่นี้ เราได้จัดทำให้ทุกเพลงอยู่ใน key B 
และจากนั้นเราได้ทำการเปลี่ยนเพลงที่เป็นสกุล .midi เป็น dataframe ด้วย library mido

```python
def extract_note_data(midi_filename):
    midi_file = MidiFile(midi_filename)
    note_data = []

    for track in midi_file.tracks:
        for msg in track:
            if msg.type == 'note_on':
                note_data.append(msg.note)

    return note_data

def transpose_notes(note_sequence, semitones):
    transposed_notes = []

    for note in note_sequence:
        transposed_note = note + semitones
        transposed_notes.append(transposed_note)

    return transposed_notes
```

```python
#Extract note and transpose to key B

note_df = []

#1
note_data = extract_note_data('midi1 C.mid')
note_data = transpose_notes(note_data, -1)
note_df.append(pd.DataFrame(note_data))

#2
note_data = extract_note_data('midi2 C#.mid')
note_data = transpose_notes(note_data, -2)
note_df.append(pd.DataFrame(note_data))

#3
note_data = extract_note_data('midi3 A.mid')
note_data = transpose_notes(note_data, 2)
note_df.append(pd.DataFrame(note_data))

#4
note_data = extract_note_data('midi4 Bb.mid')
note_data = transpose_notes(note_data, 1)
note_df.append(pd.DataFrame(note_data))

#5
note_data = extract_note_data('midi5 Ab.mid')
note_data = transpose_notes(note_data, 3)
note_df.append(pd.DataFrame(note_data))

#6
note_data = extract_note_data('midi6 B.mid')
note_df.append(pd.DataFrame(note_data))

#7
note_data = extract_note_data('midi7 Eb.mid')
note_data = transpose_notes(note_data, -4)
note_df.append(pd.DataFrame(note_data))

#8
note_data = extract_note_data('midi8 G.mid')
note_data = transpose_notes(note_data, 4)
note_df.append(pd.DataFrame(note_data))

#9
note_data = extract_note_data('midi9 E.mid')
note_data = transpose_notes(note_data, -5)
note_df.append(pd.DataFrame(note_data))

#10
note_data = extract_note_data('midi10 B.mid')
note_df.append(pd.DataFrame(note_data))

#11
note_data = extract_note_data('midi11 G.mid')
note_data = transpose_notes(note_data, 4)
note_df.append(pd.DataFrame(note_data))

#12
note_data = extract_note_data('midi12 Bb.mid')
note_data = transpose_notes(note_data, 1)
note_df.append(pd.DataFrame(note_data))

#13
note_data = extract_note_data('midi13 E.mid')
note_data = transpose_notes(note_data, -5)
note_df.append(pd.DataFrame(note_data))

#14
note_data = extract_note_data('midi14 Eb.mid')
note_data = transpose_notes(note_data, -4)
note_df.append(pd.DataFrame(note_data))

#15
note_data = extract_note_data('midi15 A.mid')
note_data = transpose_notes(note_data, 2)
note_df.append(pd.DataFrame(note_data))

#16
note_data = extract_note_data('midi17 E.mid')
note_data = transpose_notes(note_data, -5)
note_df.append(pd.DataFrame(note_data))

#17
note_data = extract_note_data('midi18 B.mid')
note_df.append(pd.DataFrame(note_data))

#18
note_data = extract_note_data('midi19 Eb.mid')
note_data = transpose_notes(note_data, -4)
note_df.append(pd.DataFrame(note_data))

#19
note_data = extract_note_data('midi20 A.mid')
note_data = transpose_notes(note_data, 2)
note_df.append(pd.DataFrame(note_data))

#20
note_data = extract_note_data('midi21 Eb.mid')
note_data = transpose_notes(note_data, -4)
note_df.append(pd.DataFrame(note_data))

#21
note_data = extract_note_data('midi22 E.mid')
note_data = transpose_notes(note_data, -5)
note_df.append(pd.DataFrame(note_data))

#22
note_data = extract_note_data('midi23 F#.mid')
note_data = transpose_notes(note_data, 5)
note_df.append(pd.DataFrame(note_data))

#23
note_data = extract_note_data('midi24 C.mid')
note_data = transpose_notes(note_data, -1)
note_df.append(pd.DataFrame(note_data))

#Dummy for output
note_data = extract_note_data('dummy.mid')
note_df.append(pd.DataFrame(note_data))
```

## 2. Use bi-gram to generate song
เมื่อเตรียม data เสร็จแล้ว ตอนนี้ data ทั้งหมดก็จะอยู่ในรูปแบบของ dataframe ต่อมาเราก็ดึงเฉพาะ column 'note' ในแต่ละ dataframe ออกมา และเปลี่ยนข้อมูลให้เป็น string เพื่อเตรียมนำเข้าขั้นตอน generate song by bi-gram

```python
#Change DataFrame to String

lst_note = []
for i in range(len(note_df)):
    lst_note.append(note_df[i][0].tolist())

    
str_lst = [[str(num) for num in inner_list] for inner_list in lst_note]
formatted_sentences = []
for inner_list in str_lst:
    sentence = ' '.join(inner_list)
    formatted_sentences.append(sentence)
```

และเริ่มต้น generate text(song) ด้วยคำสั่งด้านล่าง

```python
#Generate Text

final_text = '. '.join(formatted_sentences) + '.'
nltk.download('punkt')
corpus = final_text
tokens = nltk.word_tokenize(corpus)
bi_grams = list(bigrams(tokens))
cfd = ConditionalFreqDist(bi_grams)
cpd = nltk.ConditionalProbDist(cfd, nltk.MLEProbDist)
seed_word = "66"
generated_text = [seed_word]
max_words = len(extract_note_data('dummy.mid'))
for _ in range(max_words):
    next_word = cpd[seed_word].generate()
    generated_text.append(next_word)
    seed_word = next_word
generated_text = ' '.join(generated_text)
```

## 3. Change text back to MIDI file
สุดท้าย เมื่อเราได้ text ที่ generate ออกมาใหม่แล้ว เราก็ได้ทำการเปลี่ยน text กลับมาเป็น dataframe

```python
#Change text back to DataFrame

str_list = generated_text.split()
int_list = [int(num) for num in str_list]


note_df[-1][0] = int_list[:74]
```

และนำ dataframe ที่ได้นี้ แปลงกลับไปเป็น MIDI file

```python
#Change DataFrame to MIDI file

def create_midi_from_dataframe(dataframe, output_filename, midi_filename):
    
    midi_file = MidiFile(midi_filename)
    note_data = []
    
    mid = MidiFile()
    track = MidiTrack()
    mid.tracks.append(track)
    
    i = 0
    for track in midi_file.tracks:
        for msg in track:
            if msg.type == 'note_on':
                msg.note = note_df[-1][0][i]
                i += 1
                
    mid = midi_file
    mid.save(output_filename)
            
    
midi_filename = 'dummy.mid'   
output_filename = 'output.mid'  # Replace with the desired output filename
create_midi_from_dataframe(note_df, output_filename, midi_filename)
```

## Summary
โดยผลลัพธ์ที่ได้นั้น เราจะได้ไฟล์เพลงใหม่ที่ชื่อว่า output.mid ออกมา
