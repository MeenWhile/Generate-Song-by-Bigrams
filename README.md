# NLP-Generate-Song-by-Bigrams

## Objective
โปรเจ็คนี้เป็นโปรเจ็คที่จัดทำขึ้นเพื่อส่งในวิชา Text Analytics ของสถาบันบัณฑิตพัฒนบริหารศาสตร์(NIDA) โดยวัตถุประสงค์ของโปรเจ็คนี้ คือการ Generate เพลง ด้วย bi-gram

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
