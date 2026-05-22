from pydub import AudioSegment #audio manipulation: slice audiom change formats etc 
import librosa #audio features, visualising sound waves
import sounddevice as sd
from scipy.io.wavfile import write
import winsound
import time #to impersonate keys audio 
import numpy as np 
import wave
import random #for genetic algo 
import matplotlib.pyplot as plt 
#models libraries
from collections import defaultdict
from sklearn.neighbors import KNeighborsClassifier
from itertools import groupby #for compression 
import noisereduce as nr 

#PIPELINE
#RAW MELODY -> COMPRESSED -> SMOOTHED OUT -> FILTERED -> OPTIMIZED 
#=====================================================DATA=========================================================================================
#Piano keys- varying; will match with exact project one's later 
piano_notes={
    "C4": 261.63,
    "D4": 293.66,
    "E4": 329.63,
    "F4": 349.23,
    "G4": 392.00,
    "A4": 440.00
} #our piano has 6 keys

melodies = {
    "Mary Had A Little Lamb": 
        [("E4",1),("D4",1),("C4",1),("D4",1),("E4",3),("D4",3),("E4",1),("G4",2),("E4",1),("D4",1),("C4",1),("D4",1),("E4",4),("D4",2),("E4",1),("D4",1),("C4",1)],

    "Twinkle Twinkle Little Star": 
        [("C4",2),("G4",2),("A4",2),("G4",1),("F4",2),("E4",2),("D4",2),("C4",1),("G4",2),("F4",2),("E4",2),("D4",1),("G4",2),("F4",2),("E4",2),("D4",1),("C4",2),("G4",2),("A4",2),("G4",1),("F4",2),("E4",2),("D4",2),("C4",1)],

    "Ode To Joy": 
        [("E4",2),("F4",1),("G4",2),("F4",1),("E4",1),("D4",1),("C4",2),("D4",1),("E4",1),("D4",1),("C4",1),("E4",2),("F4",1),("G4",2),("F4",1),("E4",1),("D4",1),("C4",2),("D4",1),("E4",1),("D4",1),("C4",1)],

    "Frere Jacques": 
        [("C4",1),("D4",1),("E4",1),("C4",1),("C4",1),("D4",1),("E4",1),("C4",1),("E4",1),("F4",1),("G4",2),("E4",1),("F4",1),("G4",2),("G4",1),("A4",1),("G4",1),("F4",1),("E4",1),("C4",1),("G4",1),("A4",1),("G4",1),("F4",1),("E4",1),("C4",1),("C4",1),("G4",1),("C4",2),("C4",1),("G4",1),("C4",2)],

    "Jingle Bells": 
        [("E4",3),("E4",2),("E4",1),("G4",1),("C4",1),("D4",1),("E4",1),("F4",4),("F4",1),("F4",1),("F4",1),("E4",4),("E4",1),("E4",1),("D4",2),("E4",1),("D4",1),("G4",1),("E4",3),("E4",2),("E4",1),("G4",1),("C4",1),("D4",1),("E4",1),("F4",4),("F4",1),("F4",1),("E4",2),("G4",2),("F4",1),("D4",1),("C4",1)],

    "Row Your Boat": 
        [("C4",3),("C4",1),("D4",1),("E4",3),("D4",1),("E4",1),("F4",1),("G4",3),("C4",3),("G4",3),("E4",3),("C4",3),("G4",1),("F4",1),("E4",1),("D4",1),("C4",1)],

    "London Bridge": 
        [("G4",1),("A4",1),("G4",1),("F4",1),("E4",1),("F4",1),("G4",1),("D4",1),("E4",1),("F4",1),("E4",1),("F4",1),("G4",2),("G4",1),("A4",1),("G4",1),("F4",1),("E4",1),("F4",1),("G4",1),("D4",1),("G4",1),("E4",1),("C4",1)],

    "This Old Man": 
        [("G4",1),("E4",1),("G4",2),("E4",1),("G4",1),("A4",1),("G4",1),("F4",1),("E4",1),("D4",1),("E4",1),("F4",1),("E4",1),("F4",1),("G4",1),("C4",4),("C4",1),("D4",1),("E4",1),("F4",1),("G4",2),("D4",2),("F4",1),("E4",1),("D4",1),("C4",1)],

    "Happy Birthday": 
        [("C4",2),("D4",1),("C4",1),("F4",2),("E4",2),("C4",2),("D4",1),("C4",1),("G4",2),("F4",2),("C4",2),("C4",1),("A4",1),("F4",1),("E4",1),("D4",1),("A4",2),("A4",1),("F4",1),("G4",1),("F4",1)],

    "Wheels On The Bus": 
        [("C4",1),("F4",5),("A4",1),("C4",1),("A4",1),("F4",1),("G4",3),("E4",1),("D4",1),("C4",2),("F4",5),("A4",1),("C4",1),("A4",1),("F4",1),("G4",1),("C4",1),("F4",1)],

    "Heart And Soul": 
        [("C4",2),("E4",2),("A4",2),("C4",2),("D4",2),("F4",2),("G4",2),("A4",2)],
}
#======================================================================================================================================================================
#===============================Functions==============================================================================================================================
def construct_markov(melodies):
    note_remap={"B4": "A4", "G3": "G4", "A3": "A4", "B3": "A4", "C5": "C4", "Bb": "A4", "F#": "F4"}
    valid_notes=list(piano_notes.keys())
    transition_counts=defaultdict(lambda: defaultdict(int))
    #from this function we get probabilities of transitions 
    for song, notes in melodies.items():
        note_sequence=[]
        for note, count in notes:
            note = note_remap.get(note, note)
            if note in valid_notes:
                note_sequence.append(note)
        for i in range(len(note_sequence) - 1):
            current = note_sequence[i]
            next_note = note_sequence[i + 1]
            transition_counts[current][next_note] += 1
    #convert to probabilities
    transition_probs = {}
    for note, transitions in transition_counts.items():
        total = sum(transitions.values())
        transition_probs[note] = {
            next_n: round(count/total, 3)
            for next_n, count in transitions.items()
        }
    return transition_probs

X_train=np.array([[261.63], [293.66], [329.63], [349.23], [392.00], [440.00]])
Y_train=["C4", "D4", "E4", "F4", "G4", "A4"]
knn=KNeighborsClassifier(n_neighbors=1)#were dealing with one dimension so only one neighbor is fine
knn.fit(X_train, Y_train)

#helper functions
def closest_note(freq):
    return knn.predict([[freq]])[0]

def heuristic(melody, transition_probs, original=None):
    #evaluates the music-ness? of a melody 
    #evaluating on the basis of:
    #are the jumps sharp?
    #too much repetition will also be bad
    #penalise silence and noise
    score=0 #initial score is 0 
    if melody and isinstance(melody[0], tuple):
        notes=[str(n) for n, c in melody]
    else:
        notes=[str(n) for n in melody]
    notes_list=list(piano_notes.keys())
    for i in range(len(melody)):
        current=notes[i]
        if i>0:
            prev=notes[i-1]
            prob=transition_probs.get(prev, {}).get(current, 0)
            score+=prob*10
        if i>0:
            prev_freq=piano_notes[notes[i-1]]
            curr_freq=piano_notes[current]
            diff=abs(curr_freq-prev_freq)
            if diff<40: #stp wicse c4 to d4 very smooth
                score+=3
            elif diff<80: #small jump
                score+=1
            else:
                score-=2 #big jump
        if i>0 and notes[i]==notes[i-1]: #repetition penality 
            score-=2
        if i>1 and notes[i]==notes[i-1]==notes[i-2]:
            score-=4
        #okay thats decent enough scoring for now
        #we need to introduce a similarity score too
    #problem with similarity is that 
    #too strong? child sounds like parent completely
    #no cimilarity-> then all trials generate the same type of audio 
    if original is not None:
        similarity_score = 0
        if original and isinstance(original[0], tuple):
            original_notes = [str(n) for n, c in original]
        else:
            original_notes = [str(n) for n in original]
        
        matches = sum(1 for i in range(min(len(notes), len(original_notes)))
                     if notes[i] == original_notes[i])
        similarity_ratio = matches / max(len(original_notes), 1)
        similarity_score = similarity_ratio * 20  # max 20 points for similarity
        
        score += similarity_score

    return score

def filter_dominant(compressed):
    #remove notes that play wayyyyyyyyyyyy too long
    #treat them as noise/garbage
    total=sum(count for _, count in compressed)
    filtered=[]
    for note, count in compressed:
        if count/total<0.6:
            filtered.append((note, count))
        else:
            keep=int(total*0.3)
            filtered.append((note, keep))
    if not filtered:
        return [(str(n), c) for n, c in compressed]
    return remerge(filtered)

#we also 
def mutate_individual(melody, individual, note_neighbors, mutation_rate):
    mutated=individual.copy()
    max_change=max(1, len(mutated)//3)
    changes=0
    base_melody=[(str(note), int(count)) for note, count in melody]
    for i in range(len(mutated)):
        if changes>=max_change:
            break #max changes occured
        if random.random()<mutation_rate:
            old_note, old_count=mutated[i]
            new_note=random.choice(note_neighbors[old_note])
            original_count=base_melody[i][1]
            min_count=max(2, int(original_count*0.5))
            max_count=int(original_count*1.2)
            new_count = random.randint(min_count, max(min_count, max_count))
            mutated[i]=(new_note, new_count)
            changes+=1
    return mutated 

def genetic_algorithm(melody, transition_probs, generations=100, pop_size=20, mutation_rate=0.2):
    #a big issue with the genetic algorithm is that-> it can chnange the audio to something that is not similar to the actual audio 
    base_melody=[(str(note), int(count)) for note, count in melody]
    note_neighbors={
        "C4":["C4", "D4"],
        "D4":["C4","D4", "E4"],
        "E4":["D4", "E4", "F4"],
        "F4":["E4","F4","G4"],
        "G4":["F4","G4","A4"],
        "A4":["G4","A4"]
    }
    #extracting note names 
    #we need to keep the melody being generated similar to our raw one 
    #in iterms of notes and their count 
    population=[]
    #originaally we only have ONE MELODY
    #we can make variations of that to act as parents 
    #but we need to make sure we do not change the vibe of the melody '
    for _ in range(pop_size):
        individual=base_melody.copy()
        individual=mutate_individual(melody, individual, note_neighbors, mutation_rate)
        population.append(individual)
    #contraints:
    #to introduce optimization: keys can onlt chnage to close neighbors
    #count: if still do big-> reduce but not too much 
    #similarity score introduced in heristic 
    population[0]=base_melody.copy()

    for gen in range(generations):
        scored=[(heuristic(ind, transition_probs, base_melody), ind) for ind in population]
        scored.sort(reverse=True, key=lambda x:x[0])
        survivors=[ind for _, ind in scored[:pop_size//2]]
        children=[]
        for _ in range(pop_size//2):
            p1,p2=random.sample(survivors,2)

        if len(p1) < 2:
            child = p1[:]
        else:
            cut = random.randint(1, len(p1) - 1)
            child = p1[:cut] + p2[cut:]
            
            child=mutate_individual(melody, child, note_neighbors, mutation_rate)
            children.append(child)
        population=survivors+children
    best=max(population, key=lambda x:heuristic(x, transition_probs, base_melody))
    print("Original score:", heuristic(base_melody, transition_probs, base_melody))
    print("Best score:", heuristic(best, transition_probs, base_melody))
    print("Changed notes:", [(i, base_melody[i], best[i]) 
                            for i in range(min(len(base_melody), len(best))) 
                            if base_melody[i] != best[i]])
    return best



def smooth_melody(raw, transition_probs, threshold=0.04):
    if not raw:
        return []
    #a big goal is to also drop flickers
    #notes that last for 1 1 moment 
    smoothed=[(str(raw[0][0]), raw[0][1])]
    for i in range(1, len(raw)):
        current_note=str(raw[i][0])
        current_count=raw[i][1]
        prev_note=str(smoothed[-1][0])
        prev_count=smoothed[-1][1]
        prob=transition_probs.get(prev_note, {}).get(current_note, 0) #get the ransition probabilty of going from old note ot new note
        adjusted_threshold=threshold if current_count > 2 else threshold * 1.5
        if prob >= adjusted_threshold:
            #we'll add that in 
            if current_count==1:
                #absorb it into
                smoothed[-1]=(prev_note, prev_count+1)
                pass
            else:
                smoothed.append((str(current_note), current_count))
        else:
            #merge into previous
            smoothed[-1] = (str(prev_note), prev_count + current_count) #else go to older melody and add it into its count 

    return smoothed
def remerge(melody):
    if not melody:
        return []
    remerged = [melody[0]]
    for note, count in melody[1:]:
        if str(note) == str(remerged[-1][0]):
            remerged[-1] = (remerged[-1][0], remerged[-1][1] + count)
        else:
            remerged.append((note, count))
    return remerged

def get_freq(chunk):
    samples = np.array(chunk.get_array_of_samples()).astype(np.float32)
    if chunk.channels == 2:
        samples = samples.reshape((-1, 2)).mean(axis=1)
    samples=samples - np.mean(samples)
    if np.max(np.abs(samples)) > 0:
        samples = samples / np.max(np.abs(samples))
    corr=np.correlate(samples, samples, mode='full')
    corr=corr[len(corr)//2:]
    peak=np.argmax(corr[50:]) + 50  # ignore zero lag
    freq=chunk.frame_rate / peak if peak != 0 else None

    return freq

def extract_raw_melody(audio_path):
    y, sr=librosa.load(audio_path)
    #pyin is specifically designed for monophonic 
    #(single voice/instrument) pitch detection
    f0, voiced_flag, _=librosa.pyin(
        y,
        fmin=150,
        fmax=550,
        sr=sr
    )
    melody=[]
    for i, (freq, voiced) in enumerate(zip(f0, voiced_flag)):
        if voiced and freq is not None:
            note = str(closest_note(freq))
            melody.append(note)
    return melody
#==========================================================================================================================================================
#=============================================================MAIN CODE==================================================================================

transition_probs=construct_markov(melodies)

for note in piano_notes:
    if note in transition_probs:
        print(f"\n{note} ->")
        for next_note, prob in sorted(transition_probs[note].items(), key=lambda x: x[1], reverse=True):
            print(f"   {next_note}: {prob}")


#first we need to record audio
fs=44100
seconds=5

#===========================================================RECORDING=============================================================================================
#Recording
print("Recording started....")
my_recording=sd.rec(int(seconds*fs), samplerate=fs, channels=2)
sd.wait() #wait till recording in finished
print("Recording finsihed..")
#saving as wav file for later analasys
write('output.wav', fs, my_recording)

#============================================================================================================================================================
#Variable 
frequencies_read=[]
melody_generated=[]
raw_melody=[]
#Tester to play notes 
print("Playing tester notes: ")
for note, freq in piano_notes.items():
    winsound.Beep(int(freq), 500)  # 500 ms
    time.sleep(0.1)
#approach 1: we need to slice the audio to find frequencies of each parth
#then we will match each frquency to its nearest piano_freq

#we need to pick chunk size carefully
#tooo small-=> causes alot of repetition
#too larger=> and we miss out on frqwuency shifts 
#another option is that we separate chunks when the frequency chaanges


if __name__ == "__main__":
#============================================================CHUNKS===============================================================================================
    audio=AudioSegment.from_wav("output.wav") #grabbing back recorded file 
    splice_length=200 #basically we want to segregate splices of audio to get the best splices ot replicate melody 
    #training a model to find best splice length will not work because we have no data to train it on 
    chunks=[audio[i:i+splice_length] for i in range(0, len(audio), splice_length)] #chunks audios 

    #now we get frequency of each chunk and construct a raw melody
    #we can compress it into 
    #{"A4": 3, "A4":3}

    #==========================================================================================================
    #===========================Raw melody=============================================
    #print("Raw melody: ", raw_melody)
    raw_melody=extract_raw_melody("output.wav") #instead of dpending on chunks that get us exact frquency im moving onto a more raw-messy data
    compressed_melody=[(note, len(list(group))) for note, group in groupby(raw_melody)]
    raw_freq=[]
    print("Compressed: ", compressed_melody)
    #play original audio 

    #playing the raw melody 
    for note in raw_melody:
        frequency=piano_notes[note]
        raw_freq.append(frequency)
        #winsound.Beep(int(frequency), 100)  # 500 ms
        #time.sleep(0.1)

    #play the raw melody 
    for note, count in compressed_melody:
            base_time=60
            freq = piano_notes[str(note)]
            duration = base_time * count  # scale by note length
            winsound.Beep(int(freq), int(duration))
    time.sleep(0.5)  
    
    #=====================================================================================
    #=================================Smoother melody=====================================
    #now we need to smooth it out
    #print(construct_markov(melodies))
    print("Creating a smoother melody...")
    compressed_melody=filter_dominant(compressed_melody)
    smoothed_melody=smooth_melody(compressed_melody, transition_probs, 0.04)
    smoothed_melody=remerge(smoothed_melody)
    print("Smooth melody: ", smoothed_melody)

    #2 APPROACHES
    #HMM-> to smoothed melodies
    #basically melodies that sound more musical will rate higher and be selected 
    #Algorithm choices 

    smooth_freq=[]
    for item in smoothed_melody:
        time_=60
        play_for=item[1]
        time_=time_*play_for
        note_=str(item[0])
        frequency=piano_notes[note_]
        winsound.Beep(int(frequency), time_)
    for item in smoothed_melody:
        note_ = str(item[0])
        for _ in range(item[1]):
            smooth_freq.append(piano_notes[note_])
    time.sleep(0.5)  
    #===========================================================================================================================================
    #=========================================================Optimized=========================================================================
    if len(smoothed) < 2:
        optimized = smoothed
    else:
        optimized = genetic_algorithm(
            smoothed,
            transition_probs,
            generations=200,
            pop_size=50,
            mutation_rate=0.4
    )
    print("Optimized Melody: ", optimized_melody)

    for note, count in optimized_melody:
        duration=min(int(count) * 60, 1000)
        winsound.Beep(int(piano_notes[str(note)]), duration)

    time.sleep(0.5)  
    optimized_freq = []
    for note, count in optimized_melody:
        for _ in range(int(count)):
            optimized_freq.append(piano_notes[str(note)])
    #============================================================================================================================================
    plt.plot(raw_freq, color="red", label="raw")
    plt.plot(smooth_freq, color="blue", label="smoothed")
    plt.plot(optimized_freq, color="green", label="optimized")
    plt.yticks(list(piano_notes.values()), list(piano_notes.keys()))
    plt.legend()
    plt.title("Raw vs Smoothed vs GA Optimized")
    plt.xlabel("Time")
    plt.ylabel("Note")
    plt.show()
    #HMM-> for melody smoothening: remove sharp changes and random jumping 
    #appraoch: right now we have this certain frequency-> what is the next most likely note
    #help remove outliers


    #an issue with this approach is that HMM might remvove outliers which were intentionally like placed 
    #Genetic Algorithm -> for melody optimization basically help it sound more musical
    #create different variaitons-> which scores high on musicality