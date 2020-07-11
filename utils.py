#######################################
### reference: https://github.com/YatingMusic/remi/utils.py, model.py
#######################################
#import chord_recognition
import numpy as np
import miditoolkit
import copy
import pickle
# parameters for input
DEFAULT_VELOCITY_BINS = np.linspace(0, 128, 32+1, dtype=np.int)
DEFAULT_FRACTION = 16
DEFAULT_DURATION_BINS = np.arange(60, 3841, 60, dtype=int)
DEFAULT_TEMPO_INTERVALS = [range(30, 90), range(90, 150), range(150, 210)]

# parameters for output
DEFAULT_RESOLUTION = 480

class datapreprocessing(object):
    def __init__(self, x_len, group_size,pickle_path, checkpoint_path):
        self.x_len = x_len
        self.pickle_path = pickle_path
        self.group_size = group_size
        self.checkpoint_path = checkpoint_path
        self.event2word, self.word2event = pickle.load(open(self.pickle_path, 'rb'))
    def extract_events(self, input_path):
        note_items, tempo_items = read_items(input_path)
        note_items = quantize_items(note_items)
        max_time = note_items[-1].end
        if 'chord' in self.checkpoint_path:
            chord_items = extract_chords(note_items)
            items = chord_items + tempo_items + note_items
        else:
            items = tempo_items + note_items
        groups = group_items(items, max_time)
        events = item2event(groups)
        return events
    def prepare_data(self, midi_paths):
         # extract events
        all_events = []
        for path in midi_paths:
            events = self.extract_events(path)
            all_events.append(events)
        # event to word
        all_words = []
        for events in all_events:
            words = []
            for event in events:
                e = '{}_{}'.format(event.name, event.value)
                if e in self.event2word:
                    words.append(self.event2word[e])
                else:
                    # OOV
                    if event.name == 'Note Velocity':
                        # replace with max velocity based on our training data
                        words.append(self.event2word['Note Velocity_21'])
                    else:
                        # something is wrong
                        # you should handle it for your own purpose
                        print('something is wrong! {}'.format(e))
            all_words.append(words)
        # to training data
        segments = []
        for words in all_words:
            pairs = []
            for i in range(0, len(words)-self.x_len-1, self.x_len):
                x = words[i:i+self.x_len]
                y = words[i+1:i+self.x_len+1]
                pairs.append([x, y])
            pairs = np.array(pairs)
            # abandon the last
            for i in np.arange(0, len(pairs)-self.group_size, self.group_size*2):
                data = pairs[i:i+self.group_size]
                if len(data) == self.group_size:
                    segments.append(data)
        segments = np.array(segments)
        return segments

# read notes and tempo changes from midi (assume there is only one track)
def read_items(file_path):
    midi_obj = miditoolkit.midi.parser.MidiFile(file_path)
    # note
    note_items = []
    notes = midi_obj.instruments[0].notes
    notes.sort(key=lambda x: (x.start, x.pitch))
    for note in notes:
        note_items.append(Item(
            name='Note', 
            start=note.start, 
            end=note.end, 
            velocity=note.velocity, 
            pitch=note.pitch))
    note_items.sort(key=lambda x: x.start)
    # tempo
    tempo_items = []
    for tempo in midi_obj.tempo_changes:
        tempo_items.append(Item(
            name='Tempo',
            start=tempo.time,
            end=None,
            velocity=None,
            pitch=int(tempo.tempo)))
    tempo_items.sort(key=lambda x: x.start)
    # expand to all beat
    max_tick = tempo_items[-1].start
    existing_ticks = {item.start: item.pitch for item in tempo_items}
    wanted_ticks = np.arange(0, max_tick+1, DEFAULT_RESOLUTION)
    output = []
    for tick in wanted_ticks:
        if tick in existing_ticks:
            output.append(Item(
                name='Tempo',
                start=tick,
                end=None,
                velocity=None,
                pitch=existing_ticks[tick]))
        else:
            output.append(Item(
                name='Tempo',
                start=tick,
                end=None,
                velocity=None,
                pitch=output[-1].pitch))
    tempo_items = output
    return note_items, tempo_items

# quantize items
def quantize_items(items, ticks=120):
    # grid
    grids = np.arange(0, items[-1].start, ticks, dtype=int)
    # process
    for item in items:
        index = np.argmin(abs(grids - item.start))
        shift = grids[index] - item.start
        item.start += shift
        item.end += shift
    return items      

# extract chord
def extract_chords(items):
    method = chord_recognition.MIDIChord()
    chords = method.extract(notes=items)
    output = []
    for chord in chords:
        output.append(Item(
            name='Chord',
            start=chord[0],
            end=chord[1],
            velocity=None,
            pitch=chord[2].split('/')[0]))
    return output

# group items
def group_items(items, max_time, ticks_per_bar=DEFAULT_RESOLUTION*4):
    items.sort(key=lambda x: x.start)
    downbeats = np.arange(0, max_time+ticks_per_bar, ticks_per_bar)
    groups = []
    for db1, db2 in zip(downbeats[:-1], downbeats[1:]):
        insiders = []
        for item in items:
            if (item.start >= db1) and (item.start < db2):
                insiders.append(item)
        overall = [db1] + insiders + [db2]
        groups.append(overall)
    return groups

# item to event
def item2event(groups):
    events = []
    n_downbeat = 0
    for i in range(len(groups)):
        if 'Note' not in [item.name for item in groups[i][1:-1]]:
            continue
        bar_st, bar_et = groups[i][0], groups[i][-1]
        n_downbeat += 1
        events.append(Event(
            name='Bar',
            time=None, 
            value=None,
            text='{}'.format(n_downbeat)))
        for item in groups[i][1:-1]:
            # position
            flags = np.linspace(bar_st, bar_et, DEFAULT_FRACTION, endpoint=False)
            index = np.argmin(abs(flags-item.start))
            events.append(Event(
                name='Position', 
                time=item.start,
                value='{}/{}'.format(index+1, DEFAULT_FRACTION),
                text='{}'.format(item.start)))
            if item.name == 'Note':
                # velocity
                velocity_index = np.searchsorted(
                    DEFAULT_VELOCITY_BINS, 
                    item.velocity, 
                    side='right') - 1
                events.append(Event(
                    name='Note Velocity',
                    time=item.start, 
                    value=velocity_index,
                    text='{}/{}'.format(item.velocity, DEFAULT_VELOCITY_BINS[velocity_index])))
                # pitch
                events.append(Event(
                    name='Note On',
                    time=item.start, 
                    value=item.pitch,
                    text='{}'.format(item.pitch)))
                # duration
                duration = item.end - item.start
                index = np.argmin(abs(DEFAULT_DURATION_BINS-duration))
                events.append(Event(
                    name='Note Duration',
                    time=item.start,
                    value=index,
                    text='{}/{}'.format(duration, DEFAULT_DURATION_BINS[index])))
            elif item.name == 'Chord':
                events.append(Event(
                    name='Chord', 
                    time=item.start,
                    value=item.pitch,
                    text='{}'.format(item.pitch)))
            elif item.name == 'Tempo':
                tempo = item.pitch
                if tempo in DEFAULT_TEMPO_INTERVALS[0]:
                    tempo_style = Event('Tempo Class', item.start, 'slow', None)
                    tempo_value = Event('Tempo Value', item.start, 
                        tempo-DEFAULT_TEMPO_INTERVALS[0].start, None)
                elif tempo in DEFAULT_TEMPO_INTERVALS[1]:
                    tempo_style = Event('Tempo Class', item.start, 'mid', None)
                    tempo_value = Event('Tempo Value', item.start, 
                        tempo-DEFAULT_TEMPO_INTERVALS[1].start, None)
                elif tempo in DEFAULT_TEMPO_INTERVALS[2]:
                    tempo_style = Event('Tempo Class', item.start, 'fast', None)
                    tempo_value = Event('Tempo Value', item.start, 
                        tempo-DEFAULT_TEMPO_INTERVALS[2].start, None)
                elif tempo < DEFAULT_TEMPO_INTERVALS[0].start:
                    tempo_style = Event('Tempo Class', item.start, 'slow', None)
                    tempo_value = Event('Tempo Value', item.start, 0, None)
                elif tempo > DEFAULT_TEMPO_INTERVALS[2].stop:
                    tempo_style = Event('Tempo Class', item.start, 'fast', None)
                    tempo_value = Event('Tempo Value', item.start, 59, None)
                events.append(tempo_style)
                events.append(tempo_value)     
    return events

# define "Item" for general storage
class Item(object):
    def __init__(self, name, start, end, velocity, pitch):
        self.name = name
        self.start = start
        self.end = end
        self.velocity = velocity
        self.pitch = pitch

    def __repr__(self):
        return 'Item(name={}, start={}, end={}, velocity={}, pitch={})'.format(
            self.name, self.start, self.end, self.velocity, self.pitch)

class Event(object):
    def __init__(self, name, time, value, text):
        self.name = name
        self.time = time
        self.value = value
        self.text = text

    def __repr__(self):
        return 'Event(name={}, time={}, value={}, text={})'.format(
            self.name, self.time, self.value, self.text)