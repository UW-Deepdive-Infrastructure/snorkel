from collections import defaultdict
from snorkel.models import construct_stable_id
from snorkel.parser import Parser, ParserConnection
import re

class Gdd(Parser):
    '''
    '''
    def __init__(self, annotators=['tagger', 'parser', 'entity'],
                 lang='en', num_threads=1, verbose=False):
        super(Gdd, self).__init__(name="GDD")

    def connect(self):
        return ParserConnection(self)

    def clean_row(self, row):
        temp = re.sub('[{}"]', '', row)
        temp = re.sub(r',,,', ',|;|,', temp)
        split = temp.split(",")
        cleaned_split = [i.replace("|;|", ",") for i in split]
        return cleaned_split

    def parse(self, document, text):
        '''
        (docid, sentid, wordidx, words, poses, ners, lemmas, dep_paths, dep_parents)
        '''
        abs_char_count = 0
        position = 0
        for sent in text:
            parts = defaultdict(list)
            (docid, sentid, wordidx, words, poses, ners, lemmas, dep_paths, dep_parents) = sent

            parts['words'] = self.clean_row(words)
            parts['lemmas'] = self.clean_row(lemmas)
            parts['pos_tags'] = self.clean_row(poses)
            parts['ner_tags'] = self.clean_row(ners)

            parts["char_offsets"] = [0 for i in range(len(parts["words"]))]
            parts["abs_char_offsets"] = [0 for i in range(len(parts["words"]))]
            sentence_char_count = 0
            for wordidx in range(len(parts["words"])):
                parts["char_offsets"][wordidx] = sentence_char_count
                parts["abs_char_offsets"][wordidx] = abs_char_count
                sentence_char_count += len(parts["words"][wordidx]) + 1
                abs_char_count += len(parts["words"][wordidx]) + 1

            parts['dep_parents'] = [int(i) for i in self.clean_row(dep_parents)]
            parts['dep_labels'] = self.clean_row(dep_paths)

            # Add null entity array (matching null for CoreNLP)
            parts['entity_cids'] = ['O' for _ in parts['words']]
            parts['entity_types'] = ['O' for _ in parts['words']]

            # Link the sentence to its parent document object
            parts['document'] = document
            parts['text'] = " ".join(parts['words'])

            parts['position'] = position

            # Assign the stable id as document's stable id plus absolute
            # character offset
            abs_sent_offset = parts['abs_char_offsets'][0]
            abs_sent_offset_end = abs_sent_offset + parts['char_offsets'][-1] + len(parts['words'][-1])

            if document:
                parts['stable_id'] = construct_stable_id(document, 'sentence', abs_sent_offset, abs_sent_offset_end)

            position += 1

            yield parts
