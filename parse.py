from tqdm import tqdm
import pandas as pd

file = open('cooking.stackexchange.valid.txt')

labels = set()
samples = []

# parse file to find labels and store data in a raw list
for line in file:
    sample = {'labels': [], 'text': ''}
    tokens = line.split()
    for i, token in enumerate(tokens):
        if token.startswith('__label__'):
            # the true label is after the characters '__label__'
            label = token[9:]
            labels.add(label)
            sample['labels'].append(label)
        else:
            # we have reached the end of label list and must now parse the text
            text = ' '.join(tokens[i:])
            sample['text'] = text
            break
    samples.append(sample)

# format data list into a data frame whose columns are text + all labels
data = pd.DataFrame(columns=['text'] + list(labels))
for sample in tqdm(samples):
    data_dict  = {label: 1 for label in sample['labels']}
    data_dict['text'] = sample['text']
    data = data.append(data_dict, ignore_index=True)

data = data.fillna(0)
print(data.head())
