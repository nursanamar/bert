from torch.utils.data import Dataset;
import numpy as np

class SingleSentimentDataset(Dataset):
    LABEL2INDEX = {'positive': 0, 'neutral': 1, 'negative': 2}
    INDEX2LABEL = {0: 'positive', 1: 'neutral', 2: 'negative'}
    NUM_LABELS = 3
    
    def __init__(self, data, tokenizer, no_special_token=False, *args, **kwargs):
        self.data = data
        self.tokenizer = tokenizer
        self.no_special_token = no_special_token
    
    def __getitem__(self, index):
        text, sentiment = self.data['text'], self.data['sentiment']
        subwords = self.tokenizer.encode(text, add_special_tokens=not self.no_special_token)
        return np.array(subwords), np.array(sentiment), self.data['text']
    
    def __len__(self):
        return 1  