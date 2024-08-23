import tqdm
from indonlu.utils.data_utils import DocumentSentimentDataset, DocumentSentimentDataLoader
from transformers import BertForSequenceClassification, BertTokenizer,BertConfig
import torch.nn as nn
import torch
import pandas as pd

from indonlu.utils.forward_fn import forward_sequence_classification

tokenizer = BertTokenizer.from_pretrained('indobenchmark/indobert-large-p2', do_lower_case=True)
config = BertConfig.from_pretrained('indobenchmark/indobert-large-p2')
config.num_labels = DocumentSentimentDataset.NUM_LABELS
w2i, i2w = DocumentSentimentDataset.LABEL2INDEX, DocumentSentimentDataset.INDEX2LABEL

class BertForSequenceClassificationWithDropout(BertForSequenceClassification):
    def __init__(self, config, dropout_prob=0.5):
        super().__init__(config)
        self.dropout = nn.Dropout(dropout_prob)  # Tambahkan dropout dengan probabilitas 0.3

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, labels=None):
        outputs = self.bert(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        pooled_output = outputs[1]
        pooled_output = self.dropout(pooled_output)  # Terapkan dropout pada output pooling
        logits = self.classifier(pooled_output)

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        return loss, logits


device = torch.device('cpu')
model = BertForSequenceClassificationWithDropout.from_pretrained(
    "indobenchmark/indobert-large-p2",
    config = config
)
model.load_state_dict(torch.load("model/model.pth",map_location=device))
 

def toCsv(text):
    df = pd.DataFrame({'category': [text], 'sentiment': ["neutral"]})
    df.to_csv("input.csv", sep='\t', header=None, index=False)
    
def predict(text):
    toCsv(text)
    
    result = []
    model.eval()
    torch.set_grad_enabled(False)
    data = DocumentSentimentDataset("test.csv",tokenizer, lowercase=True)
    loader = DocumentSentimentDataLoader(dataset=data, max_seq_len=512, batch_size=32, num_workers=0, shuffle=False)
    pbar = tqdm.tqdm(loader, leave=True, total=len(loader))
    for i, batch_data in enumerate(pbar):
        loss, batch_hyp, batch_label = forward_sequence_classification(model, batch_data[:-1], i2w=i2w)
        result = batch_hyp
    
    if len(result) == 0:
        return False
    
    return result[0]


if __name__ == "__main__":
    res = predict("tidak perlu ada pemilihan lah biar presiden tentukan jawatan lagipon korang mana boleh lawan master scheme dari thing unless kalau korang ada kawan dengan yang cukup jumlah lah")
    print(res)
    
