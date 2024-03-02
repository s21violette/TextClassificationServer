from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-multilingual-cased")
model = AutoModelForSequenceClassification.from_pretrained("google-bert/bert-base-multilingual-cased")


def predict_sentiment(txt):
    encoded_input = tokenizer(txt, return_tensors='pt')
    output = model(**encoded_input)
    scores = output.logits.detach()
    predicted_class = torch.argmax(scores)
    return predicted_class


text = "You are an asshole"
sentiment = predict_sentiment(text)
answers = ["Negative sentiment", "Positive sentiment"]
print(answers[sentiment])
# import ssl
#
# from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
# # !pip install vader-multi
#
# ssl._create_default_https_context = ssl._create_unverified_context
#
# analyzer = SentimentIntensityAnalyzer()
# text = "ты очень умный"
#
# res = analyzer.polarity_scores(text)
# print(res)
# print(max(res, key=res.get))
