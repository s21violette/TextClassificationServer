from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch


# Модель сохраняется в файл
# tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-multilingual-cased")
# model = AutoModelForSequenceClassification.from_pretrained("google-bert/bert-base-multilingual-cased")
#
# tokenizer.save_pretrained('tokenizer')
# model.save_pretrained('model')


class SentimentClassificator:
    __tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-multilingual-cased")
    __model = AutoModelForSequenceClassification.from_pretrained("google-bert/bert-base-multilingual-cased")
    # __tokenizer = AutoTokenizer.from_pretrained("tokenizer")
    # __model = AutoModelForSequenceClassification.from_pretrained("model")

    def predict_sentiment(self, txt):
        answers = ["Negative sentiment", "Positive sentiment"]
        encoded_input = self.__tokenizer(txt, return_tensors='pt')
        output = self.__model(**encoded_input)
        scores = output.logits.detach()
        predicted_class = torch.argmax(scores)

        return answers[predicted_class]


# import ssl
#
# from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
# # !pip install vader-multi
#
#
# def predict_sentiment(txt):
#     ssl._create_default_https_context = ssl._create_unverified_context
#
#     analyzer = SentimentIntensityAnalyzer()
#
#     res = analyzer.polarity_scores(txt)
#     # print(res)
#     # print(max(res, key=res.get))
#     return max(res, key=res.get)
