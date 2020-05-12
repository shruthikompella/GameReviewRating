
from flask import Flask,render_template,url_for,flash,redirect
from forms import RatingForm
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
from nltk import word_tokenize
from nltk.stem import PorterStemmer
import re
import numpy as np
import pickle

app = Flask(__name__)
app.config["DEBUG"] = True
app.config['SECRET_KEY']='10ec2857c5dfee847c6f84a6af7c2058'
rating = '67'

@app.route('/', methods=['POST', 'GET'])
def home():
    form = RatingForm()
    # def clean_text(text):
    #     text = text.lower()
    #     clean = re.compile('<.*?>')
    #     text = re.sub(clean,'',text)
    #     text = re.sub(r"""[,.;@#?!&$]+\ *"""," ",text, flags=re.VERBOSE)
    #     text = text.replace('\r\n', '-').replace('\n', ' ')
    #     check = text.replace(" ", "").isalpha()
    #     if check:
    #         text = [word for word in word_tokenize(text)]
    #     else:
    #         text = [word for word in word_tokenize(text) if word.isalpha()]        
    # return text
    # def stem(text):
    #     text = ' '.join([porter.stem(word) for word in text.split()])
    # return text
    if form.validate_on_submit():
        review = form.review.data
        #review = clean_text(review)
        review = review.lower()
        clean = re.compile('<.*?>')
        review = re.sub(clean,'',review)
        review = re.sub(r"""[,.;@#?!&$]+\ *"""," ",review, flags=re.VERBOSE)
        review = review.replace('\r\n', '-').replace('\n', ' ')
        review = [word for word in word_tokenize(review) if word.isalpha()]
        STOPWORDS= set(["a","about","above","after","again","against","am","an","and","any","are","as","at","be","been","before","being","below","between","both","by","can","could","did","do","does","doing","down","during","each","few","for","from","further","had","has","have","having","he","he'd","he'll","he's","her","here","here's","hers","herself","him","himself","his","how","how's","i","i'd","i'll","i'm","i've","in","into","is","isn't","it","it's","its","itself","let's","me","more","myself","of","off","on","once","only","or","other","ought","our","ours" ,"ourselves","out","over","own","same","she","she'd","she'll","she's","should","so","some","such","than","that","that's","the","their","theirs","them","hemselves","then","there","there's","these","they","they'd","they'll","they're","they've","this","those","through","to","too","under","until","up","very","was","we","we'd","we'll","we're","we've","were","what","what's","when","when's","where","where's","which","while","who","who's","whom","why","why's","with","would","you","you'd","you'll","you're","you've","your","yours","yourself","yourselves","return","hes","heres","hows","im","its","lets","shes","thats","theres","theyll","theyre","theyve","were","whats","whens","wheres","whos","whys","youd","youll","youre","youve"])
        STOPWORDS_dict = dict.fromkeys(STOPWORDS, 0)
        review = [word for word in review if word not in STOPWORDS_dict]
        review = ' '.join(word for word in review)
        porter=PorterStemmer()
        review = ' '.join([porter.stem(word) for word in review.split()])
        vocab = open('models/Vocab.pickle', 'rb')      
        vocab_pickled = pickle.load(vocab) 
        vectorizer = TfidfVectorizer(vocabulary=vocab_pickled.vocabulary_)
        model = open('models/LinearRegression.pickle', 'rb')      
        linearregression_pickled = pickle.load(model) 
        p = vectorizer.fit_transform([review])
        rate = round(linearregression_pickled.predict(p)[0],2)
        flash(f'YOUR REVIEW : {review}','success')
        flash(f'MY MODEL RATING : { rate }','success')
        return redirect(url_for('predict'))
    return render_template('PredictReviewRating.html',form=form)


@app.route('/predict', methods=['POST', 'GET'])
def predict():   
    return render_template('Rating.html',title='Rating',posts=rating)



if __name__ == '__main__':
    app.run(debug=True)
