
from flask import Flask,render_template,url_for,flash,redirect
from forms import RatingForm
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle

app = Flask(__name__)
app.config["DEBUG"] = True
app.config['SECRET_KEY']='10ec2857c5dfee847c6f84a6af7c2058'
rating = '67'

@app.route('/', methods=['POST', 'GET'])
def home():
    form = RatingForm()
    if form.validate_on_submit():
        review = form.review.data
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
