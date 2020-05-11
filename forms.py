from flask_wtf import FlaskForm
from wtforms import StringField,SubmitField

class RatingForm(FlaskForm):
    review = StringField('Review')
    predict = SubmitField('Predict')