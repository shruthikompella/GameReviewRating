# GameReviewRating
* Deploying a  Flask Application on heroku
* heroku login
* start by cloning the application.
* heroku create your-first-heroku-app --buildpack heroku/python
* Adding a Procfile
* In order for us to successfully deploy any application to Heroku, we must add a Procfile to that application.
* Before we can add a Procfile, we need to first install a web server called Gunicorn. Run the following command within the application folder.
 * pip install gunicorn
* Update the requirements file by running
*  pip freeze > requirements.txt
* Create a new file with Procfile as the name and do not add any extension. Add this line below
 * web: gunicorn app:app
* Stage the files for a commit and commit them
 * git add .
 * git commit -m "First commit for heroku"
* Push the changes from your local master branch to heroku’s master branch.
 * git push heroku master
* Now fire up your browser and got to: ( remember to replace your-first-heroku-app with the name of the app you created.)
* https://your-first-heroku-app.herokuapp.com



