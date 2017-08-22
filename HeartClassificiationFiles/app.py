from flask import Flask, send_file, request
from flask import jsonify

from gaSearch import *

app = Flask(__name__)

@app.route("/")
def index():
    return send_file("templates/index.html")

@app.route('/sendUser', methods=['POST'])
def signUp():
    my_user=[]
    data = request.json['user']
    user=data.get('user')
   
    my_user.append(float(user.get('age')))
    if (user.get('gender') == 'male'):
        my_user.append(1)
    else:
        my_user.append(0)
        
        
    if (user.get('cp') == 'asymptomatic'):
        my_user.append(0)
   
    if (user.get('cp') == 'typical'):
        my_user.append(3)
        
    if (user.get('cp') == 'non-anginal'):
        my_user.append(2)
    
    if (user.get('cp')== 'atypical'):
        my_user.append(1)
    my_user.append(user.get('trestbps'))
    my_user.append(user.get('chol'))
    if (user.get('fbs') == 'true'):
        my_user.append(1)
    else:
        my_user.append(0)
        
        
        
    if (user.get('restecg') == 'hypertrophy'):
        my_user.append(0)
   
    if (user.get('restecg') == 'stt'):
        my_user.append(2)
        
    if (user.get('restecg') == 'normal'):
        my_user.append(1)
    my_user.append(user.get('thalach'))
    
    if (user.get('exang') == 'yes'):
        my_user.append(1)
    else:
        my_user.append(0)
        
    my_user.append(user.get('oldpeak'))
    
    if (user.get('slope') == 'upsloping'):
        my_user.append(2)
   
    if (user.get('slope') == 'downsloping'):
        my_user.append(0)
        
    if (user.get('slope') == 'flat'):
        my_user.append(1)
    my_user.append(user.get('ca'))
    
    
    if (user.get('thal') == 'fixed'):
        my_user.append(0)
   
    if (user.get('thal') == 'normal'):
        my_user.append(1)
        
    if (user.get('thal') == 'reversable'):
        my_user.append(2)
    print(my_user)
    predict_new_data(my_user,3,1)
   
    
    return "";
	
if __name__ == "__main__":
    app.run(host='0.0.0.0')