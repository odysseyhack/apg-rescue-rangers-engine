from flask import Flask
from flask import jsonify
from flask_cors import CORS, cross_origin
import historicalTwitter
from flask import request

app = Flask(__name__)

cors = CORS(app)
# app.config['CORS_HEADERS'] = 'Access-Control-Allow-Origin'
app.config['CORS_HEADERS'] = 'Content-Type'

QuestionIndicator = False
AnswerIndicator = False

QList = "" 
AList = 0

@app.route("/getQuestion")
@cross_origin()
def getQuestion():
	global QuestionIndicator
	global QList
	QuestionIndicator = False
	return jsonify({'results':QList})

@app.route("/getAnswer")
@cross_origin()
def getAnswer():
	global AnswerIndicator
	global AList
	AnswerIndicator = False
	return jsonify({'results':AList})

@app.route("/updateQuestion")
@cross_origin()
def updateQuestion():
	global QList
	QList = request.args.get('QList')
	return jsonify({'results':"success"})

@app.route("/updataAnswer")
@cross_origin()
def updataAnswer():
	global AList
	AList =  request.args.get('AList') 
	return  jsonify({'results':"success"})

@app.route("/getQIndicator")    
@cross_origin()
def getQIndicator():
	return  jsonify({'results':QuestionIndicator})

@app.route("/getAIndicator")
@cross_origin()
def getAIndicator():
	return  jsonify({'results':AnswerIndicator})


@app.route("/sendtags")
@cross_origin()
def printRandomSent():
	tag = request.args.get('tag')
	return jsonify(historicalTwitter.retrievelInfo(tag))

if __name__ == '__main__':
     app.run(port=5003)