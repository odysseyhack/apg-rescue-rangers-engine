from flask import Flask
from flask import jsonify
from flask_cors import CORS, cross_origin
import historicalTwitter
from flask import request

app = Flask(__name__)

cors = CORS(app)
# app.config['CORS_HEADERS'] = 'Access-Control-Allow-Origin'
app.config['CORS_HEADERS'] = 'Content-Type'

@app.route("/topicmodels")
@cross_origin()
def sendTopicModels():
    return jsonify(historicalTwitter.getTexts())

@app.route("/images")
@cross_origin()
def sendImages():
	return jsonify(historicalTwitter.getImages())

@app.route("/sendtags")
@cross_origin()
def printRandomSent():
	tag = request.args.get('tag')
	return jsonify(tag)

if __name__ == '__main__':
     app.run(port=5003)