from flask import Flask, render_template

COLOR_RED = '#FF0000'
COLOR_BLUE = '#0000FF'
COLOR_GREEN = '#1A7E55'
COLOR_PURPLE = '#6617B5'


app = Flask(__name__)

@app.route("/")
def main():
	words = [  #some words to visualize
        { 
            'word': 'this', 
            'size': 55,
            'color': COLOR_RED,
            'font': '\'Indie Flower\', cursive',
            'angle': '45'
        },
        { 
            'word': 'Test', 
            'size': 73,
            'color': COLOR_BLUE,
            'font': '\'Open Sans\', sans-serif',
            'angle': '-30'
        },
        { 
            'word': 'kinDA', 
            'size': 153,
            'color': COLOR_GREEN,
            'font': '\'Indie Flower\', cursive',
            'angle': '-150'
        },
        { 
            'word': 'WERKS', 
            'size': 33,
            'color': COLOR_PURPLE,
            'font': '\'Open Sans\', sans-serif',
            'angle': '90'
        }
    ]
	return render_template('index.html',
		title = 'Word Clouds',
		words = words)

if __name__ == "__main__":
	app.run()
