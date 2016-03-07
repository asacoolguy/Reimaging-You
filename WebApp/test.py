#!/usr/bin/python

print 'Content-type: text/html\n'

@app.route("/reimagine")
def main():
    return render_template(
        'index.html',
        title = 'yay word clouds!',
        pic = url_for('generateWordCloud'))

