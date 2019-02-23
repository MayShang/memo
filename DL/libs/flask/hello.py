import flask 
from flask import redirect, url_for, request, render_template, make_response
from werkzeug import secure_filename

app = flask.Flask(__name__)

@app.route('/')
def index():
    return "hello there!"

@app.route('/demo')
def demo():
    return "here is demo"

@app.route('/postdir/<int:postid>')
def postdir(postid):
    return "this is flask %d" % postid

@app.route('/admin')
def hello_admin():
    return "hello admin"

@app.route('/welcome/<name>')
def welcome(name):
    return "hello %s" % name

@app.route('/login/<name>')
def login(name):
    if name == 'admin':
        return redirect(url_for('hello_admin'))
    else:
        return redirect(url_for('welcome', name=name))

#################################################################
# form demo
#################################################################

@app.route('/form_act', methods = ['POST', 'GET'])
def form_act():
    if request.method == "POST":
        user = request.form["ctx"]
        return redirect(url_for('welcome', name=user))
    else:
        user = request.args.get('ctx')
        return redirect(url_for('welcome', name=user))

@app.route('/form_demo')
def form_demo():
    return '''
<html>
<body>
    <form action="/form_act" method="get">
    <p> please enter something </p>
    <input type="text" name="ctx"/>
    <input type="submit" value="submit"/>
  </form>
</body>
</html>
'''

#################################################################
# render demo
# the `variable` part of url insert at {{user}} place holder
# {%  %} for statements
# {{  }} for variables
# {#  #} for comments
#################################################################

@app.route('/render_demo/<name>/<int:mark>')
def render_demo_x(name, mark):
    return render_template("hello.html", user = name, score=mark)

@app.route('/render_demo/<int:mark>')
def render_demo(mark):
    return render_template("hello.html", score=mark)

@app.route('/js_demo')
def js_demo():
    return render_template("js_demo.html")


#################################################################
# send form data to template demo
# student.html collect form data 
# result() function to get request form data and render
# result.html
#################################################################
@app.route("/student")
def student_demo():
    return render_template("student.html")

@app.route('/result', methods = ['POST', 'GET'])
def result():
    if request.method == "POST":
        usr_data = request.form
        return render_template("result.html", data = usr_data)

#################################################################
# cookie demo
# student.html form2 collect form data 
# save_cookie() function to save cookie
# get_cookie() return cookie
#################################################################
@app.route("/save_cookie", methods = ['POST', 'GET'])
def save_cookie():
    if request.method == "POST":
        usr_data = request.form
        # rep = make_response(render_template("result.html", data=usr_data))
        rep = make_response()
        rep.set_cookie("usrId", usr_data['usr_id'])
        rep.set_cookie("math", usr_data['math'])

    return "<h1> working fine </h1>"

@app.route('/get_cookie')
def get_cookie():
    name = request.cookies.get('usrId')
    math = request.cookies.get('math')
    return "<h3> welcome " + name + " math is " + math + '</h3>'

#################################################################
# upload file demo
# student.html form2 collect file
# upload_file() function to operate file
#################################################################
@app.route("/upload_file", methods = ['POST', 'GET'])
def upload_file():
    if request.method == "POST":
        f = request.files['file']
        f.save(secure_filename(f.filename))
        return "<h3> uploaded </h3>"

if __name__ == '__main__':
    app.run(host='0:0:0:0', port=int(5000), debug=True, use_reload=False)
