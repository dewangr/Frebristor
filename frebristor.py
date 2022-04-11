from flask import Flask, render_template, request, session
from pip import main

frebristor = Flask(__name__)

@frebristor.route("/")
def index():
    return render_template("index.html")

@frebristor.route("/prediksi")
def prediksi():
    return render_template("prediksi.html")

@frebristor.route("/riwayat")
def riwayat():
    return render_template("riwayat.html")

@frebristor.route("/logout")
def logout():
    session.pop("username")
    return "Logout berhasil"

@frebristor.route("/login")
def LogIn():
    return render_template("login.html")

if __name__ == "__main__":
    frebristor.run(debug=True)