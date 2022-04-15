from flask import Flask, render_template, request, session
from pip import main

frebristorapp = Flask(__name__)

@frebristorapp.route("/")
def index():
    return render_template("index.html")

@frebristorapp.route("/prediksi")
def prediksi():
    return render_template("prediksi.html")

@frebristorapp.route("/riwayat")
def riwayat():
    return render_template("riwayat.html")

@frebristorapp.route("/logout")
def logout():
    session.pop("username")
    return "Logout berhasil"

@frebristorapp.route("/login")
def LogIn():
    return render_template("login.html")

if __name__ == "__main__":
    frebristorapp.run(debug=True)