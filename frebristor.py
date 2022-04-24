from urllib import response
from wsgiref.util import request_uri
from flask import Flask, redirect, request, session, render_template, url_for, flash
from flask_restful import Resource, Api
from flask_cors import CORS
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from numpy import asarray
from pandas import DataFrame
from scipy.spatial import distance
import random
import mysql.connector
from mysql.connector import cursor
from datetime import datetime
# import scipy
# import sklearn
# import seaborn as sns
# import matplotlib

frebristor = Flask(__name__)
frebristor.config["SECRET_KEY"] = "skripsiSecret"

frebristorApi= Api(frebristor)

CORS (frebristor) 

mydb = mysql.connector.connect(
    host="localhost",
    user="root",
    passwd="",
    database="frebristor",
)
mycursor = mydb.cursor()

class OlahData(Resource):
    def loaddata():
        df = pd.read_csv('data.csv', delimiter=";")
        df = df.drop(['No RM'], axis=1)
        return df
    
    def kolom_object(df):
        object_col = []
        for col in df.columns:
            if df[col].dtype == "object":
                object_col.append(col)
        return object_col
        
    def nama_kolom(df):
        name_col = []
        for i in df.columns:
            name_col.append(i)
        return name_col
    
    def labelendcoder(df, kolom):
        data = df.copy()
        encoder = LabelEncoder()
        for col in kolom:
            data[col] = encoder.fit_transform(data[col])
        return data
    
    def normalisasi(df):
        datas = df.copy()
        name_col = OlahData.nama_kolom(datas)
        data = asarray(datas)
        scaler=MinMaxScaler()
        trscaled = scaler.fit_transform(data)
        dataset = DataFrame(trscaled, columns = name_col)
        return dataset
    
    def setdataset(df):
        tf = df.copy()
        dhf = df.copy()
        
        dhf.pop('TF')
        dhf.rename(columns={'DHF':'diag'}, inplace=True)
        
        tf.pop('DHF')
        tf.rename(columns={'TF':'diag'}, inplace=True)
        
        return dhf, tf

class PseudoNN(Resource):
    def separate_class(dataset,value):
        t = dataset.copy()
        t = t[t['diag'] == value]
        # print("separate "+str(value))
        # print(t)
        t.pop('diag')
        return t

    def get_neighbours(train, test_row, num_neighbours):
        row_distances = list()
        distances = []
        for i in range(0,len(train)):
            train_row = train.iloc[i][:]
            train_row, test_row = asarray(train_row), asarray(test_row)
            train_row, test_row = train_row.reshape(-1,1), test_row.reshape(-1,1)
            dist = distance.euclidean(test_row, train_row)
            distances.append(dist)
            row_distances.append((train_row, dist))
        distances = asarray(distances)
        distances.sort()
        neighbours = np.zeros(num_neighbours)
        # print("neighbour "+str(value))
        # print(train)
        # print("distances")
        # print(distances)
        # print("neighbours")
        # print(neighbours)
        for i in range(0,num_neighbours):
            neighbours[i] = distances[i]
        return neighbours

    def sum_neighbours(neighbours):
        sum = 0
        for i in range(0,len(neighbours)):
            temp = neighbours[i] * (1/(i+1)) 
            sum = sum + temp
        return sum

    def predict_classification(train, test_row ):
        Xt = PseudoNN.separate_class(train,0)
        Xy = PseudoNN.separate_class(train,1)
        yNeighbours = PseudoNN.get_neighbours(Xy, test_row, 7)
        tNeighbours = PseudoNN.get_neighbours(Xt, test_row, 7)
        sumY = PseudoNN.sum_neighbours(yNeighbours)
        sumT = PseudoNN.sum_neighbours(tNeighbours)
        if sumY < sumT:
            prediction = 1
        elif sumT < sumY:
            prediction = 0
        else:
            prediction = random.randint(0,1)
        return prediction

class AksiDatabase(Resource):
    def hasilprediksi(tf, dhf):
        if tf == 1 and dhf == 1:
            return 'TF dan DHF'
        elif tf == 1 and dhf == 0:
            return 'TF'
        elif tf == 0 and dhf == 1:
            return 'DHF'
        else:
            return 'Penyakit Lain'
        
    # ambil gejala px yg = 1
    def splitkata(gejala):
        gejalaPx = gejala.copy()
        gejalaPx.pop('Nafsu Makan')
        gejalaPx.pop('Lama Demam')
        gejalaPx.pop('TF')
        gejalaPx.pop('DHF')
        gejalaPx.pop('Kes')
        gjl = []
        for col in gejalaPx.columns :
            if gejalaPx.loc[0][col] == '1':
                gjl.append(col)
        return gjl
    
    #buat array gejala yg dialami px (gejala = 1) --> nama kolom sudah dispasi
    def daftargejala(barisdata):
        argejala = AksiDatabase.splitkata(barisdata)
        simpulan = []
        for g in range(0,len(argejala)):
            data = argejala[g]
            dx = data.split(" ")
            if len(dx) > 1:
                kata = dx[0]+" "+  dx[1]
                simpulan.append(kata)
            else:
                kata = data
                simpulan.append(kata)
        return simpulan

@frebristor.route("/")
def index():
    if "name" in session:
        return render_template("index.html")
    else:
        return redirect(url_for('login'))

@frebristor.route("/prediksi", methods=["POST", "GET"])
def prediksi():
    if "name" in session:
        if request.method == "POST":
            noRM = request.form['rm']
            lahir = request.form['lahir']
            lamaDemam = request.form['demam']
            suhuTubuh = request.form['suhu']
            kesadaran = request.form['kes']
            sembelit = request.form['radioSembelit']
            nafsuMakan = request.form['radioMaMi']
            diare = request.form['radioDiare']
            pusing = request.form['radioPusing']
            perutKembung = request.form['radioKembung']
            mual = request.form['radioMual']
            nyeriKepala = request.form['radioNKepala']
            muntah = request.form['radioMuntah']
            nyeriOtot = request.form['radioNOtot']
            batuk = request.form['radioBatuk']
            nyeriSendi = request.form['radioNSendi']
            mimisan = request.form['radioMimisan']
            nyeriRetroOrbital = request.form['radioNRetro']
            lidahKotor = request.form['radioLidah']
            ruamKulit = request.form['radioRuam']
            now = datetime.now()
            tglPeriksa = now.strftime('%Y-%m-%d %H:%M:%S')  
            
            gejala = np.array([
                [lamaDemam, suhuTubuh, kesadaran, nafsuMakan, pusing, mual, muntah, batuk, sembelit, diare, perutKembung, 
                mimisan, nyeriKepala, nyeriOtot, nyeriSendi, nyeriRetroOrbital, ruamKulit, lidahKotor, 0, 0]     
            ])
            
            df = OlahData.loaddata()
            name_col = OlahData.nama_kolom(df)
            object_col = OlahData.kolom_object(df)
            
            pxtodb = DataFrame(gejala)
            pxtodb.columns = name_col
            # px = asarray(gejala)
            # px = px.reshape(20,1)
            # name_col = asarray(name_col)
            # pasien = DataFrame(gejala, columns=[name_col])
            # df = df.append(pasien)
            data = np.vstack((df, gejala))
            data = DataFrame(data)
            data.columns = name_col
            
            dataset = OlahData.labelendcoder(data, object_col)
            # dataset = OlahData.normalisasi(dff)
            df.info()
            df.nunique()
            # print("df")
            # print(df)
            # print(object_col)
            # print("label")
            # print(dataset)
            
            gejalaPx = dataset.tail(1)
            gejalaPx.pop('TF')
            gejalaPx.pop('DHF')
            dataset = dataset.drop(dataset.tail(1).index)
            dhf, tf = OlahData.setdataset(dataset)
            # print("tf")
            # print(tf)
            # print("dhf")
            # print(dhf)
            
            pred_tf = PseudoNN.predict_classification(tf, gejalaPx)
            pred_dhf = PseudoNN.predict_classification(dhf, gejalaPx)
            prediksi = AksiDatabase.hasilprediksi(pred_tf, pred_dhf)
            gejalanya = AksiDatabase.daftargejala(pxtodb)
            
            val = (tglPeriksa, noRM, lahir, lamaDemam, suhuTubuh, kesadaran, nafsuMakan, pusing, mual, muntah, batuk, sembelit, diare, perutKembung, 
                mimisan, nyeriKepala, nyeriOtot, nyeriSendi, nyeriRetroOrbital, ruamKulit, lidahKotor, pred_tf, pred_dhf,(", ").join(gejalanya),prediksi)
            # sql = "INSERT INTO riwayat(tglPeriksa, RM, ttl, lama_demam, suhu, kes, nafsu_makan, pusing, mual, muntah, batuk, sembelit, diare, perut_kembung, mimisan, nyeri_kepala, nyeri_otot,nyeri_sendi, nyeri_retro_orbital, ruam_kulit, lidah_kotor, dhf, tf) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)" %val
            mycursor.execute("""
                INSERT INTO 
                    riwayat_detail(tglPeriksa, no_RM, ttl, lama_demam, suhu, kes, nafsu_makan, pusing, mual, muntah, batuk, sembelit, diare, perut_kembung, mimisan, nyeri_kepala, nyeri_otot,nyeri_sendi, nyeri_retro_orbital, ruam_kulit, lidah_kotor, tf, dhf, gejala, prediksi) 
                VALUES 
                    (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)"""
                    , val)
            mydb.commit()

            mycursor.execute("SELECT * FROM riwayat_detail")
            res = mycursor.fetchall()
            lastrow = res[-1][:]
            id = lastrow[0]
            
            # sql = "UPDATE `riwayat` SET `gejala`="+(", ").join(gejalanya)+",`prediksi`="+prediksi+" WHERE `id_detail` ="+id+" "
            # mycursor.execute(sql)
            # mydb.commit()
            
            return redirect(url_for('predPx',id=id))
        else:
            return render_template('prediksi.html')
    else:
        return redirect(url_for('index'))

@frebristor.route("/riwayat")
def riwayat():
    if "name" in session:
        mycursor.execute("SELECT * FROM riwayat_detail") 
        res = mycursor.fetchall()
        return render_template("riwayat.html", riwayat=res)
    else:
        return redirect(url_for('index'))

@frebristor.route("/logout")
def logout():
    if "name" in session:
        session.pop("name")
        return redirect(url_for('index'))
    else:
        return redirect(url_for('index'))

@frebristor.route("/login", methods=["GET", "POST"])
def login():
    if "name" in session:
        return redirect(url_for('index'))
    else:
        if request.method == "POST":
            uname = request.form['username']
            pword = request.form['password']
            role = "petugas"
            mycursor.execute("select * from user")
            res = mycursor.fetchall()
            for row in res:
                if uname == row[0]:
                    if row[1] == pword  and row[2] == role:
                        name = row[3]
                        session['name'] = name
                        return redirect(url_for('index') )
                    else:
                        flash('Password yang Anda masukkan salah. Silahkan masukkan ulang password!', 'danger')
                        return render_template("login.html")
                else:
                    flash('Username yang Anda masukkan tidak terdaftar.', 'warning')
                    return render_template("login.html")
        return render_template("login.html")


@frebristor.route("/hasilprediksi")
def predPx():
    if "name" in session:
        id = request.args.get("id")
        mycursor.execute("select * from riwayat_detail where id_detail = " +id+"")
        res = mycursor.fetchall()

        currentDate = datetime.today().date()
        
        age = currentDate.year - res[0][3].year
        monthVeri = currentDate.month - res[0][3].month
        dateVeri = currentDate.day - res[0][3].day

        #Type conversion here
        age = int(age)
        monthVeri = int(monthVeri)
        dateVeri = int(dateVeri)

        # some decisions
        if monthVeri < 0 :
            age = age-1
        elif dateVeri < 0 and monthVeri == 0:
            age = age-1
        
        waktu, gejala  = str(res[0][1]), str(res[0][-2])
        waktu = waktu.split()
        if len(gejala) != 0:
            gejala = gejala.split(",")
        else:
            gejala = 0
        return render_template('hasil.html', hasil=res, waktu=waktu, gejala=gejala, usia=age)
    else:
        return redirect(url_for('index'))
    
@frebristor.route("/kembali")
def kembali():
    if "name" in session:
        return redirect(url_for('prediksi'))
    else:
        return redirect(url_for('index'))
    
# @frebristor.route('/hapus')
# def hapus():
#     if "name" in session:
#         id = request.args.get("id")
#         confirm = request.args.get("confirm")
#         if confirm == 'confirmed':
#             mycursor.execute("DELETE FROM `riwayat_detail` WHERE `id_detail` = " +id+"")
#             mydb.commit()
#             return redirect(url_for('riwayat'))
#         else:
#             norm = mycursor.execute("select RM from `riwayat_detail` WHERE `id_detail` = "+id+"")
#             flash("Apakah Anda akan menghapus hasil prediksi pasien "+str(norm)+"?", 'warning')
#             return redirect(url_for('riwayat'))
#     else:
#         return redirect(url_for('index'))

# route admin =============================================================================================
    
@frebristor.route("/admin/login", methods=["GET", "POST"])
def loginadmin():
    if "role" in session:
        return redirect(url_for('dashboard'))
    else:
        if request.method == "POST":
            user = request.form['username']
            passw = request.form['password']
            role = "admin"
            mycursor.execute("select * from user where role = 'admin'")
            res = mycursor.fetchall()
            # print("role: "+str(res))
            for row in res:
                if user == row[0]:
                    if row[1] == passw  and row[2] == role:
                        name = row[3]
                        session['nama'] = name
                        session['role'] = role
                        return redirect(url_for('dashboard') )
                    else:
                        flash('Password yang Anda masukkan salah. Silahkan masukkan ulang password!', 'danger')
                        return redirect(url_for('loginadmin'))
                else:
                    flash('Username yang Anda masukkan tidak terdaftar.', 'warning')
                    return redirect(url_for('loginadmin'))
        return render_template("admin-login.html")

@frebristor.route("/admin")
def dashboard():
    if "role" in session:
        mycursor.execute("SELECT * FROM riwayat_detail") 
        res = mycursor.fetchall()
        return render_template("admin-dashboard.html", riwayat=res)
    else:
        return redirect(url_for('loginadmin'))

@frebristor.route('/admin/petugas/tambah', methods=["GET", "POST"])
def tambah():
    if request.method == "POST":
        uname = request.form["username"]
        fullname = request.form["fullname"]
        role = request.form["roleForm"]
        pword = request.form["password"]
        confirm = request.form["confirm"]
        
        if pword != confirm:
            flash('Password yang Anda masukkan berbeda', 'warning')
            return redirect(url_for('tambah'))
        else:
            val = (uname, pword, role, fullname )
            mycursor.execute("""
                            INSERT INTO 
                                `user`(`username`, `password`, `role`, `nama`)
                            VALUES 
                                (%s, %s,%s,%s)
                            
                            """, val)
            mydb.commit()
            return redirect(url_for('petugas'))
    return render_template("admin-tambah.html")
    
@frebristor.route('/admin/petugas',methods=["GET", "POST"])
def petugas():
    mycursor.execute("SELECT * FROM user") 
    res = mycursor.fetchall()
    index = enumerate(res)
    return render_template("admin-petugas.html", listuser = index)

@frebristor.route('/admin/petugas/hapus')
def hapus():
    if "role" in session:
        username = request.args.get("uname")
        mycursor.execute("DELETE FROM `user` WHERE `username` = " +username+"")
        mydb.commit()
        return redirect(url_for('petugas'))
    else:
        return redirect(url_for('dashboard'))

@frebristor.route("/admin/logout")
def adminlogout():
    if "role" in session:
        session.pop("role")
        session.pop("nama")
        return redirect(url_for('dashboard'))
    else:
        return redirect(url_for('dashboard'))

if __name__ == "__main__":
    frebristor.run(debug=True)