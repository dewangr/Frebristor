from xmlrpc.client import Boolean, boolean
from flask import Flask, redirect, request, session, render_template, url_for
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
        df = df.drop(['NoRM', 'TglLahir','Demam', 'SkalaNyeri', 'Lokasi', 'Ket'], axis=1)
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
        datas = df
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
        yNeighbours = PseudoNN.get_neighbours(Xy, test_row, 3)
        tNeighbours = PseudoNN.get_neighbours(Xt, test_row, 3)
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
        gejalaPx.pop('Nafsu_Makan')
        gejalaPx.pop('Lama_Demam')
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
            dx = data.split("_")
            if len(dx) > 1:
                kata = dx[0]+" "+  dx[1]
                simpulan.append(kata)
            else:
                kata = data
                simpulan.append(kata)
        return simpulan

@frebristor.route("/")
def index():
    return render_template("index.html")

@frebristor.route("/prediksi", methods=["POST", "GET"])
def prediksi():
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
        
        dff = OlahData.labelendcoder(data, object_col)
        dataset = OlahData.normalisasi(dff)
        
        gejalaPx = dataset.tail(1)
        gejalaPx.pop('TF')
        gejalaPx.pop('DHF')
        dataset = dataset.drop(dataset.tail(1).index)
        dhf, tf = OlahData.setdataset(dataset)
        
        pred_tf = PseudoNN.predict_classification(tf, gejalaPx)
        pred_dhf = PseudoNN.predict_classification(dhf, gejalaPx)
        prediksi = AksiDatabase.hasilprediksi(pred_tf, pred_dhf)
        gejalanya = AksiDatabase.daftargejala(pxtodb)
        
        val = (tglPeriksa, noRM, lahir, lamaDemam, suhuTubuh, kesadaran, nafsuMakan, pusing, mual, muntah, batuk, sembelit, diare, perutKembung, 
            mimisan, nyeriKepala, nyeriOtot, nyeriSendi, nyeriRetroOrbital, ruamKulit, lidahKotor, pred_tf, pred_dhf,(", ").join(gejalanya),prediksi)
        # sql = "INSERT INTO riwayat(tglPeriksa, RM, ttl, lama_demam, suhu, kes, nafsu_makan, pusing, mual, muntah, batuk, sembelit, diare, perut_kembung, mimisan, nyeri_kepala, nyeri_otot,nyeri_sendi, nyeri_retro_orbital, ruam_kulit, lidah_kotor, dhf, tf) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)" %val
        mycursor.execute("""
            INSERT INTO 
                detail_riwayat(tglPeriksa, RM, ttl, lama_demam, suhu, kes, nafsu_makan, pusing, mual, muntah, batuk, sembelit, diare, perut_kembung, mimisan, nyeri_kepala, nyeri_otot,nyeri_sendi, nyeri_retro_orbital, ruam_kulit, lidah_kotor, tf, dhf,gejala,prediksi) 
            VALUES 
                (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)"""
                , val)
        mydb.commit()

        mycursor.execute("SELECT * FROM detail_riwayat")
        res = mycursor.fetchall()
        lastrow = res[-1][:]
        id = lastrow[0]
        
        # sql = "UPDATE `riwayat` SET `gejala`="+(", ").join(gejalanya)+",`prediksi`="+prediksi+" WHERE `id_detail` ="+id+" "
        # mycursor.execute(sql)
        # mydb.commit()
        
        return redirect(url_for('predPx',id=id))
    else:
        return render_template('prediksi.html')

@frebristor.route("/riwayat")
def riwayat():
    mycursor.execute("SELECT * FROM detail_riwayat") 
    res = mycursor.fetchall()
    no = 1
    # mycursor.execute("SELECT * FROM detail_riwayat")
    # result = mycursor.fetchall()
    
    return render_template("riwayat.html", riwayat=res)

@frebristor.route("/logout")
def logout():
    session.pop("username")
    return redirect(url_for('index'))

@frebristor.route("/login")
def login():
    return render_template("login.html")

@frebristor.route("/hasilprediksi")
def predPx():
    id = request.args.get("id")
    mycursor.execute("select * from detail_riwayat where id_detail = " +id+"")
    res = mycursor.fetchall()
    waktu, gejala  = str(res[0][1]), str(res[0][-2])
    waktu = waktu.split()
    gejala = gejala.split(",")
    return render_template('hasil.html', hasil=res, waktu=waktu, gejala=gejala)
    
@frebristor.route("/kembali")
def kembali():
    return redirect(url_for('prediksi'))

if __name__ == "__main__":
    frebristor.run(debug=True)