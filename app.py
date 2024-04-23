from flask import Flask, render_template, request
import joblib,os
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
data_train = pd.read_excel("Book1.xlsx")


vectorizer = TfidfVectorizer()
train_vectors = vectorizer.fit_transform(data_train['Gejala'])


app = Flask(__name__)
app.config["TEMPLATES_AUTO_RELOAD"] = True

@app.route('/')
def index():
    data_gejala = pd.DataFrame(data_train)
    data_gejala = data_gejala.drop(['No', 'Penyakit'], axis=1)
    data_gejala.drop_duplicates(inplace=True, keep='first', subset=['Gejala'])
    data_gejala.reset_index(drop=True, inplace=True)
    gejala = data_gejala['Gejala'].str.title()

    return render_template('index.html', array=gejala, len = len(gejala))

@app.route('/hasil', methods = ["POST"])
def predict():
    model = load("gigi_clf.pkl")
    data_gejala = pd.DataFrame(data_train)
    data_gejala = data_gejala.drop('Penyakit', axis=1)
    data_gejala.drop_duplicates(inplace=True, keep='first', subset=['Gejala'])
    data_gejala.reset_index(drop=True, inplace=True)
    gejala = data_gejala['Gejala'].str.title()

    array = request.form.getlist('check')
    jml_array = len(array)
    if (jml_array > 0):
        teks = ', '.join([str(elem) for elem in array])
        teks_vector = vectorizer.transform([teks])
        prediction = model.predict(teks_vector)


        # deskripsi
        data = pd.read_excel(io="/Test/deskripsi.xlsx")
        df = pd.DataFrame(data)
        nilai_yang_dicari = prediction[0]
        nilai_kolom_kedua = None
        for index, value in df['Penyakit'].items():
            if value == nilai_yang_dicari:
                nilai_kolom_kedua = df.at[index, 'Deskripsi']
                break
            
        return render_template("index.html", hasil_prediksi = prediction[0], deskripsi = nilai_kolom_kedua, array=gejala, len = len(gejala))
    else:
        return render_template("index.html", error = "Tidak ada masukan")
        

def load(file):
    load = joblib.load(open(os.path.join(file), "rb"))
    return load


if __name__ == '__main__':
    app.run(debug=True)