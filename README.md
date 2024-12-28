# Laporan Proyek Machine Learning - Muhammad Dava Pasha

## Domain Proyek

Industri wine merupakan salah satu sektor yang memiliki nilai ekonomi tinggi dalam industri minuman global. Kualitas wine menjadi faktor krusial yang mempengaruhi harga jual dan kepuasan konsumen. Penentuan kualitas wine secara tradisional dilakukan melalui evaluasi ahli sommeliers yang menilai berbagai karakteristik seperti aroma, rasa, warna, dan tekstur.

Pengembangan model prediksi kualitas wine menggunakan machine learning menjadi solusi yang menjanjikan untuk mengatasi tantangan dalam industri wine modern. Dengan mengkombinasikan data historis penilaian ahli dan parameter fisikokimia, model dapat "belajar" pola yang menentukan kualitas wine dan membuat prediksi yang akurat untuk sampel baru.

Proyek ini bertujuan untuk mengembangkan sistem prediksi kualitas wine yang dapat memberikan penilaian objektif dan konsisten, sambil tetap mempertahankan standar kualitas tinggi yang diharapkan oleh industri dan konsumen wine.

**Rubrik/Kriteria Tambahan (Opsional)**:
- Dikarenakannya susah menentukan suatu kualitas dari sebuah wine, menggunakan machine learning dapat membantu menentukan kualitasnya.
- [Measuring Wine Quality and Typicity](https://www.mdpi.com/2306-5710/9/2/41)
  
## Business Understanding

Secara umum kita bisa menentukan kualitas wine dengan parameter sebagai berikut:
1. Keseimbangan Keasaman
    - Fixed acidity dan pH harus berada dalam rentang optimal (pH sekitar 3-4).
    - Volatile acidity sebaiknya rendah (umumnya < 0.7 g/L) karena jika terlalu tinggi akan memberikan rasa cuka.
    - Citric acid membantu memberikan kesegaran, tapi dalam jumlah seimbang
2. Kandungan Gula dan Alkohol
    - Residual sugar menentukan tingkat kemanisan.
    - Alcohol content biasanya lebih tinggi (11-14%) pada wine berkualitas baik.
    - Density wine berkualitas baik biasanya lebih rendah karena kandungan alkohol yang lebih tinggi.
3. Preservasi dan Stabilitas
    - Free sulfur dioxide dan total sulfur dioxide harus cukup untuk mengawetkan wine (biasanya 25-50 mg/L untuk free SO2).
    - Sulphates membantu menjaga stabilitas dan mencegah oksidasi.
    - Namun kadar terlalu tinggi dapat merusak rasa.
4. Mineral dan Rasa
    - Chlorides sebaiknya rendah karena mempengaruhi rasa asin.
    - Keseimbangan antara semua komponen sangat penting.

Faktor-faktor yang menandakan wine berkualitas baik:
- Keseimbangan yang baik antara asam, gula, dan alkohol
- Volatile acidity rendah
- Kadar SO2 yang cukup tapi tidak berlebihan
- pH yang sesuai (3-3.6)
- Kandungan alkohol yang proporsional

Dengan kita mengetahui cara umum kita untuk menentukan kualitas wine maka kita bisa menggunakan machine learning untuk dapat mempersingkat dan membuat proses menentukan kualitas wine dengan lebih cepat.

### Problem Statements

Menjelaskan pernyataan masalah latar belakang:
- Bagaimana cara kita menentukan kualitas wine?

### Goals
- Menentukan kualitas wine dengan objektif, cepat, optimal dan akurat.

**Rubrik/Kriteria Tambahan (Opsional)**:
### Solution statements
  - Menentukan kualitas wine dengan machine learning kita bisa menggunakan library scikit-learn dan kita bisa menggunakan algoritma Random Forest Regressor.
  - Menentukan kualitas wine dengan machine learning kita bisa menggunakan algoritma lain di scikit-learn yaitu LinearRegression.

## Data Understanding
Data yang saya gunakan diambil dari Kaggle yaitu [WineQuality](https://www.kaggle.com/datasets/uciml/red-wine-quality-cortez-et-al-2009)

Pada data tersebut memiliki 12 kolom dan 1599 baris dan tidak memiliki data yang kosong atau missing values dan data tersebut tidak memiliki outlier dengan begitu data dipastikan data bersih dan bisa langsung kita gunakan.


### Variabel-variabel pada WineQuality dataset adalah sebagai berikut:
- fixed acidity : Keasaman tetap dalam wine, terkait dengan asam tartarat.
- volatile acidity : Keasaman yang mudah menguap, terutama asam asetat. Jika terlalu tinggi dapat menyebabkan rasa asam yang tidak diinginkan.
- citric acid : Asam sitrat yang terdapat dalam wine atau rasa segar pada wine.
- residual sugar : Jumlah gula yang tersisa setelah fermentasi berhenti. Mempengaruhi tingkat kemanisan wine.
- chlorides : Kandungan garam dalam wine.  Mempengaruhi rasa asin wine.
- free sulfur dioxide : antimikroba dan antioksidan dalam wine.
- total sulfur dioxide : Total SO2 dalam wine (bebas + terikat). Jumlah tinggi dapat mempengaruhi aroma dan rasa yang tidak diinginkan.
- density : Kepadatan wine, terkait dengan kandungan alkohol dan gula.
- pH : Tingkat keasaman wine pada skala 0-14
- sulphates : Aditif wine yang berkontribusi pada SO2 dan berfungsi sebagai antimikroba dan antioksidan.
- alcohol : Persentase kandungan alkohol dalam wine.
- quality : Kualitas wine



**Rubrik/Kriteria Tambahan (Opsional)**:
### Dari dataset tersebut kita mendapatkan berbagai karakteristik sebagai berikut
- Fixed acidity: Nilai berkisaran dari 7.1 sampai 7.3.
- Volatile acidity: Nilai berkisaran dari 0.53 sampai 1.07.
- Citric acid: Nilai relatif rendah dengan kisaran 0.06 sampai 0.09.
- Residual sugar: Nilai berkisaran dari 1.7 sampai 2.0.
- Chlorides: Nilai berkisaran dari 0.071 sampai 0.178.
- Free sulfur dioxide and total sulfur dioxide: Nilai ini umumnya lebih tinggi dibandingkan anggur putih, dengan kisaran masing-masing 8.0 hingga 15.0 dan 24.0 hingga 89.0.
- Density: Nilai berkisaran dari 0.9951 to 0.9962.
- pH: Nilai berkisaran dari 3.29 to 3.67.
- Sulphates: Nilai berkisaran dari 0.66 to 0.73.
- Alcohol: Nilai berkisaran dari 9.0 sampai 10.8.
- Semua variable memiliki efek masing masing terhadap kualitas wine yang akan kita tentukan nanti.

## Data Preparation
Pada data preparation kita akan menggunakan library scikit-learn untuk membagi data train dan test yaitu menggunakan fungsi `train_test_split` dan disini kita akan membagi 75% training data dan 25% sebagai test data dengan kita menggunakan kolom selain `quality` sebagai data train nya dan kolom `quality` sebagai data testnya.

**Rubrik/Kriteria Tambahan (Opsional)**: 
- Menggunakan library pandas pada Python untuk memuat dataset yang sudah kita unduh.
- Selanjutnya kita akan menggunakan kolom selain `quality` sebagai data trainnya serta kolom `quality` sebagai data testnya.
- Memanggil fungsi `train_test_split` untuk membagi data train dan data test dengan perbandingan 75% data train dan 25% data test.
- Mengapa menggunakan pandas? karena pandas adalah library yang sangat mudah digunakan dan banyak dokumentasinya.


## Modeling

### Random Forest Regressor
- Kita akan menggunakan class RandomForestRegressor dengan parameter `random_state` sebagai `100` dan sisanya menggunakan default parameter dengan kita menggunakan parameter `random_state` kita akan mendapatkan hasil yang selalu sama.
- Selanjutnya kita akan melakukan Model Improvement yaitu menggunakan teknik GridSearchCV dengan parameter sebagai berikut, parameter pertama ialah class RandomForestRegressor dengan `random_state` sebagai `45` selanjutnya, `param_grid` yaitu adalah parameter yang akan dicari oleh GridSearchCV yang disini kita akan mendefinisikan `n_estimators` jumlah pohon 100, 200 dan 300 selanjutnya `max_depth` yang berarti maksimal kedalaman pohon disini kita akan mencoba None, 10, 20, dan 30 selanjutnya `min_samples_split` yaitu minimal sampel untuk membagi sebuah cabang dimana kita akan mencoba 2, 5 dan 10 dan yang terakhir yaitu `min_samples_leaf` dimana kita akan mencoba 1, 2 dan 4.
- Dari hasil GridSearchCV kita mendapatkan hasil yang terbaik ialah `max_depth` adalah `None`, `min_samples_leaf` adalah `1`, `min_samples_split` adalah `2` dan yang terakhir `n_estimators` adalah `3`.

### Linear Regression
- Kita akan menggunakan class LinearRegression dengan default parameternya.
- Selanjutnya kita masukan data trainingnya.
- Terakhir kita akan coba memprediksi menggunakan data testnya.

### Cara Kerja Random Forest Regressor dan Linear Regression
1. Random Forest Regressor
    - Merupakan ensemble learning yang terdiri dari banyak Decision Tree
    - Bekerja dengan membuat multiple decision tree yang dilatih pada sampel data berbeda (bootstrap sampling)
    - Setiap tree memberikan prediksi, kemudian hasil akhirnya adalah rata-rata dari semua prediksi tree
2. Linear Regression
    - Ini adalah algoritma sederhana yang mencari hubungan linear antara variabel input (X)dan output (y)
    - Bekerja dengan mencari garis lurus terbaik yang meminimalkan jarak error antara titik data sebenarnya dengan prediksi
    - Menggunakan formula: y = mx + b, dimana m adalah slope dan b adalah intercept Cocok untuk data yang memiliki hubungan linear dan sederhana

**Rubrik/Kriteria Tambahan (Opsional)**: 
- Dari hasil evaluasi Random Forest Regressor dan Linear Regression kita simpulkan bahwa algoritma Random Forest Regressor lebih unggul dari Linear Regression karena Random Forest Regressor sendiri memilik keunggulan sendiri seperti berbentuk Pohon yang berisi berbagai peraturan oleh sebab itu ini menjadi unggulan sendiri.
- Dalam menggunakan Random Forest Regressor saya menggunakan default parameter yang disediakan oleh scikit-learn dan menghasilkan 0.6293 dalam metrik RMSE dan selanjutnya saya mencoba menggunakan teknik GridSearchCV dan mendapatkan hasil yang kurang dari menggunakan default parameter.
- Kesimpulan akhir, Random Forest Regressor lebih unggul dari pada Linear Regression oleh sebab itu algoritma Random Forest Regressor yang terbaik dalam kasus ini.

## Evaluation
Pada kasus ini yaitu regression kita menggunakan metrik evaluasi Mean Absolute Error (MAE), Mean Squared Error (MSE), Root Mean Squared Error (RMSE) dan R-squared (R²)

Penjelasan dari metrik tersebut:
- MAE adalah rata-rata dari kesalahan dengan nilai absolut antara nilai sebenarnya dan nilai prediksi.
- MSE adalah nilai rata-rata dari kuadrat kesalahan antara nilai sebenarnya dan nilai prediksi. MSE dapat digambarkan dengan rumus matematika seperti berikut.
- RMSE adalah akar kuadrat dari MSE sehingga lebih mudah untuk diinterpretasikan.
- R-squared mengevaluasi seberapa baik model regresi linear

Hasil Evaluasi pada Random Forest Regressor:
- R-Squared adalah `0.3830`
- MSE adalah `0.3960`
- RMSE adalah `0.6293`
- MAE adalah `0.4443`

Hasil Evaluasi pada Random Forest Regressor dengan hasil dari GridSearchCV:
- R-Squared adalah `0.3793`
- RMSE adalah `0.6312`
- MAE adalah `0.4452`

Hasil Evaluasi pada Linear Regression dengan:
- R-Squared adalah `0.2567`
- MSE adalah `0.4770`
- RMSE adalah `0.6907`
- MAE adalah `0.5344`

Kesimpulan Akhir: 
- Dari hasil evaluasi Random Forest Regressor dan Linear Regression kita simpulkan bahwa algoritma Random Forest Regressor lebih unggul dari Linear Regression karena Random Forest Regressor sendiri memilik keunggulan sendiri seperti berbentuk Pohon yang berisi berbagai peraturan oleh sebab itu ini menjadi unggulan sendiri.
- Random Forest Regressor lebih unggul dari pada Linear Regression oleh sebab itu algoritma Random Forest Regressor yang terbaik dalam kasus ini.


**Rubrik/Kriteria Tambahan (Opsional)**: 
- Mean Absolute Error (MAE), MAE adalah nilai rata-rata dari kesalahan dengan nilai absolut antara nilai sebenarnya dan nilai prediksi.
- Root Mean Squared Error (RMSE), RMSE adalah akar kuadrat dari MSE.
- R-squared (R²), R-squared adalah ukuran yang menunjukkan seberapa baik model regresi kita dapat menjelaskan variasi dalam data. Nilai ini dihitung dengan membandingkan dua jenis variasi: variasi total yang ada dalam data asli, dan variasi yang berhasil dijelaskan oleh model regresi yang kita buat.