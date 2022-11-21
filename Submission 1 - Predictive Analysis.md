# Laporan Proyek Machine Learning - Muhammad Imron
## Used Cars Price Prediction (Regression)

Industri mobil bekas merupakan industri yang terus meningkat beberapa tahun ini. Hal tersebut ditandai dengan munculnya berbagai portal mobil bekas online antara lain CarDheko, Quikr, Carwale dan Cars24 [1]. Keakuratan standar harga mobil bekas sangat penting bagi para pelaku dalam industri mobil bekas. Harga prediksi mobil yang akurat membutuhkan pakar ahli karena biasanya harga mobil ditentukan berdasarkan berbagai faktor. Beberapa faktor tersebut antara lain merek, model, usia, tipe bensin, tipe mesin, dan *mileage* [2]. Keefisienan dalam menentukan standar harga juga dibutuhkan oleh para pelaku industri demi mendapatkan profit maksimal dengan mengeluarkan biaya seminimum mungkin. Salah satu cara meningkatkan keefisienan dalam menentukan standar harga mobil bekas adalah melakukan automasisasi prediksi harga mobil seperti yang ditunjukkan oleh Venkatasubbu *et al* [1] dan Gegic *et al* [2]. Tujuan utama dari proyek ini adalah meneliti performa algoritma *machine learning* dalam mengatasi masalah automasisasi prediksi harga mobil bekas. Dataset yang digunakan dalam proyek ini adalah dataset yang dibuat oleh Aleksandr Glotov [3]. Dataset tersebut berisi data mengenai mobil bekas yang dikumpulkan dari berbagai situs penjualan online mobil bekas terkenal di Polandia.

## Business Understandings

### *Problem Statements*
Berdasarkan latar belakang yang telah diuraikan sebelumnya, proyek ini akan mengembangkan sebuah sistem prediksi harga diamonds untuk menjawab permasalahan berikut.
- Dari serangkaian fitur yang ada, fitur apa yang paling berpengaruh terhadap harga mobil bekas?
- Berapa harga pasar mobil bekas dengan karakteristik atau fitur tertentu?

### *Goals*
Untuk  menjawab pertanyaan tersebut, proyek ini akan membuat predictive modelling dengan tujuan atau goals sebagai berikut:
- Mengetahui fitur yang paling berkorelasi dengan harga mobil bekas.
- Membuat model *machine learning* yang dapat memprediksi harga mobil bekas seakurat mungkin berdasarkan fitur-fitur yang ada.

### *Solution Statements*
Untuk  meraih tujuan tersebut, proyek ini akan mengimplementasikan hal berikut:
- Mengimplementasikan *Exploratory Data Analysis* (EDA) untuk mengetahui fitur yang paling berkorelasi dengan harga mobil bekas.
- Membandingkan performa tiga algoritma *machine learning* yaitu *K-Nearest Neighbor*, *Random Forest*, dan *AdaBoost* untuk memprediksi harga mobil bekas seakurat mungkin berdasarkan fitur-fitur yang ada.
- Menggunakan matriks *Mean Squared Error* sebagai matriks evaluasi.

## Data Understandings

Dataset yang digunakan dalam proyek ini adalah dataset yang diambil dari website [kaggle]. Dataset tersebut berisi data mengenai mobil bekas yang dikumpulkan dari berbagai situs penjualan online mobil bekas terkenal di Polandia pada bulan Januari 2022. Dataset tersebut berisi informasi tentang merek, model, generasi, tahun produksi, jarak tempuh, jenis dan volume mesin, lokalisasi dan harga. Penjelasan lengkap mengenai variabel dalam dataset dapat dilihat pada list di bawah.

**Variabel-variabel pada dataset _Car Prices Prediction_ adalah sebagai berikut:**
*   Unnamed: 0: kolom kosong dari dataset (sudah default dari dataset asli)
*   mark: jenis dari mobil
*   model: model dari mobil
*   generation_name: generasi mobil
*   year: tahun mobil diproduksi
*   mileage: jarak tempuh mobil sejak diproduksi
*   vol_engine: berat mesin mobil
*   fuel: tipe bensin yang digunakan mobil
*   city: kota mobil diproduksi
*   province: provinsi mobil diproduksi
*   price: harga dari mobil (target)

### *Exploratory Data Analysis* (EDA)
Pada proyek ini, terdapat 4 langkah EDA yang dilakukan. Keempat langkah tersebut adalah sebagai berikut:
1. **Deskripsi variabel**
Pada langkah ini, dataset dijabarkan dengan menggunakan fungsi *info()* dan *describe()*. 
![info_1.jpg](https://drive.google.com/uc?export=view&id=1mR6oSIjUMdQeK8jUU9RezqXj327gjKNp "Dataset Info")
Gambar 1. Informasi dataset (awal).
Berdasarkan gambar 1 diatas, dapat diambil informasi bahwa dataset memiliki 6 kolom bertipe *object* antara lain *mark*, *model*, *generation_name*, *fuel*, *city*, dan *province*. Sisa 5 kolom lain bertipe numerik yaitu *Unnamed: 0*, *year*, *mileage*, *vol_engine*, dan *price*.
![describe_1.jpg](https://drive.google.com/uc?export=view&id=1goBxdYOouZb-OjMyxVceifO8OPbXcJc6 "Dataset Describe")
Gambar 2. Informasi statistik kolom numerik (awal).
Berdasarkan gambar 2 di atas, dapat diambil informasi bahwa kolom *Unnamed: 0*, *mileage*, dan *vol_engine* memiliki nilai minimal 0, hal tersebut tidak wajar dan akan ditangani pada tahap *data cleaning*.
2. **_Data cleaning_**
Pada tahap ini, dilakukan berbagai langkah untuk membersihkan data kotor pada dataset. Langkah-langkah tersebut adalah sebagai berikut.
    - Langkah pertama adalah menghapus kolom *Unnamed: 0*, karena kolom tersebut hanyalah kolom kosong, jadi tidak diperlukan. Hasil dataset setelah kolom tersebut dihapus dapat dilihat pada gambar 3.
    ![remove_unnamed.jpg](https://drive.google.com/uc?export=view&id=13Pfk3GM5y6hqM-D725HkwAtyLVdegwZf "remove unnamed")
    Gambar 3. Dataset setelah *Unnamed: 0* dihapus.
    ![generation_name_missing_value.jpg](https://drive.google.com/uc?export=view&id=1MFZxYVXVpmkpCp77ACwlAP2qkuFVbpKH "generation name missing value")
    Gambar 4. Jumlah *missing value* setiap kolom
    - Berdasarkan gambar 4, nilai *missing value* pada kolom *generation_name* sangat banyak, yaitu melebihi 25% dari total jumlah dataset. Daripada menangani *missing value* sebanyak itu hanya pada satu kolom, lebih baik kolom tersebut dihapus. Berdasarkan penjelasan tersebut, maka langkah selanjutnya adalah menghapus kolom *generation_name*.
    - Seperti yang telah disebutkan dalam tahap deskripsi variabel, langkah selanjutnya adalah menangani nilai 0 pada kolom *mileage* dan *vol_engine*. 
    ![mileage_size.jpg](https://drive.google.com/uc?export=view&id=1bg836HUwvxC28POgTqJYYaUn056UAAMt "mileage_size")
    Gambar 5. Ukuran *mileage* yang mengandung nilai nol
    ![vol_engine_size.jpg](https://drive.google.com/uc?export=view&id=1cqQk9pdlaAvkLa9L6tjdTE0WY1ELG6fE "vol_engine_size")
    Gambar 6. Ukuran *vol_engine* yang mengandung nilai nol
    Berdasarkan gambar 5 dan 6, dapat disimpulkan bahwa jumlah baris yang mengandung nilai 0 dari kolom *mileage* dan *vol_engine* sangat sedikit, yaitu sekitar 0,01% dari jumlah dataset sehingga cara yang tepat untuk mengatasi masalah tersebut adalah dengan cara menghapus baris yang mengandung nilai 0 dari kolom *mileage* dan *vol_engine*.
    - Langkah selanjutnya adalah menghapus nilai duplikat pada dataset. Penghapusan nilai duplikat ini sangat penting agar saat pelatihan model memiliki data yang berbeda dan tidak akan menambah *weight* dari data yang sama. Proses menghapus data yang duplikat dapat dilakukan dengan memanggil fungsi *drop_duplicates()* seperti yang terlihat pada gambar 7. Parameter *inplace* yang digunakan pada fungsi tersebut akan membuat hasil *dataframe* dari fungsi memiliki indeks baru.
    ![duplicate.jpg](https://drive.google.com/uc?export=view&id=1gAx-WPf_1BDbP3DVjte_-vU9V9LTs9Zu "duplicate")
    Gambar 7. Proses penghapusan nilai duplikat.
    - Langkah selanjutnya adalah menangani *outlier*. Pada proyek ini, metode yang digunakan untuk menangani outlier adalah *Inter Quartile Range* atau disingkat IQR. IQR bekerja dengan cara membuat batas bawah dan batas atas yang diambil dari nilai kuartil 1 dan 3 pada kolom target. Jika ada suatu nilai yang berada di luar batas bawah dan batas atas, maka nilai tersebut akan dikenal sebagai *outlier* oleh IQR dan akan dihapus. Proses IQR dapat dilihat pada gambar 8.
    ![outlier.jpg](https://drive.google.com/uc?export=view&id=1MFCXvJC7yErfi5fmOlgE-7gOM_Sx2C6j "outlier")
    Gambar 8. Algoritma IQR.
    Hasil dataset setelah penanganan *outlier* dapat dilihat pada gambar 9.
    ![shape_after_outlier.jpg](https://drive.google.com/uc?export=view&id=17D7QnCNZfV3eBefeDz4sC_FE-06CDr3W "shape_after_outlier")
    Gambar 9. *Shape* dataset setelah penanganan *outlier*.
    - Langkah selanjutnya adalah mengubah tipe data kolom numerik selain kolom target, yaitu kolom *price*. Hal ini dilakukan untuk mempermudah proses perhitungan yang terjadi pada *data preprocessing*. Proses perubahan tipe data kolom numerik selain kolom *price* dapat dilihat pada gambar 10.
    ![change_type.jpg](https://drive.google.com/uc?export=view&id=10-yOEpKjfc9MBHtmEmtM_Dxj3HijwNKS "change_type")
    Gambar 10. Perubahan tipe data pada kolom *year*, *mileage*, dan *vol_engine*.
    - Langkah terakhir pada tahap *data cleaning* adalah mengecek kembali deskripsi variabel dataset. Pengecekan dilakukan sama seperti tahap deskripsi variabel, yaitu dengan cara memanggil fungsi *info()* dan *describe()*. Hasil pemanggilan fungsi tersebut dapat dilihat pada gambar 11 dan 12.
    ![info_after_clean.jpg](https://drive.google.com/uc?export=view&id=1FS-tE4TGccR1wzYvAlSfIg5QCnr7v1Qq "info_after_clean")
    Gambar 11. Info dataset setelah *data cleaning*.
    ![describe_after_clean.jpg](https://drive.google.com/uc?export=view&id=1f1SnJEtn0iAhoBrkcsZSSn3fyPywhOWo "describe_after_clean")
    Gambar 12. deskripsi statistik dataset setelah *data cleaning*.
    Berdasarkan gambar 11 dan 12, maka dapat disimpulkan bahwa dataset sekarang sudah 'bersih' sehingga tahap selanjutnya adalah analisis pada setiap kolom yang ada dalam dataset, yaitu *univariate analysis*.
3. **_Univariate analysis_**
Pada tahap ini, dilakukan analisis terhadap setiap kolom dalam dataset. Analisis dilakukan dengan membagi semua kolom ke dalam dua jenis, yaitu kolom kategori dan kolom numerik. Kolom kategori adalah kelompok kolom yang tipe datanya objek sedangkan kolom numerik adalah kelompok kolom yang tipe datanya numerik. Pertama adalah analisis pada kolom kategori. Hasil analisis kolom kategori adalah sebagai berikut.
    - Analisis pada kolom *mark*
    ![univariate_mark.jpg](https://drive.google.com/uc?export=view&id=1CdMe8VrwhCpANWB32xvzgWhHwpRu-j7i "univariate_mark")
    Gambar 13. Visualisasi *count plot* kolom *mark*
    Berdasarkan gambar 13, terdapat 23 kategori pada kolom *mark*, dengan jumlah terbanyak adalah kategori 'opel' dan jumlah paling sedikit adalah 'chevrolet'. Sekitar 20% lebih sampel berada pada kategori 'opel', 'volkswagen', dan 'ford'.
    - Analisis pada kolom *model*
    ![univariate_model.jpg](https://drive.google.com/uc?export=view&id=1WjPzIlMstZHvwQ-aI1hFEdAbpo1uQ2dC "univariate_model")
    Gambar 14. Visualisasi *count plot* kolom *model*
    Berdasarkan gambar 14, terdapat 290 kategori pada kolom *model* dengan jumlah terbanyak adalah kategori 'astra' dan paling sedikit adalah kategori 'f150'. Jumlah kategori pada kolom *model* terlalu banyak sehingga akan berakibat pada pelatihan model, sehingga kolom *model* dihapus.
    - Analisis pada kolom *fuel*
    ![univariate_fuel.jpg](https://drive.google.com/uc?export=view&id=1G7ppzqS4f7IR6esjSsbv30fwbtfepAQs "univariate_fuel")
    Gambar 15. Visualisasi *count plot* kolom *fuel*
    Berdasarkan gambar 15, terdapat 5 kategori pada kolom *fuel* dengan jumlah terbanyak adalah kategori 'Gasiline' dan paling sedikit adalah kategori 'Electric'. Dapat dilihat 90% lebih sampel merupakan kategori 'Gasoline' dan 'Diesel'.
    - Analisis pada kolom *city*
    ![univariate_city.jpg](https://drive.google.com/uc?export=view&id=1uihOWBi7W36UrGfLCDILNfCLjJe1bb3A "univariate_city")
    Gambar 16. Visualisasi *count plot* kolom *city*
    Berdasarkan gambar 16, terdapat 4151 kategori dalam kolom *city* dengan jumlah terbanyak adalah kategori 'Warszawa' dan paling sedikit adalah kategori 'Bledzew'. Jumlah kategori pada kolom *city* terlalu banyak sehingga akan berakibat pada pelatihan model, sehingga kolom *city* dihapus.
    - Analisis pada kolom *province*
    ![univariate_province.jpg](https://drive.google.com/uc?export=view&id=1CdMe8VrwhCpANWB32xvzgWhHwpRu-j7i "univariate_province")
    Gambar 17. Visualisasi *count plot* kolom *province*
    Berdasarkan gambar 17, terdapat 23 kategori pada kolom *province* dengan jumlah sampel terbanyak berada pada kategori 'Mazowieckie' dan paling sedikit adalah kategori 'Northern-Westfalen'. Dapat dilihat 30% lebih sampel berasal dari kategori 'Mazowieckie', 'Slaskie', dan 'Wielkopolskie'.

    Analisis pada kolom jenis kategori telah dilakukan. Kali ini adalah analisis terhadap kolom berjenis numerik.
    ![univariate_numeric.jpg](https://drive.google.com/uc?export=view&id=1R5S6Rg0osXAlne6xcqA11-Q0Z7eB52rs "univariate_numeric")
    Gambar 18. Visualisasi histogram kolom berjenis numerik
    Berdasarkan gambar 18, maka dapat disimpulkan beberapa hal berikut.
    - Pada rentang tahun produksi sekitar 1990-an hingga 2017, harga mobil bekas semakin naik naik dan semakin turun pada tahun produksi setelah 2017.
    - Harga mobil bekasi semakin naik hingga mobil berjarak tempuh 200.000 km dan semakin turun setelahnya.
    - Harga mobil bervariasi berdasarkan tipe mesin dan tidak dapat ditentukan polanya
    - Peningkatan harga mobil sebanding dengan penurunan jumlah sampel
    - Rentang harga mobil cukup tinggi yaitu meningkat hingga $162.000
    - Setengah harga mobil berada di bawah $100.000
    - Distribusi harga miring ke kanan.
    - Kesimpulan akhir, berdasarkan grafik terdapat suatu pola yang jelas antara tahun produksi, jarak tempuh, dan harga, tetapi tidak terdapat pola yang jelas antara tipe mesin dan harga.
    
    *Univariate analysis* telah selesai dilakukan, tahap selanjutnya adalah *multivariate analysis*.    
4. **_Multivariate analysis_**
Pada tahap ini, dianalisis hubungan dan korelasi setiap kolom terhadap kolom *price*. Sama seperti *univariate analysis*, pada tahap ini, analisis dilakukan dengan membagi semua kolom ke dalam dua jenis, yaitu kolom kategori dan kolom numerik. Kolom kategori adalah kelompok kolom yang tipe datanya objek sedangkan kolom numerik adalah kelompok kolom yang tipe datanya numerik. Pertama adalah analisis pada kolom kategori.
    ![mulitvariate_category.jpg](https://drive.google.com/uc?export=view&id=1AbeTuIdviJaI2LdRCkqh9nQ2W6knKAB9 "multivariate_category")
    Gambar 19. Visualisasi *catplot* kolom berjenis kategori terhadap kolom *price*
    Berdasarkan gambar 19, maka dapat diambil beberapa kesimpulan sebagai berikut.
    - Pada kolom *mark*, harga mobil relatif bervariasi tergantung merek mobil, sehingga dapat disimpulkan bahwa kolom *mark* berpengaruh besar terhadap harga mobil.
    - Pada kolom *fuel*, harga mobil sangat tinggi pada dua kategori, yaitu *hybrid* dan *electric* sehingga dapat disimpulkan kolom *fuel* juga memiliki pengaruh besar terhadap *price*.
    - Pada kolom *province*, harga mobil cenderung tinggi pada beberapa kategori, sehingga dapat disimpulkan bahwa harga mobil dipengaruhi oleh kolom *province*.
    - Kesimpulan akhir, semua kolom berjenis kategori memiliki pengaruh besar terhadap kolom *price*.
    
    ![multivariate_numeric.jpg](https://drive.google.com/uc?export=view&id=1CRikPZkJwszmshQBBrWHdjUbAkYomKXQ "mulitvariate_numeric")
    Gambar 20. Visualisasi *pairplot* kolom berjenis numerik
    ![heatmap_numeric.jpg](https://drive.google.com/uc?export=view&id=1qLk_QXjE5ZZxULqvTLlnYa0ucmKOIs0F "heatmap_numeric")
    Gambar 21. Visualisasi *heatmap* kolom berjenis numerik
    Berdasarkan gambar 20 dan 21, dapat disimpulkan bahwa hanya kolom *year* dan *mileage* yang memiliki cukup korelasi dengan kolom *price*, sehingga kolom *vol_engine* dihapus. *Multivariate analysis* telah selesai dilakukan yang berarti proses EDA juga telah selesai dilakukan. Setelah melalui proses EDA, hasil akhir dataset dapat dilihat pada gambar 22.
    ![final_after_eda.jpg](https://drive.google.com/uc?export=view&id=1T66MMk8Z5nKowaPODLc4GZll-rltLuAv "final_after_eda")
    Gambar 22. Hasil akhir dataset setelah EDA.

## Data Preparations
Pada tahap ini, terdapat tiga teknik yang digunakan dalam melakukan *data preparation*. Teknik-teknik tersebut adalah sebagai berikut.
1. **_One Hot Encoding_**
Teknik ini dilakukan untuk melakukan proses *encoding* pada kolom kategori seperti kolom *mark*, *fuel*, dan *province*. 
![one_hot.jpg](https://drive.google.com/uc?export=view&id=1Lojtm_ofhsJFz8ReXRmKpQb6UVdn0UNE "one_hot")
Gambar 23. Proses *one hot encoding*.
Berikut penjelasan proses yang dilakukan pada gambar 23.
    - Membuat label *one hot encoding* pada kolom *mark*, *fuel*, dan *province* dengan memanfaatkan fungsi *get_dummies()*.
    - Menghapus kolom *mark*, *fuel*, dan *province* karena sudah diberi label.
    - Terakhir, mengecek hasil pelabelan dengan memanggil fungsi *head()*. Hasil pelabelan dapat dilihat pada gambar 24.
    ![hasil_one_hot.jpg](https://drive.google.com/uc?export=view&id=1Y7WGePTzncnu0T8YBG549a4FvNGKodSF "hasil_one_hot")
    Gambar 24. Hasil *one hot encoding*.
    
    Berdasarkan gambar 24, terlihat terdapat kolom baru seperti *mark_audi*, *mark_bow*, dan *mark_chevrolet*. Kolom tersebut adalah hasil dari output dari fungsi *get_dummies()*. Nilai satu pada kolom tersebut berarti *True* dan nol berarti *False*.
2. **_Train Test Split_**
Teknik ini dilakukan untuk membagi dataset menjadi dua bagian, yaitu data latih dan data uji. data latih akan digunakan untuk melatih model sedangkan data uji akan digunakan untuk evaluasi model. Hal tersebut perlu diterapkan agar model yang telah dilatih dapat diuji menggunakan data yang belum pernah dianalisa oleh model.
![split.jpg](https://drive.google.com/uc?export=view&id=16dF-asrT_FN264Osqg9Tgxg74S8fj48I "split")
Gambar 25. Proses *train test split*.
Berdasarkan gambar 25, langkah-langkah yang dilakukan dalam menerapkan teknik ini adalah sebagai berikut.
    - Membagi dataset terlebih dahulu menjadi data y sebagai data target dan data X sebagai data fitur.
    - Membagi y dan X menjadi data latih dan data uji dengan rasio 0.975 : 0.025. Rasio tersebut dilakukan mengingat jumlah dataset yang besar setelah *data cleaning* yaitu sekitar 90.000. Pembagian dataset dilakukan dengan memanfaatkan *library train_test_split*.
    - Terakhir, mengecek masing-masing ukuran keseluruhan dataset, X, dan y untuk memastikan pembagian dataset berhasil diterapkan. Hasil *train test split* dapat dilihat pada gambar 26.
    ![hasil_split.jpg](https://drive.google.com/uc?export=view&id=1QRgbQLhLKlraZUDRSbwGdDT_lDKIaypc "hasil_split")
    Gambar 26. Hasil *train test split*.

    Berdasarkan gambar 26, proses pembagian dataset menjadi data latih dan data uji telah berhasil dilakukan. Dari 94393 baris keseluruhan dataset setelah melalui tahap *data cleaning*, terdapat 92033 baris merupakan data latih dan 2360 baris merupakan data uji.
3. **Standarisasi**
Teknik ini dilakukan untuk menyamakan skala antara kolom *year* dan *mileage*.
![standarisasi.jpg](https://drive.google.com/uc?export=view&id=1jaE9kwsjQMgOKDSFit1PS5a-8EHFLzPA "standarisasi")
Gambar 27. Proses standarisasi.
Berdasarkan gambar 27, langkah-langkah yang dilakukan dalam melakukan standarisasi adalah sebagai berikut.
    - Menyamakan skala antara kolom *year* dan *mileage* dengan memanfaatkan *library StandardScaler*.
    - Mengecek nilai *mean* dan standar deviasi untuk memastikan standarisasi berhasil diterapkan. Hasil pengecekan nilai *mean* dan standar deviasi dapat dilihat pada gambar 28.
    ![hasil_standarisasi.jpg](https://drive.google.com/uc?export=view&id=1AS7EspWqDD6CHCP7aTTE52a1nGn1MJOp "hasil_standarisasi")
    Gambar 28. Hasil standarisasi.

    Berdasarkan gambar 28, maka dapat terlihat bahwa proses standarisasi kolom *year* dan *mileage* telah berhasil dilakukan. Hal tersebut ditunjukkan dengan nilai *mean* dan standar deviasi masing-masing adalah nol dan satu.

## Modelling
Pada tahap ini, terdapat tiga algoritma *machine learning* yang digunakan dalam proyek ini. Ketiga algoritma *machine learning* tersebut adalah sebagai berikut.
1. **_K-Nearest Neighbor (KNN)_**
KNN adalah algoritma yang menggunakan ‘kesamaan fitur’ untuk memprediksi nilai dari setiap data yang baru. Dengan kata lain, setiap data baru diberi nilai berdasarkan seberapa mirip titik tersebut dalam data latih. Pada kasus regresi di proyek ini, KNN bekerja dengan membandingkan jarak satu sampel ke sampel pelatihan lain dengan memilih sejumlah k tetangga terdekat (*k nearest neighbor*). Kelebihan dari KNN adalah proses pelatihan yang cepat dan cocok untuk dataset besar. Hal tersebut sangat cocok dengan dataset yang digunakan pada proyek ini, yaitu dataset berjumlah besar. Kemudian untuk kekurangan KNN adalah KNN tidak efektif untuk melatih model dengan fitur yang terlalu banyak. Hal inilah yang menjadi alasan kolom *generation_name* dan *city* dihapus, karena setelah pelabelan, jika kolom tersebut tidak dihapus maka jumlah fitur yang ada akan sangat banyak, melebihin 1000 fitur. Pelatihan model ini menggunakan *n_neighbor* sebesar 15 dan metriks *euclidan* untuk mengukur jarak antar titik. Tahapan yang dilakukan untuk melatih model ini adalah sebagai berikut
    - Membuat *instance* dari *library KNeighborsRegressor*.
    - Melakukan pelatihan dengan *instance* yang dibuat.
    - Melakukan evaluasi terhadap performa pelatihan model dengan menggunakan *mean square error* dan menyimpannya pada *dataframe* pelatihan.
2. **_Random Forest_**
*Random forest* atau disingkat RF merupakan algoritma yang masuk ke dalam kategori *ensemble learning*. Kategori tersebut menggabungkan beberapa algoritma yang berbeda untuk mendapatkan tingkat keberhasilan yang lebih tinggi. Pada *ensemble learning* terdapat dua teknik pendekatan, yaitu *bagging* dan *boosting*. RF merupakan algoritma *ensemble* yang menggunakan pendekatan *bagging*. Seperti arti dari namanya, *bagging* akan mengumpulkan hasil prediksi seluruh algoritma yang digunakan. Pada kasus regresi di proyek ini, RF yang tersusun dari banyak algoritma *decision tree* akan melakukan pembelajaran dengan teknik *sampel with replacement*. Teknik tersebut akan memastikan *decision tree* yang berada dalam RF akan memiliki hasil prediksi yang berbeda. Setelah itu, hasil prediksi dari seluruh *decision tree* akan dirata-rata menjadi satu hasil prediksi. Hasil prediksi tersebut adalah hasil output dari RF. Kelebihan algoritma ini adalah memiliki stabilitas yang tinggi dan cocok digunakan baik untuk kasus klasifikasi dan regresi. Pada proyek ini, kasus yang terjadi adalah kasus regresi sehingga RF merupakan pilihan algoritma yang tepat. Kekurangan dari RF adalah proses pelatihan model yang dapat berjalan sangat lambat tergantung parameter yang digunakan. Hal tersebut dapat dilihat dari salah satu parameter yaitu jumlah *tree* yang digunakan dalam *random forest*. Semakin banyak *tree* maka proses pelatihan setiap *tree* dalam *random forest* akan bertambah. Pelatihan model ini menggunakan *n_estimators* sebesar 50, *max_depth* sebesar 8, *random_state* sebesar 55, dan *n_jobs* sebesar -1. Tahapan yang dilakukan untuk melatih model ini adalah sebagai berikut
    - Membuat *instance* dari *library RandomForestRegressor*.
    - Melakukan pelatihan dengan *instance* yang dibuat.
    - Melakukan evaluasi terhadap performa pelatihan model dengan menggunakan *mean square error* dan menyimpannya pada *dataframe* pelatihan.
3. **_AdaBoost_**
*AdaBoost* merupakan salah satu algoritma yang masuk ke dalam kategori *ensemble learning*. Berbeda dengan RF yang menggunakan teknik pendekatan *bagging*, *AdaBoost* menggunakan teknik pendekatan *boosting*. Pada kasus regresi di proyek ini, teknik yang digunakan tidak bekerja dengan cara mengumpulkan hasil prediksi dari seluruh model seperti pada RF, tetapi bekerja secara berurutan atau iteratif. Sebagai contoh pada iteratif pertama, model pertama akan melakukan pembelajaran terlebih dahulu, jika sudah selesai, 'kesalahan' dari model pertama akan dilanjutkan oleh model kedua. Proses tersebut akan terus diulang hingga mendapatkan hasil prediksi terbaik. Kelebihan algoritma ini adalah peningkatan nilai akurasi yang *powerful*. Hal tersebut dicapai karena proses iteratif yang dilakukan saat pelatihan model *AdaBoost*. Kekurangan algoritma ini adalah proses pelatihan yang kompleks. Pelatihan model ini menggunakan *learning_rate* sebesar 0.1 dan *random_state* sebesar 55. Tahapan yang dilakukan untuk melatih model ini adalah sebagai berikut
    - Membuat *instance* dari *library AdaBoostRegressor*.
    - Melakukan pelatihan dengan *instance* yang dibuat.
    - Melakukan evaluasi terhadap performa pelatihan model dengan menggunakan *mean square error* dan menyimpannya pada *dataframe* pelatihan.

## Evaluation
### *Mean Square Error (MSE)*
MSE merupakan metriks evaluasi yang sering digunakan dalam masalah regresi. metriks tersebut bekerja dengan menghitung jumlah selisih kuadrat rata-rata antara nilai sebenarnya dengan nilai yang diprediksi oleh model. Semakin tinggi hasil perhitungan MSE, maka semakin tinggi nilai error model. MSE didefinisikan dalam persamaan berikut.
> MSE = $$1 \over N$$ $$\sum_{i=1}^{N}(y_i-ypred_i)^2$$

### Hasil Pelatihan Model
Metriks performasi yang digunakan dalam proyek ini adalah MSE (Penjelasan MSE ada di atas). Evaluasi hasil pelatihan dimulai dengan melakukan standarisasi pada data uji. Hal tersebut dilakukan untuk menyamakan skala kolom numerik pada data uji. Setelah melakukan standarisasi pada data uji, kemudian model akan dievaluasi menggunakan MSE. Hasil evaluasi kemudian disimpan dalam tabel dan divisualisasikan seperti pada gambar 28 dan 29.

![mse_table.jpg](https://drive.google.com/uc?export=view&id=1GEhzmxJbpzSJmAuCl4kvVQsfjDBgOrzB "mse_table")
Gambar 29. Tabel MSE hasil pelatihan model.
![mse_barplot.jpg](https://drive.google.com/uc?export=view&id=1DOCzT-cHhJHI6f1ovJ15dAhAfNmEsvbC "mse_barplot")
Gambar 30. Visualisasi *barplot* MSE.

Berdasarkan gambar 29 dan 30, dapat diambil kesimpulan, bahwa model yang memiliki nilai error paling sedikit adalah KNN, sedangkan *AdaBoost* merupakan model yang nilai error-nya paling besar. Berdasarkan gambar 29 dan 30 juga dapat disimpulkan bahwa nilai error pada data latih lebih kecil dibanding nilai error pada data uji.

![mse_barplot.jpg](https://drive.google.com/uc?export=view&id=1-PnpAMvLEqAMoLHFti2tJ5_31DlOwZO0 "mse_barplot")
Gambar 31. Hasil prediksi model terhadap nilai sebenarnya.

Berdasarkan gambar 31, dapat disimpulkan bahwa prediksi model KNN, prediksi model RF, dan prediksi model *AdaBoost* adalah $20073, $25660, dan $26287 dari $19900. Dari ketiga model, model yang memiliki nilai prediksi meleset sangat kecil adalah model KNN dan model yang memiliki nilai prediksi meleset sangat besar adalah *AdaBoost*. Hal tersebut sejalan dengan hasil error pada pelatihan model, yaitu nilai error paling kecil adalah nilai error model KNN dan nilai error paling besar adalah nilai error model *AdaBoost*.

### Kesimpulan
Berdasarkan proyek yang telah dilakukan, maka dapat disimpulkan beberapa hal berikut.
- Berdasarkan EDA, fitur yang paling berkolerasi dengan harga mobil bekas adalah jarak tempuh dan tahun produksi mobil.
- Berdasarkan hasil MSE, model yang memiliki performa terbaik adalah model dengan algoritma *K-Nearest Neighbor (KNN)*.

> Proyek ini dapat diakses melalui url: https://colab.research.google.com/drive/1T_2amzSHvZPnpOXPjBcQRwVhgOB10lhi?usp=sharing

**Daftar Referensi**

- [1] Venkatasubbu, P., & Ganesh, M. (2019). *Used Cars Price Prediction using Supervised Learning Techniques*. Int. J. Eng. Adv. Technol.(IJEAT), 9(1S3).
- [2] Gegic, E., Isakovic, B., Keco, D., Masetic, Z., & Kevric, J. (2019). *Car price prediction using machine learning techniques*. TEM Journal, 8(1), 113.
- [3] Glotov, A. (2022). *Car Prices Poland*. https://www.kaggle.com/datasets/aleksandrglotov/car-prices-poland, diakses tanggal 18 November 2022.

**---Ini adalah bagian akhir dari laporan---**

[//]: # (These are reference links used in the body of this note and get stripped out when the markdown processor does its job. There is no need to format nicely because it shouldn't be seen. Thanks SO - http://stackoverflow.com/questions/4823468/store-comments-in-markdown-syntax)
   [kaggle]: <https://www.kaggle.com/datasets/aleksandrglotov/car-prices-poland>