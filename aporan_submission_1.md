# Laporan Proyek Machine Learning - Imam Asrowardi

## Domain Proyek

Nama Proyek : : Prediksi Kelayakan Pemberian Pinjaman
Dalam dunia keuangan, kelayakan kredit merupakan aspek krusial bagi lembaga keuangan, seperti bank dan perusahaan pemberi pinjaman, untuk meminimalkan risiko gagal bayar serta mengoptimalkan profitabilitas. Dengan meningkatnya jumlah permintaan pinjaman, perusahaan perlu membuat keputusan cepat dan akurat mengenai kelayakan kredit para pemohon. Data yang berkaitan dengan profil calon peminjam—seperti usia, pendidikan, pendapatan, pengalaman kerja, kepemilikan rumah, tujuan pinjaman, dan riwayat kredit sebelumnya—dapat memberikan wawasan penting dalam memprediksi kemungkinan gagal bayar.


**Rubrik/Kriteria Tambahan (Opsional)**:

- Alasan Masalah Perlu Diselesaikan** Memastikan bahwa hanya peminjam yang memenuhi syarat yang disetujui pinjamannya akan membantu perusahaan dalam:
    1. Mengurangi Risiko Keuangan: Meminimalkan potensi kerugian yang timbul akibat gagal bayar pinjaman.
Mengoptimalkan Proses Persetujuan Kredit: Dengan menerapkan model prediksi yang otomatis, perusahaan dapat menghemat waktu dan sumber daya.
    2. Meningkatkan Profitabilitas: Peningkatan akurasi dalam proses seleksi peminjam dapat membantu perusahaan fokus pada peminjam dengan profil yang sesuai, sehingga meningkatkan tingkat pengembalian dan profitabilitas.
Memberikan Keputusan yang Adil: Model berbasis data yang akurat dapat mengurangi bias dalam keputusan kredit yang mungkin timbul dari penilaian manusia.
  3. Pendekatan dan Metode Dengan menggunakan dataset yang mencakup informasi demografis serta riwayat kredit peminjam, proyek ini bertujuan membangun model klasifikasi untuk memprediksi "loan_status," di mana nilai "1" menandakan pinjaman yang berhasil (tidak mengalami default) dan "0" menandakan pinjaman gagal bayar.

Beberapa algoritma pembelajaran mesin yang relevan untuk digunakan adalah Logistic Regression, Random Forest, dan Gradient Boosting yang umum diterapkan pada masalah klasifikasi biner.

- Hasil Penelitian Terkait
1. Risk Modeling in Credit Loan Approval
    - Credit Risk Prediction using Extra Tree Ensembling Technique with Genetic Algorithm (https://ieeexplore.ieee.org/document/10307028

        Jurnal ini memaparkan model prediksi risiko kredit menggunakan teknik Extra Tree yang digabungkan dengan Genetic Algorithm. Penelitian ini bertujuan untuk meningkatkan akurasi prediksi risiko kredit melalui seleksi fitur menggunakan Genetic Algorithm serta teknik bagging dengan algoritma Extra Trees. Metode ini menunjukkan peningkatan dalam akurasi dan nilai f1-score, yang dievaluasi melalui validasi silang 10-fold.
      
   - Prediction of Credit Card Approval https://www.ijsce.org/wp-content/uploads/papers/v11i2/B35350111222.pdf
     
       Penelitian ini fokus pada prediksi persetujuan kartu kredit dengan model machine learning. Data pemohon dianalisis untuk memprediksi apakah aplikasi kredit akan disetujui. Teknik seperti random forest dan logistic regression digunakan, dengan hasil akurasi prediksi sebesar 86%. Proses evaluasi dilakukan melalui berbagai metode, termasuk analisis eksplorasi data dan pencarian grid untuk meningkatkan performa model.

    - Improving Credit Risk Assessment through Deep Learning-based Consumer Loan Default Prediction Model https://www.ssbfnet.com/ojs/index.php/ijfbs/article/view/2579
        Jurnal ini mengusulkan model prediksi risiko kredit berbasis deep learning yang dirancang untuk meningkatkan akurasi dalam menilai risiko gagal bayar pinjaman konsumen. Model ini menggunakan teknik data mining dan machine learning dengan tingkat akurasi prediksi default sebesar 95,2% pada set data uji. Studi ini menunjukkan potensi tinggi model ini untuk meminimalkan risiko kredit bagi bank melalui prediksi yang lebih akurat.-
      
2. Machine Learning in Financial Decision Making
    - Using Machine Learning Approach to Evaluate the Excessive Financialization Risks of Trading Enterprises https://link.springer.com/article/10.1007/s10614-020-10090-6

        Studi ini membahas model machine learning untuk mengidentifikasi risiko berlebih dalam sektor keuangan perusahaan perdagangan. Algoritma seperti decision tree, random forest, dan gradient boosting diterapkan untuk mengevaluasi data keuangan dalam mengontrol risiko. Model ini terbukti meningkatkan efisiensi prediksi dan memberikan dukungan bagi perusahaan dalam menghadapi risiko keuangan dengan model fusion yang mencakup berbagai teknik ensemble.

    - Loan Default Prediction of Chinese P2P Market: A Machine Learning Methodology https://www.nature.com/articles/s41598-021-98361-6

        Dalam studi ini, metode machine learning diterapkan untuk memprediksi kegagalan pembayaran dalam pasar peer-to-peer (P2P) di Tiongkok. Algoritma seperti random forest dan gradient boosting digunakan untuk mengidentifikasi faktor-faktor penting dalam prediksi gagal bayar. Hasilnya menunjukkan bahwa verifikasi identitas dan aset memiliki dampak signifikan dalam menurunkan risiko gagal bayar, dengan akurasi model yang mencapai lebih dari 90%.

      - Machine Learning Models for Predicting Bank Loan Eligibility https://ieeexplore.ieee.org/document/9803172

Artikel ini mengevaluasi model machine learning untuk prediksi kelayakan pinjaman bank dengan menggunakan algoritma seperti random forest, gradient boost, dan decision tree. Dengan menggunakan dataset historis, hasil penelitian menunjukkan bahwa random forest memiliki akurasi tertinggi, yakni 95,55%, di antara model lainnya. Studi ini menunjukkan kemampuan algoritma untuk mempercepat dan meningkatkan akurasi proses persetujuan pinjaman di bank.



## Business Understanding

Pada bagian ini, kamu perlu menjelaskan proses klarifikasi masalah.

Bagian laporan ini mencakup:

### Problem Statements

Menjelaskan pernyataan masalah latar belakang:
- Pernyataan Masalah 1
- Pernyataan Masalah 2
- Pernyataan Masalah n

### Goals

Menjelaskan tujuan dari pernyataan masalah:
- Jawaban pernyataan masalah 1
- Jawaban pernyataan masalah 2
- Jawaban pernyataan masalah n

Semua poin di atas harus diuraikan dengan jelas. Anda bebas menuliskan berapa pernyataan masalah dan juga goals yang diinginkan.

**Rubrik/Kriteria Tambahan (Opsional)**:
- Menambahkan bagian “Solution Statement” yang menguraikan cara untuk meraih goals. Bagian ini dibuat dengan ketentuan sebagai berikut: 

    ### Solution statements
    - Mengajukan 2 atau lebih solution statement. Misalnya, menggunakan dua atau lebih algoritma untuk mencapai solusi yang diinginkan atau melakukan improvement pada baseline model dengan hyperparameter tuning.
    - Solusi yang diberikan harus dapat terukur dengan metrik evaluasi.

## Data Understanding
Paragraf awal bagian ini menjelaskan informasi mengenai data yang Anda gunakan dalam proyek. Sertakan juga sumber atau tautan untuk mengunduh dataset. Contoh: [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/Restaurant+%26+consumer+data).

Selanjutnya uraikanlah seluruh variabel atau fitur pada data. Sebagai contoh:  

### Variabel-variabel pada Restaurant UCI dataset adalah sebagai berikut:
- accepts : merupakan jenis pembayaran yang diterima pada restoran tertentu.
- cuisine : merupakan jenis masakan yang disajikan pada restoran.
- dst

**Rubrik/Kriteria Tambahan (Opsional)**:
- Melakukan beberapa tahapan yang diperlukan untuk memahami data, contohnya teknik visualisasi data atau exploratory data analysis.

## Data Preparation
Pada bagian ini Anda menerapkan dan menyebutkan teknik data preparation yang dilakukan. Teknik yang digunakan pada notebook dan laporan harus berurutan.

**Rubrik/Kriteria Tambahan (Opsional)**: 
- Menjelaskan proses data preparation yang dilakukan
- Menjelaskan alasan mengapa diperlukan tahapan data preparation tersebut.

## Modeling
Tahapan ini membahas mengenai model machine learning yang digunakan untuk menyelesaikan permasalahan. Anda perlu menjelaskan tahapan dan parameter yang digunakan pada proses pemodelan.

**Rubrik/Kriteria Tambahan (Opsional)**: 
- Menjelaskan kelebihan dan kekurangan dari setiap algoritma yang digunakan.
- Jika menggunakan satu algoritma pada solution statement, lakukan proses improvement terhadap model dengan hyperparameter tuning. **Jelaskan proses improvement yang dilakukan**.
- Jika menggunakan dua atau lebih algoritma pada solution statement, maka pilih model terbaik sebagai solusi. **Jelaskan mengapa memilih model tersebut sebagai model terbaik**.

## Evaluation
Pada bagian ini anda perlu menyebutkan metrik evaluasi yang digunakan. Lalu anda perlu menjelaskan hasil proyek berdasarkan metrik evaluasi yang digunakan.

Sebagai contoh, Anda memiih kasus klasifikasi dan menggunakan metrik **akurasi, precision, recall, dan F1 score**. Jelaskan mengenai beberapa hal berikut:
- Penjelasan mengenai metrik yang digunakan
- Menjelaskan hasil proyek berdasarkan metrik evaluasi

Ingatlah, metrik evaluasi yang digunakan harus sesuai dengan konteks data, problem statement, dan solusi yang diinginkan.

**Rubrik/Kriteria Tambahan (Opsional)**: 
- Menjelaskan formula metrik dan bagaimana metrik tersebut bekerja.

**---Ini adalah bagian akhir laporan---**

_Catatan:_
- _Anda dapat menambahkan gambar, kode, atau tabel ke dalam laporan jika diperlukan. Temukan caranya pada contoh dokumen markdown di situs editor [Dillinger](https://dillinger.io/), [Github Guides: Mastering markdown](https://guides.github.com/features/mastering-markdown/), atau sumber lain di internet. Semangat!_
- Jika terdapat penjelasan yang harus menyertakan code snippet, tuliskan dengan sewajarnya. Tidak perlu menuliskan keseluruhan kode project, cukup bagian yang ingin dijelaskan saja.

