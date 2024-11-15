# Laporan Proyek Machine Learning - Imam Asrowardi

## Domain Proyek

Nama Proyek : Prediksi Kelayakan Pemberian Pinjaman

Dalam dunia keuangan, kelayakan kredit merupakan aspek krusial bagi lembaga keuangan, seperti bank dan perusahaan pemberi pinjaman, untuk meminimalkan risiko gagal bayar serta mengoptimalkan profitabilitas. Dengan meningkatnya jumlah permintaan pinjaman, perusahaan perlu membuat keputusan cepat dan akurat mengenai kelayakan kredit para pemohon. Data yang berkaitan dengan profil calon peminjam—seperti usia, pendidikan, pendapatan, pengalaman kerja, kepemilikan rumah, tujuan pinjaman, dan riwayat kredit sebelumnya—dapat memberikan wawasan penting dalam memprediksi kemungkinan gagal bayar [1].

Beberapa algoritma pembelajaran mesin yang relevan untuk digunakan adalah Logistic Regression, Random Forest, dan Gradient Boosting yang umum diterapkan pada masalah klasifikasi biner [2]-[4].


**Rubrik/Kriteria Tambahan (Opsional)**:

- Alasan Masalah Perlu Diselesaikan**

  Memastikan bahwa hanya peminjam yang memenuhi syarat yang disetujui pinjamannya akan membantu perusahaan dalam:
    1. Mengurangi Risiko Keuangan: Meminimalkan potensi kerugian yang timbul akibat gagal bayar pinjaman.
Mengoptimalkan Proses Persetujuan Kredit: Dengan menerapkan model prediksi yang otomatis, perusahaan dapat menghemat waktu dan sumber daya.
    2. Meningkatkan Profitabilitas: Peningkatan akurasi dalam proses seleksi peminjam dapat membantu perusahaan fokus pada peminjam dengan profil yang sesuai, sehingga meningkatkan tingkat pengembalian dan profitabilitas.
Memberikan Keputusan yang Adil: Model berbasis data yang akurat dapat mengurangi bias dalam keputusan kredit yang mungkin timbul dari penilaian manusia.
  3. Pendekatan dan Metode Dengan menggunakan dataset yang mencakup informasi demografis serta riwayat kredit peminjam, proyek ini bertujuan membangun model klasifikasi untuk memprediksi "loan_status," di mana nilai "1" menandakan pinjaman yang berhasil (tidak mengalami default) dan "0" menandakan pinjaman gagal bayar.

Beberapa algoritma pembelajaran mesin yang relevan untuk digunakan adalah Logistic Regression, Random Forest, dan Gradient Boosting yang umum diterapkan pada masalah klasifikasi biner.

## Hasil Penelitian Terkait

Berikut ini adalah beberapa hasil penelitian yang relevan dengan model prediksi kelayakan kredit:

1. **Risk Modeling in Credit Loan Approval**  
   Jurnal ini memaparkan model prediksi risiko kredit menggunakan teknik Extra Tree yang digabungkan dengan Genetic Algorithm. Penelitian ini bertujuan untuk meningkatkan akurasi prediksi risiko kredit melalui seleksi fitur menggunakan Genetic Algorithm serta teknik bagging dengan algoritma Extra Trees. Metode ini menunjukkan peningkatan dalam akurasi dan nilai f1-score, yang dievaluasi melalui validasi silang 10-fold [2].

2. **Prediction of Credit Card Approval**  
   Penelitian ini fokus pada prediksi persetujuan kartu kredit dengan model machine learning. Data pemohon dianalisis untuk memprediksi apakah aplikasi kredit akan disetujui. Teknik seperti random forest dan logistic regression digunakan, dengan hasil akurasi prediksi sebesar 86%. Proses evaluasi dilakukan melalui berbagai metode, termasuk analisis eksplorasi data dan pencarian grid untuk meningkatkan performa model [3].

3. **Improving Credit Risk Assessment through Deep Learning-based Consumer Loan Default Prediction Model**  
   Jurnal ini mengusulkan model prediksi risiko kredit berbasis deep learning yang dirancang untuk meningkatkan akurasi dalam menilai risiko gagal bayar pinjaman konsumen. Model ini menggunakan teknik data mining dan machine learning dengan tingkat akurasi prediksi default sebesar 95,2% pada set data uji. Studi ini menunjukkan potensi tinggi model ini untuk meminimalkan risiko kredit bagi bank melalui prediksi yang lebih akurat [4].

4. **Using Machine Learning Approach to Evaluate the Excessive Financialization Risks of Trading Enterprises**  
   Studi ini membahas model machine learning untuk mengidentifikasi risiko berlebih dalam sektor keuangan perusahaan perdagangan. Algoritma seperti decision tree, random forest, dan gradient boosting diterapkan untuk mengevaluasi data keuangan dalam mengontrol risiko. Model ini terbukti meningkatkan efisiensi prediksi dan memberikan dukungan bagi perusahaan dalam menghadapi risiko keuangan dengan model fusion yang mencakup berbagai teknik ensemble [5].

5. **Loan Default Prediction of Chinese P2P Market: A Machine Learning Methodology**  
   Dalam studi ini, metode machine learning diterapkan untuk memprediksi kegagalan pembayaran dalam pasar peer-to-peer (P2P) di Tiongkok. Algoritma seperti random forest dan gradient boosting digunakan untuk mengidentifikasi faktor-faktor penting dalam prediksi gagal bayar. Hasilnya menunjukkan bahwa verifikasi identitas dan aset memiliki dampak signifikan dalam menurunkan risiko gagal bayar, dengan akurasi model yang mencapai lebih dari 90% [6].

6. **Machine Learning Models for Predicting Bank Loan Eligibility**  
   Artikel ini mengevaluasi model machine learning untuk prediksi kelayakan pinjaman bank dengan menggunakan algoritma seperti random forest, gradient boost, dan decision tree. Dengan menggunakan dataset historis, hasil penelitian menunjukkan bahwa random forest memiliki akurasi tertinggi, yakni 95,55%, di antara model lainnya. Studi ini menunjukkan kemampuan algoritma untuk mempercepat dan meningkatkan akurasi proses persetujuan pinjaman di bank [7].

## Referensi :
[1] T. J. Thomas and J. Crook, "Risk Modeling in Credit Loan Approval," *Journal of Risk Management*, vol. 12, pp. 23–34, 2020.  
[2] A. Kumar et al., "Credit Risk Prediction using Extra Tree Ensembling Technique with Genetic Algorithm," *IEEE Transactions on Computational Social Systems*, 2023. Available: [https://ieeexplore.ieee.org/document/10307028](https://ieeexplore.ieee.org/document/10307028)  
[3] S. Roy, "Prediction of Credit Card Approval," *International Journal of Soft Computing and Engineering*, vol. 11, no. 2, pp. 12–15, 2022. Available: [https://www.ijsce.org/wp-content/uploads/papers/v11i2/B35350111222.pdf](https://www.ijsce.org/wp-content/uploads/papers/v11i2/B35350111222.pdf)  
[4] D. Park et al., "Improving Credit Risk Assessment through Deep Learning-based Consumer Loan Default Prediction Model," *International Journal of Finance & Banking Studies*, vol. 10, no. 2, pp. 54–65, 2021. Available: [https://www.ssbfnet.com/ojs/index.php/ijfbs/article/view/2579](https://www.ssbfnet.com/ojs/index.php/ijfbs/article/view/2579)  
[5] Y. Zhao and H. Wang, "Using Machine Learning Approach to Evaluate the Excessive Financialization Risks of Trading Enterprises," *Computational Economics*, vol. 56, pp. 215–229, 2020. Available: [https://link.springer.com/article/10.1007/s10614-020-10090-6](https://link.springer.com/article/10.1007/s10614-020-10090-6)  
[6] J. Chen et al., "Loan Default Prediction of Chinese P2P Market: A Machine Learning Methodology," *Scientific Reports*, vol. 11, no. 38, pp. 1–10, 2021. Available: [https://www.nature.com/articles/s41598-021-98361-6](https://www.nature.com/articles/s41598-021-98361-6)  
[7] S. Li, X. Zhang, and M. Yang, "Machine Learning Models for Predicting Bank Loan Eligibility," in *Proceedings of the IEEE International Conference on Big Data (Big Data)*, 2022, pp. 1234–1241. Available: [https://ieeexplore.ieee.org/document/9803172](https://ieeexplore.ieee.org/document/9803172) 



## Business Understanding
### Problem Statements
Perusahaan pemberi pinjaman menghadapi masalah utama dalam menentukan kelayakan peminjam, yang secara langsung berdampak pada potensi risiko keuangan akibat gagal bayar. Dalam data yang tersedia, "loan_status" menunjukkan apakah peminjam berhasil memenuhi kewajibannya atau mengalami gagal bayar. Tantangan utama yang perlu dijawab adalah:
1. Bagaimana perusahaan dapat membedakan peminjam yang layak dan berisiko? Memahami faktor-faktor yang memengaruhi kelayakan peminjam akan membantu meminimalkan risiko gagal bayar.
2. Bagaimana meningkatkan keakuratan prediksi kelayakan kredit menggunakan data demografis dan riwayat kredit? Dengan data ini, perusahaan diharapkan bisa mendapatkan wawasan untuk meningkatkan proses persetujuan kredit.

### Goals
    Membangun model klasifikasi yang akurat untuk memprediksi kelayakan pinjaman.

Model ini diharapkan dapat menganalisis berbagai karakteristik peminjam, seperti riwayat keuangan, profil demografis, dan faktor risiko lainnya, guna memprediksi kemungkinan gagal bayar atau keberhasilan dalam memenuhi kewajiban pinjaman. Dengan adanya prediksi yang akurat, perusahaan akan lebih mudah menentukan apakah permohonan kredit layak diterima atau sebaiknya ditolak, sehingga risiko keuangan dapat diminimalkan dan kualitas portofolio kredit terjaga. Lebih jauh, model ini juga memberikan keuntungan dalam hal efisiensi operasional, karena proses persetujuan kredit yang sebelumnya memerlukan waktu panjang kini dapat dilakukan dengan lebih cepat dan sistematis. Selain itu, model prediktif ini dapat membantu perusahaan untuk mengidentifikasi pola risiko dan memperkuat strategi mitigasi risiko di masa mendatang, sehingga menjaga kestabilan arus kas dan meningkatkan profitabilitas secara keseluruhan. Implementasi model ini tidak hanya mendukung pengambilan keputusan yang berbasis data, tetapi juga meningkatkan keadilan dalam pemberian kredit, karena keputusan dibuat berdasarkan analisis obyektif atas data peminjam.

**Rubrik/Kriteria Tambahan (Opsional)**:
   ### Solution statements
Untuk mencapai tujuan tersebut, beberapa solusi berikut diusulkan:
    - Algoritma Logistic Regression: Model regresi logistik sering digunakan sebagai model dasar karena kesederhanaannya dan interpretabilitasnya. Logistic regression dapat membantu mengidentifikasi pengaruh masing-masing fitur pada probabilitas gagal bayar.
    - Metode Evaluasi: Akurasi dan area under the ROC curve (AUC) akan menjadi metrik evaluasi utama. Logistic regression akan memberikan baseline yang berguna untuk mengevaluasi solusi lanjutan.

## Data Understanding

1. Informasi Jumlah Data
    - Dataset ini memiliki:
        - Jumlah Baris (Data Points): 45.000
        - Jumlah Kolom (Fitur): 14
2. Kondisi Data
    - Missing Values: Tidak ada nilai yang hilang pada dataset ini. Setiap kolom memiliki data lengkap di setiap baris.
    - Duplicate Values: tidak ditemukan duplikasi data.
    - Outliers: Beberapa outlier teridentifikasi pada fitur seperti person_age (usia maksimum 144 tahun), person_income (pendapatan sangat tinggi), dan loan_amnt. Hal ini mengindikasikan kemungkinan adanya data ekstrem atau data yang perlu diperiksa lebih lanjut untuk analisis.
    - Sumber Data : https://www.kaggle.com/datasets/taweilo/loan-approval-classification-data?select=loan_data.csv
3. Uraian Fitur pada Data Berikut deskripsi lengkap dari masing-masing fitur dalam dataset:
    - person_age: Usia peminjam dalam tahun.
    - person_gender: Jenis kelamin peminjam.
    - person_education: Tingkat pendidikan peminjam.
    - person_income: Pendapatan tahunan peminjam dalam satuan mata uang lokal.
    - person_emp_exp: Lama pengalaman kerja peminjam (dalam tahun).
    - person_home_ownership: Status kepemilikan rumah peminjam (RENT, OWN, atau MORTGAGE).
    - loan_amnt: Jumlah pinjaman yang diajukan oleh peminjam.
    - loan_intent: Tujuan atau niat penggunaan pinjaman (misalnya, untuk keperluan kesehatan, pendidikan, konsumsi pribadi).
    - loan_int_rate: Suku bunga pinjaman, yang menunjukkan biaya tahunan untuk pinjaman.
    - loan_percent_income: Persentase jumlah pinjaman terhadap pendapatan tahunan peminjam.
    - cb_person_cred_hist_length: Panjang riwayat kredit peminjam, yang mencerminkan usia akun kredit tertua.
    - credit_score: Skor kredit peminjam yang menunjukkan tingkat risiko kredit mereka.
    - previous_loan_defaults_on_file: Riwayat gagal bayar sebelumnya (Yes untuk pernah gagal bayar, No jika tidak).
    - loan_status: Status akhir pinjaman (1 menandakan keberhasilan pembayaran, 0 menandakan gagal bayar).

**Rubrik/Kriteria Tambahan (Opsional)**:
- Exploratory Data Analysis (EDA) digunakan untuk memahami lebih lanjut mengenai distribusi dan korelasi antar variabel, mari kita lakukan beberapa visualisasi. Pertama, kita akan melihat distribusi usia, pendapatan, dan jumlah pinjaman, serta memeriksa korelasi antara variabel numerik Hasil Exploratory Data Analysis (EDA) Distribusi Usia:
    1. Distribusi usia peminjam menunjukkan puncak pada sekitar usia 25 hingga 30 tahun. Namun, terdapat beberapa outlier, terutama pada usia yang lebih tinggi, yang perlu diperiksa lebih lanjut.
    2. Distribusi Pendapatan: Pendapatan tahunan memiliki distribusi yang lebar dengan beberapa outlier signifikan pada rentang yang sangat tinggi. Ini mungkin mengindikasikan adanya peminjam dari kelompok pendapatan yang jauh di atas rata-rata.
Distribusi Jumlah Pinjaman: Jumlah pinjaman yang diajukan sebagian besar berkisar antara 5.000 hingga 12.000, dengan puncak yang signifikan pada batas bawah sekitar 5.000.
    3. Heatmap Korelasi: Korelasi antar variabel numerik menunjukkan beberapa poin penting:
Variabel loan_percent_income memiliki korelasi moderat dengan loan_amnt, yang menunjukkan bahwa persentase pendapatan terhadap pinjaman naik seiring dengan meningkatnya jumlah pinjaman.
    4. credit_score menunjukkan korelasi negatif ringan dengan loan_status, yang berarti peminjam dengan skor kredit lebih tinggi cenderung memiliki status pinjaman yang lebih baik.

## Data Preparation
Langkah-langkah data preparation ini dilakukan untuk membersihkan, mengubah, dan mengoptimalkan data sehingga model dapat memahami pola dengan lebih baik dan menghasilkan prediksi yang lebih akurat. Langkah-langkah Data Preparation
1. Mengidentifikasi dan Menangani Outlier
- Proses: Kami melakukan pengecekan outlier pada kolom person_age dan person_income. Untuk dataset pinjaman, data yang memiliki usia lebih dari 100 tahun atau pendapatan tahunan lebih dari 500.000 dianggap sebagai outlier yang tidak realistis dalam konteks umum.
- Alasan: Outlier dapat mempengaruhi hasil model karena beberapa algoritma rentan terhadap nilai ekstrem. Menghapus outlier membantu model untuk lebih fokus pada pola-pola yang relevan dan umum.
2. Encoding Variabel Kategorikal
- Proses: Variabel kategorikal seperti person_gender, person_education, person_home_ownership, loan_intent, dan previous_loan_defaults_on_file diubah menjadi angka menggunakan Label Encoding. Dengan metode ini, setiap kategori diberi label angka unik.
- Alasan: Model pembelajaran mesin tidak dapat langsung bekerja dengan data dalam format kategorikal. Encoding diperlukan untuk mengonversi data tersebut menjadi numerik, sehingga model dapat memprosesnya dalam algoritma komputasi.
3. Feature Scaling (Standarisasi)
- Proses: Kami menggunakan StandardScaler untuk menstandarkan kolom numerik seperti person_income, loan_amnt, loan_int_rate, credit_score, loan_percent_income, dan cb_person_cred_hist_length agar setiap fitur memiliki mean 0 dan standar deviasi 1.
- Alasan: Model sering kali bekerja lebih baik dengan data yang distandarisasi, terutama ketika data numerik memiliki skala yang sangat berbeda. Dengan standarisasi, kita memastikan semua fitur berada dalam rentang yang sama, sehingga tidak ada fitur yang mendominasi atau terlalu berpengaruh pada hasil model.
4. Split Data (Pembagian Data)
- Proses: Kami membagi data menjadi data training (80%) dan data testing (20%) untuk mengevaluasi kinerja model. Data training digunakan untuk melatih model, sedangkan data testing digunakan untuk pengujian akhir.
- Alasan: Pembagian data sangat penting agar kita memiliki data terpisah untuk melatih model dan mengukur performanya pada data baru yang belum pernah dilihat model. Ini membantu kita memahami kemampuan model dalam generalisasi.


## Modeling

Proyek ini menggunakan tiga algoritma machine learning utama untuk memprediksi kelayakan pinjaman, yaitu **Logistic Regression**, **Random Forest Classifier**, dan **Gradient Boosting Classifier**. Masing-masing algoritma memiliki cara kerja, kelebihan, kekurangan, dan kegunaan khusus. Berikut adalah penjelasan terkait ketiga algoritma tersebut.

### 1. Logistic Regression
Logistic Regression adalah algoritma yang sering digunakan sebagai model baseline dalam masalah klasifikasi biner. Algoritma ini bekerja dengan mengukur hubungan antara variabel input (fitur) dan variabel target biner, seperti "lulus/tidak lulus" atau "mampu bayar/tidak mampu bayar". Logistic Regression menghasilkan probabilitas untuk setiap kelas, yang kemudian digunakan untuk memutuskan kelas akhir dari sebuah data.

Cara kerja Logistic Regression dimulai dengan membuat fungsi linear dari fitur-fitur input, yang disebut **logit**. Logit ini kemudian dimasukkan ke dalam **fungsi sigmoid**, yang mengubah nilai linear menjadi probabilitas antara 0 dan 1. Probabilitas ini menunjukkan peluang suatu data termasuk dalam kelas positif. Biasanya, ambang batas 0.5 digunakan untuk menentukan kelas: jika probabilitas lebih besar dari 0.5, data diklasifikasikan sebagai kelas positif, sebaliknya sebagai kelas negatif.

Logistic Regression sangat berguna ketika hubungan antara fitur dan variabel target bersifat linier dan sederhana. Model ini cepat dan mudah diinterpretasikan, tetapi performanya bisa menurun pada dataset yang memiliki pola non-linear atau interaksi kompleks antar-fitur.

- **Contoh Penerapan**: Menilai kelayakan pinjaman, memprediksi kemungkinan penyakit, dan klasifikasi email sebagai spam atau bukan spam.
- **Kelebihan**: Cepat, efisien, dan mudah diinterpretasi; cocok untuk dataset besar dengan pola linier.
- **Kekurangan**: Terbatas dalam menangkap pola non-linear dan rentan terhadap multikolinearitas serta outlier.

### 2. Random Forest Classifier
Random Forest adalah algoritma **ensemble** yang menggabungkan banyak pohon keputusan untuk membuat prediksi yang lebih kuat dan akurat. Algoritma ini bekerja berdasarkan prinsip **bagging (Bootstrap Aggregating)**, yang artinya banyak pohon keputusan independen dilatih pada subset data yang berbeda, dan prediksi akhir diambil berdasarkan suara mayoritas (untuk klasifikasi) atau rata-rata (untuk regresi).

Setiap pohon dalam Random Forest dilatih pada subset data acak yang diambil dari data pelatihan. Selain itu, hanya sebagian fitur yang dipilih pada setiap pembagian di dalam pohon, sehingga pohon-pohon menjadi berbeda dan tidak saling tergantung. Ketika membuat prediksi, setiap pohon memberikan suara, dan kelas yang memiliki suara terbanyak dipilih sebagai prediksi akhir.

Random Forest sangat efektif untuk menangani data yang memiliki fitur numerik maupun kategorikal, serta tahan terhadap overfitting karena tiap pohon hanya mempelajari sebagian dari keseluruhan data. Algoritma ini sangat akurat dan andal, terutama ketika ada banyak fitur yang relevan.

- **Contoh Penerapan**: Deteksi penipuan keuangan, peringkat risiko kredit, dan klasifikasi objek dalam visi komputer.
- **Kelebihan**: Tahan terhadap overfitting, mampu menangani banyak fitur, bekerja baik dengan variabel numerik dan kategorikal.
- **Kekurangan**: Memerlukan banyak sumber daya komputasi dan lebih sulit diinterpretasi dibandingkan model sederhana.

### 3. Gradient Boosting Classifier
Gradient Boosting adalah algoritma **boosting** yang bertujuan untuk meningkatkan akurasi model dengan cara memperbaiki kesalahan model-model sebelumnya. Berbeda dengan Random Forest yang membuat banyak pohon secara bersamaan, Gradient Boosting membangun pohon keputusan satu per satu secara bertahap. Setiap pohon baru ditambahkan untuk mengurangi kesalahan dari pohon sebelumnya, dengan tujuan untuk meminimalkan fungsi loss secara keseluruhan.

Gradient Boosting bekerja dengan menghitung **residual** atau kesalahan prediksi dari model sebelumnya, lalu melatih pohon berikutnya untuk memprediksi residual tersebut. Proses ini terus berlanjut hingga jumlah pohon yang diinginkan tercapai atau error yang dihasilkan telah minimal. Pada akhirnya, setiap pohon memberikan kontribusi dalam proporsi tertentu untuk menghasilkan prediksi akhir, yang merupakan kombinasi dari seluruh pohon yang telah dilatih.

Gradient Boosting sangat efektif untuk menangani dataset yang kompleks dengan pola non-linear. Algoritma ini sering kali menghasilkan akurasi tinggi, tetapi membutuhkan waktu pelatihan yang lebih lama dan rentan terhadap overfitting tanpa pengaturan parameter yang tepat.

- **Contoh Penerapan**: Analisis risiko investasi, peringkat relevansi dalam mesin pencari, dan deteksi penipuan dalam transaksi keuangan.
- **Kelebihan**: Sangat akurat dalam menangani data non-linear dan cocok untuk dataset kompleks.
- **Kekurangan**: Membutuhkan lebih banyak waktu dan sumber daya komputasi, serta lebih sulit diinterpretasi karena proses boosting yang kompleks.

Dari ketiga algoritma ini:
- **Logistic Regression** cocok sebagai model baseline karena kesederhanaannya dan kecepatan dalam menghasilkan prediksi.
- **Random Forest** memberikan akurasi yang tinggi dengan toleransi terhadap overfitting, menjadikannya pilihan ideal untuk aplikasi dengan data yang kompleks.
- **Gradient Boosting** memberikan akurasi yang lebih tinggi lagi, tetapi membutuhkan lebih banyak waktu pelatihan, menjadikannya cocok untuk aplikasi yang membutuhkan presisi tinggi.

Pemilihan model terbaik didasarkan pada hasil evaluasi metrik performa yang menunjukkan bahwa **Random Forest** adalah model dengan keseimbangan terbaik antara akurasi dan efisiensi dalam proyek ini.

1. Parameter yang Digunakan:
    - Logistic Regression: Menggunakan parameter default dengan regularisasi L2, untuk menghindari overfitting.
    - Random Forest Classifier: Menggunakan parameter default untuk jumlah pohon (n_estimators=100) serta max_depth yang akan disesuaikan pada tahap hyperparameter tuning.
    - Gradient Boosting Classifier: Menggunakan learning_rate=0.1 dan n_estimators=100 sebagai parameter awal, dan nantinya akan dituning.
2. Evaluasi Model: Metrik yang akan digunakan meliputi:
    - Accuracy: Persentase prediksi yang benar. AUC (Area Under the Curve): Mengukur kemampuan model dalam memisahkan dua kelas.
    - F1-Score: Memberikan keseimbangan antara precision dan recall.
      

## Evaluation
Berdasarkan hasil perbandingan tiga model yang digunakan, yaitu Logistic Regression, Random Forest, dan Gradient Boosting, terlihat bahwa masing-masing model memiliki performa yang berbeda berdasarkan tiga metrik evaluasi utama: Accuracy, AUC (Area Under the Curve), dan F1 Score. Metrik-metrik ini digunakan untuk mengukur kemampuan model dalam memprediksi kelayakan pinjaman dengan akurasi dan keandalan yang tinggi sehingga dapat disimpulkan:
1. Random Forest adalah model terbaik dengan akurasi tertinggi (92.39%), AUC tertinggi (0.9711), dan F1 Score tertinggi (0.8162). Model ini menunjukkan performa yang baik dalam mengidentifikasi peminjam berisiko tinggi sambil mempertahankan keseimbangan yang baik antara precision dan recall.
2. AUC yang tinggi menunjukkan bahwa model mampu membedakan dengan baik antara peminjam yang akan membayar dan yang akan gagal bayar, yang sangat penting untuk keputusan berbasis risiko.
3. F1 Score yang tinggi memastikan bahwa model efektif dalam menghindari kedua jenis kesalahan yang dapat membawa risiko keuangan.

Kesimpulan Model Random Forest dipilih sebagai model terbaik untuk permasalahan ini. Model ini memberikan keseimbangan optimal antara akurasi, kemampuan untuk membedakan kelas, dan keseimbangan antara precision dan recall. Dengan metrik evaluasi ini, perusahaan dapat lebih yakin dalam menggunakan model untuk meminimalkan risiko gagal bayar pada proses penilaian kelayakan pinjaman.
