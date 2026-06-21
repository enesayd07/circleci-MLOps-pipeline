# Uctan Uca MLOps Pipeline: Fashion MNIST

Bu projenin temel odak noktası; **CircleCI kullanarak CI/CD/CT (Sürekli Entegrasyon / Sürekli Dağıtım / Sürekli Eğitim) boru hatlarının izlenmesi ve otomatikleştirilmesi** ile **Render platformu üzerinde Docker tabanlı modern bir bulut API mimarisinin kurulmasıdır.**

Siz sadece kodunuzda veya veri setinizde bir değişiklik yapıp bunu GitHub'a gönderirsiniz. Geri kalan tüm süreç (verinin hazırlanması, modelin eğitilmesi, test edilmesi ve bulut sunucusuna yüklenmesi) CircleCI ve Render tarafından otomatik olarak yönetilir.

## Projenin Kokleri ve Makaleden Evrimi

Projenin çıkış noktası ve temel iş akışı (pipeline) mantığı, TensorFlow'un resmi eğitimlerinden olan TensorFlow REST Simple Tutorial projesine ve CircleCI'ın resmi blogunda yayınlanan temel MLOps makalelerine dayanmaktadır.

Bu mimari kurgulanırken aşağıdaki kaynaklardan ilham alınmıştır:

* **TensorFlow TFX ve Model Sunumu:** [TensorFlow REST Simple Tutorial](https://github.com/tensorflow/tfx/blob/master/docs/tutorials/serving/rest_simple.ipynb)
* **CircleCI ile Sürekli Entegrasyon (CI):** [CI for Machine Learning](https://circleci.com/blog/ci-for-machine-learning/)
* **CircleCI ile Sürekli Dağıtım (CD):** [CD for Machine Learning](https://circleci.com/blog/cd-for-machine-learning/)

**Ancak orijinal makalelerdeki yapı alınıp üzerinde su kritik degisiklikler ve eklemeler yapilmistir:**

1. **TensorFlow Serving Yerine FastAPI:** Orijinal proje, modeli sunmak için hantal ve yapılandırması zor olan `TensorFlow Serving` kullanıyordu. Biz bu yapıyı tamamen kaldırıp, yerine çok daha hafif, Pythonik ve özelleştirilebilir modern bir web framework'ü olan **FastAPI** mimarisini entegre ettik.
2. **Docker ve Tam İzolasyon:** Orijinal makale modelleri SSH üzerinden sunuculara kopyalamayı hedefliyordu. Biz projeyi tamamen konteynerize ederek (Docker) platform bağımsız hale getirdik. Modelin ağırlıkları, Docker inşası (build) sırasında doğrudan konteynerin içine gömülecek şekilde pürüzsüz bir yapı tasarlandı.
3. **Render Bulut ve Webhook Otomasyonu:** Manuel sunucu kurulumları yerine, CircleCI'ın testleri başarıyla bitirdiği an Render platformuna "Webhook" üzerinden tetikleme gönderdiği tam otomatik bir bulut (Cloud) dağıtım mimarisi eklendi.
4. **Quality Gate:** Modele sadece eğitim değil, barajı geçemeyen (örneğin %90 doğruluk altındaki) modellerin canlıya çıkmasını engelleyen katı bir `test.py` kalite kontrol aşaması eklendi.
5. **Canlı E2E (Uçtan Uca) Test Sistemi:** Dağıtım bittikten sonra "Acaba çalıştı mı?" demek yerine, Render sunucusunun uyku modundan çıkmasını akıllı bir döngü ile bekleyen ve canlı API'ye sahte bir resim göndererek sistemin gerçekten çalıştığını doğrulayan yepyeni bir `test_deployed_model.py` betiği yazıldı.

## Proje Dizin Yapisi ve Detayli Aciklamalari

Karmaşık yapay zeka süreçlerini daha kolay yönetebilmek için mimarimizi küçük ve bağımsız betiklere böldük. Projenin genel klasör ağacı şu şekildedir:

```text
circleci-MLOps-pipeline/
│
├── .circleci/
│   └── config.yml              # CI/CD/CT boru hattı konfigürasyonu
├── ml/
│   ├── build.py                # Veri indirme ve ön işleme
│   ├── train.py                # CNN modeli mimarisi ve eğitimi
│   ├── test.py                 # Model başarı (Quality Gate) testi
│   ├── retrain.py              # Gece vardiyası, model kayması (Drift) simülasyonu
│   └── test_deployed_model.py  # Canlı API uçtan uca (E2E) test betiği
├── tools/
│   ├── install.sh              # Bağımlılık kurulum aracı
│   ├── test_build.sh           # Lokal test aracı
│   └── test_retrain.sh         # Lokal retrain test aracı
├── Dockerfile                  # API'nin Docker konteyner reçetesi
├── main.py                     # FastAPI uygulaması (Uç noktalar)
├── Makefile                    # Hızlı komut yöneticisi
└── requirements.txt            # Python bağımlılıkları
```

İşte bu mimarideki her bir dosyanın detaylı görevi:

### Ana Dizin Dosyaları

* **main.py:** Projemizin dış dünyaya açılan kapısıdır. FastAPI kullanılarak yazılmıştır. Sunucu ayağa kalktığında eğitilmiş modeli diskin içinden okuyarak belleğe alır (startup event). Kullanıcılardan gelen kıyafet resimlerini kabul eden, bu resimleri modelin anlayacağı formata çeviren ve tahmin sonuçlarını JSON olarak döndüren uç noktayı (`/predict`) barındırır.
* **Dockerfile:** Projemizin çalışacağı Render bulut sunucusunun reçetesidir. İşletim sistemi olarak hafif bir Python sürümü seçer, gerekli C++ görüntü kütüphanelerini (`libgl1`) kurar, Python paketlerini yükler ve en önemlisi inşanın (build) bir parçası olarak modeli baştan eğitip sunucunun diskine gömer.
* **Makefile:** Terminalde uzun uzun komutlar yazmak yerine, süreçleri kısaltan bir komut yöneticisidir. Sadece `make pipeline` yazarak lokal bilgisayarınızda tüm eğitimi ve testleri tek tuşla başlatmanızı sağlar.
* **requirements.txt:** Projenin ihtiyaç duyduğu kütüphanelerin (TensorFlow, FastAPI, Uvicorn, NumPy vb.) ve bu kütüphanelerin çalışması için gereken sürümlerin listesidir.

### .circleci Klasörü

* **config.yml:** MLOps otomasyonunun beynidir. GitHub'a yeni kod geldiğinde veya gece yarısı zamanlanmış görev (cron job) tetiklendiğinde hangi işlemlerin sırasıyla yapılacağını tanımlar. Tüm CI/CD/CT akışı, barajı geçemeyen modellerin reddedilmesi ve Render webhook'unun tetiklenmesi gibi adımlar burada yönetilir.

### ml Klasörü (Makine Öğrenimi Süreçleri)

* **build.py:** Sürecin ilk adımıdır. İnternetten veya yerel bir kaynaktan verileri indirir, temizler, normalize eder ve bir sonraki adımın kullanabilmesi için NumPy matrisleri (`.npy`) olarak diske kaydeder.
* **train.py:** Modelin inşa edildiği ve öğrenme işleminin gerçekleştiği dosyadır. `build.py` tarafından hazırlanan verileri alır, Evrişimli Sinir Ağı (CNN) katmanlarını oluşturur ve eğitimi gerçekleştirip son modeli diske kaydeder.
* **test.py:** Kalite kontrol (Quality Gate) aşamasıdır. Eğitilmiş modeli daha önce hiç görmediği "Kör Test" verileriyle sınar. Eğer modelin doğruluk oranı bizim belirlediğimiz eşiğin altındaysa, bir hata fırlatarak CircleCI sürecini acımasızca durdurur ve kötü modelin canlıya çıkmasını engeller.
* **test_deployed_model.py:** Sistem başarıyla canlıya alındıktan sonra çalışan uçtan uca (E2E) test betiğidir. Render sunucusunun uyanmasını sabırla bekler. Sunucu uyanınca sahte bir resim oluşturup canlı API'ye HTTP POST isteği atar ve sistemin çalıştığını doğrular.
* **retrain.py:** Sürekli Eğitim (Continuous Training - CT) simülasyonudur. Gerçek dünyada veriler zamanla değişir (Data Drift). Bu dosya, eski verilere bilerek gürültü (noise) ekleyerek verinin bozulmasını simüle eder ve modeli bu yeni verilerle baştan eğitir.

### tools Klasörü (Geliştirici Araçları)

* **install.sh:** Geliştiricinin lokal bilgisayarında veya CI ortamında temiz bir Python sanal ortamı (venv) oluşturup gerekli kütüphaneleri kuran bir bash betiğidir.
* **test_build.sh & test_retrain.sh:** Geliştiricinin, kodları GitHub'a göndermeden önce makine öğrenimi süreçlerini kendi bilgisayarında güvenle test etmesini sağlayan betiklerdir.

## MLOps Mimarisinin Isleyisi (CI/CD/CT Boru Hatti)

Projede herhangi bir değişiklik yapılıp GitHub'a gönderildiğinde (`git push`), CircleCI otomatik olarak devreye girer ve sırasıyla aşağıdaki adımları hatasız geçmek zorundadır:

1. **Install & Build:** Bağımlılıklar kurulur, veri seti indirilir ve eğitime hazır matrislere dönüştürülür.
2. **Train:** Derin öğrenme modeli bu verilerle eğitilir.
3. **Quality Gate (Test):** Eğitilen model kör test verileriyle sınava sokulur. Barajı geçemezse süreç durdurulur.
4. **Deploy:** Model barajı geçerse, Render platformuna bir webhook sinyali gönderilir. Render projeyi çekip Dockerfile üzerinden inşa eder ve FastAPI sunucusunu yayına alır.
5. **Test Deployment (E2E):** Süreç bittikten sonra, CircleCI canlı API'ye bir ağ isteği atarak sistemin tam anlamıyla ayağa kalktığını teyit eder.

## Canli API Kullanimi

Bu repo aracılığıyla otomatik olarak eğitilen ve dağıtılan model, Render üzerinde barındırılmaktadır. Canlı sisteme dışarıdan bir resim göndererek yapay zekanın tahmin yapmasını sağlayabilirsiniz.

### Tahmin Uç Noktası (POST /predict)

API'nin tahmin yapabilmesi için bir resim dosyası (örneğin bir ayakkabı veya tişört resmi) göndermeniz gerekmektedir.

Ornek cURL Istegi:

```bash
curl -X 'POST' \
  '[https://circleci-mlops-pipeline.onrender.com/predict](https://circleci-mlops-pipeline.onrender.com/predict)' \
  -H 'accept: application/json' \
  -H 'Content-Type: multipart/form-data' \
  -F 'file=@ornek_kiyafet.png;type=image/png'
```

Ornek JSON Yaniti:

```json
{
  "tahmin": "Sneaker",
  "sinif_kodu": 7,
  "guven_skoru_yuzde": 98.45
}
```

## Lokalde Calistirma

Projeyi bilgisayarınıza indirip kodları kurcalamak isterseniz, kök dizinde bulunan `Makefile` dosyası sayesinde işlemleri saniyeler içinde yapabilirsiniz.

1. Kütüphaneleri ve sanal ortamı kurun:

```bash
make install
```

2. Veriyi hazırlayın, eğitin ve lokal testten geçirin:

```bash
make pipeline
```

3. Geliştirme sunucusunu (API) kendi bilgisayarınızda başlatın:

```bash
make api
```

Tarayıcınızda `http://localhost:8000/docs` adresine giderek, FastAPI'nin sunduğu otomatik Swagger arayüzü üzerinden hiç kod yazmadan API'nizi test edebilir ve resim yükleyebilirsiniz.