# Kıyafet Tanıma Laboratuvarı

Eğitilmiş Fashion-MNIST CNN'ini tarayıcıda gerçekten çalıştıran fotoğraf yükleme demosu. Eğitim projesi: https://github.com/gullcan/fashion-mnist-cnn-study.

## Akış

1. Hazır doğrulama örneği seç veya PNG/JPEG/WebP ürün fotoğrafı yükle.
2. Fotoğrafta gri tonlama, açık/koyu zemin seçimi, isteğe bağlı ortalama ve oranı koruyan28×28 dönüşüm uygulanır. Hazır Fashion-MNIST örnekleri bu dönüşümleri atlar; özgün pikselleri kullanır.
3. Kaydedilmiş57.190 katsayıyla iki convolution, ReLU, average pooling ve iki Linear katmanı çalışır.
4. İlk üç sınıf skoru, modelin gerçekten aldığı28×28 girdi ve ilk kanalın ara haritaları gösterilir.

Fotoğraf dosyası tarayıcı belleğinde işlenir. Sunucuya fotoğraf gönderen bir tahmin API'si yoktur; sayfa yalnızca kendi model/arayüz varlıklarını yükler. Model ağırlıkları indirilebilir statik varlıklardır.

## Öğrenme ve kullanım sınırı

Model yalnızca Fashion-MNIST'in10 sınıfını tanır. Gösterilen%91.08 doğruluk, seçilmiş seed42 modelinin aynı resmî Fashion-MNIST testinin yeniden kullanımındaki sonucudur; yüklenen günlük fotoğrafların doğruluğu değildir. Tek ürün, sade ve kontrastlı zemin, benzer çekim açısı daha uygun bir giriş sağlar. Baskı, renk, marka, beden veya çoklu nesne tespiti değerlendirilmedi.

Fotoğraf hazırlama kodu basit parlaklık/bounding-box sezgisidir; nesne segmentasyonu veya öğrenilmiş arka plan silme değildir. Şeffaf PNG'de beyaz ürün varsa koyu zemin seçimini denemek gerekebilir. Kullanıcı model girdisini ve dönüşüm uyarılarını görebilir. Model alakasız bir görselde bile on sınıftan birine yüksek skor verebilir; skor kalibre doğruluk garantisi değildir.

## Kaynak kod

| Dosya | Görev |
|---|---|
| `dist/app.mjs` | Dosya/örnek seçimi, görünüm ve gerçek tahmin akışı |
| `dist/preprocess.mjs` | Saydam ve test edilebilir fotoğraf dönüşümleri |
| `dist/inference.mjs` | Convolution/aktivasyon/pooling/Linear/softmax ileri hesabı |
| `dist/model/model.json` | Şekiller, sınıf sırası, model kimliği ve kaynak ölçümler |
| `dist/model/weights.bin` | Değiştirilmeden dışarı aktarılan float32 katsayılar |
| `export_model.py` | PyTorch checkpoint'inden tekrar dışa aktarma ve bağımsız referans üretimi |

## Yerelde açmak

Bu site için Python sunucusu yalnızca statik dosyaları servis eder; tahmin tarayıcıda çalışır. Ana deponun kökünden:

```powershell
python -m http.server 8000 --bind 127.0.0.1 --directory web/dist
```

Ardından http://127.0.0.1:8000 adresini aç. Python web framework'ü, paket kurulumu veya Node build adımı gerekmez.

Modeli yeniden dışarı aktarmak için, eğitim projesinin ortamında:

```powershell
python web/export_model.py --project . --out runs/web-export/dist --references runs/web-export/references
```

Proje ortamını etkinleştirip paketi kurmuş olmalısın. Dışa aktarıcı ortak `fashion_cnn.models.build_cnn` tanımını kullanır. Bu komut eğitim yapmaz; yalnızca sabit model ve train=True doğrulama kaynaklarını okur. İlk kaydedilmiş doğrulama örneği her sınıftan alınır; doğruluğuna göre örnek seçilmez. Yukarıdaki yeni çıktı yolları depodaki yayın varlıklarını korur.

## Kontroller

```powershell
node web/tests/test-runtime.mjs
node web/tests/test-preprocess.mjs
```

100 dengeli, sabit doğrulama girdisinde JavaScript ve bağımsız PyTorch CPU referansı100/100 aynı top1 tahmini verdi. En büyük olasılık farkı yaklaşık3.43e-7, logit farkı4.77e-6, aktivasyon farkı6.68e-6; ilan edilmiş mutlak tolerans1e-4. Bu sayılar runtime eşleşmesini gösterir;100/100 sınıflandırma doğruluğu veya gerçek fotoğraf başarısı değildir. Kaynak checkpoint SHA256:

`013745dcd66e4d5c462ba64607faed70831ea7e9caa27721afcdc34693b245f2`

`references/runtime-validation.json` ölçümleri, diğer referans dosyaları PyTorch kontrol girdilerini taşır; bunlar statik yayına dahil değildir. Ön işleme testleri boyutlar, açık/koyu/otomatik zemin, alfa, oran, sınır kutusu, boş/düşük kontrast ve görüntü-tensör eşleşmesini kontrol eder.

## Veri ve lisans

Hazır örnekler Zalando Research Fashion-MNIST verisinden türetilmiştir; MIT bildirimi `dist/examples/LICENSE.txt` içindedir. Özgün kaynak: https://github.com/zalandoresearch/fashion-mnist. Proje, LeNet'ten esinlenen küçük bir CNN uyarlamasıdır; LeNet-5'in birebir tekrarı değildir. Modelin eğitim/değerlendirme protokolü ve sınırlılıkları ana GitHub deposunda bulunur.
