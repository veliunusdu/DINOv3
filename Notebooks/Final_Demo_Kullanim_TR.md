# Final demo notebook kullanımı

Dosya: `DINOv3_Adaptive_Zoom_Final_Demo.ipynb`.

## Yeni Colab oturumu

1. Notebook'u PC'ye kaydet, Colab'da Dosya → Notebook yükle ile aç; T4 GPU seç.
2. 1. hücre bağımlılıkları kurar.
3. 2. hücrede checkpoint içeren tam deney ZIP'ini yükle.
4. 3. hücrede Hugging Face girişini tamamla; backbone ve head geri yüklenir.
5. VOC örneği için 4. hücreyi çalıştır. SOURCE="upload" seçersen veri hücresi gerekmez.
6. 5. hücre demo fonksiyonlarını tanımlar; 6. hücre seçilen görüntüyü işler ve ZIP indirir.
7. Oturum sonunda 7. hücre checkpoint ve sonuçların tamamını indirir. Notebook'u ayrıca `.ipynb` olarak sakla.

## Model ve veri zaten hazırsa

Yeni notebook'un yalnızca `demo-helpers` ve `demo-run` kod hücrelerini mevcut oturumda çalıştırmak yeterlidir. Setup, ZIP recovery, model veya veri yükleme tekrarına gerek yoktur.

## Parametreler

Demo hücresinin başında SOURCE, IMAGE_INDEX, K, CROP_SIZE, STRIDE ve OVERLAP_THRESHOLD bulunur. SOURCE="upload" kendi fotoğrafını seçtirir. EXIF yönü düzeltilir; fotoğrafta GT yoksa mIoU/pixel accuracy hesaplanmaz. Model VOC'nin 21 sınıfını tahmin eder.

Varsayılan SOURCE="voc", IMAGE_INDEX=9, K=3, CROP_SIZE=112, STRIDE=28, OVERLAP_THRESHOLD=0.30. NMS sonrasında gerçek crop sayısı K'dan az olabilir; gerçek sayı ve inference bütçesi kaydedilir. Padding 0, aynı model, örtüşen crop'larda local logits ortalaması kullanılır.

## Çıktılar

Demo ZIP'i orijinal görüntü, varsa GT, baseline/final sınıf-ID maskeleri ve renkli maskeler, entropy/coverage/global logits, crop fotoğrafları, crop logits ve native prediction, koordinatlar/skorlar/parametreler/metrics JSON, local prediction montajı ve yedi panelli final görselini içerir.

Sınıf-ID PNG'leri metrik/kayıt içindir; renkli PNG'ler görselleştirmedir. Yedi panelin sırası original, GT, global prediction, entropy, crop kutuları, local predictions, final segmentation. GT ignore=255 gri gösterilir.

## Doğrulama sınırı

Yerelde yedi hücre compile kontrolünden geçti; koordinat dönüşümü ve box IoU kontrolleri çalıştırıldı. Sonrasında kullanıcı Colab'da birleşik demoyu çalıştırdı: index 9 baseline=0.8301 ve K=3=0.9512 ile önceki referans eşleşti. Index 10 için baseline=0.4658, K=3=0.3208 çıktı; başarı ve failure yedi panelli görselleri incelendi. 8 Ekim'de ayrıca özel JPG görüntüsü (736×1041) upload edilerek K=3/3 ve dört geçişle inference/sonuç ZIP'i üretildi; GT olmadığı için metrik hesaplanmadı. Özel görüntünün yedi paneli kullanıcı ekran görüntüsünden incelendi: insan/arka plan tahmini ve bacak çevresindeki keskin geçişler görüldü. GT olmayan bu illüstrasyon üzerinden doğruluk artışı veya gerçek fotoğraflara genelleme iddia edilmez. Notebook index 9 için K=1/3/5 sonuçlarına 0.9214/0.9512/0.9597 referans kontrolleri içerir. Diğer görüntüler için sabit bir skor beklenmez.

Bu notebook eğitim veya büyük değerlendirmeleri çalıştırmaz. Demo sonucu bir görüntünün performansıdır; tüm validation başarısı yerine sunulmamalıdır. Büyük sonuçların kayıtları yüklenen tam deney paketinde korunur.
