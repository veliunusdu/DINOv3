# Yöntem karşılaştırması CSV açıklaması

Bu tablo, tamamlanmış deneylerin `Docs/DINOv3_Final_Deney_Raporu_TR.md` dosyasındaki raporlanmış değerlerinden üretildi. Yeni inference yapılmadı. Ham kayıtların tam hassasiyetli değerleri yerine raporda gösterilen yuvarlatılmış değerler kullanıldı; hesaplanan fark ve oranlar da bu hassasiyete tabidir.

- **mIoU:** 1.399 VOC2012 validation görüntüsünde, orijinal çözünürlükte pooled mIoU. Background dahil, ignore=255. 50 development görüntüsü hariçtir.
- **Delta mIoU:** baseline'a göre fark, **yüzde puan**. `(mIoU − baseline mIoU) × 100`.
- **Inference Time:** aynı 100 pilot görüntüde, aynı Tesla T4 oturumunda ölçülen **ortalama ms/görüntü**. Bu 100 görüntü, ana 1.399 görüntünün içindedir. mIoU ve süre kolonlarının örneklem büyüklükleri farklıdır.
- **Relative Cost:** yöntemin ortalama süresi / baseline ortalama süresi. Boyutsuz süre oranıdır; FLOPs veya GPU ücret oranı değildir.
- **Extra Inference:** yöntemin toplam backbone geçişi − baseline'ın 1 geçişi. Adaptive/Random K=3 için 3, K=1 için 1, sliding için 24. Sliding 25 yerel inference yapar, ayrıca global pass çalıştırmaz; bu kolondaki 24 ek crop sayısı değildir.
- **K:** işlenen yerel crop sayısı. Baseline için 0, sliding için tüm 25 aday, adaptive/random için seçilen K. Sliding Top-K selector kullanmaz.
- **Crop Size:** 224×224 seçim haritasındaki pencere kenarı, piksel. Yerel crop'lar orijinal RGB görüntüden alınır ve 224×224 model girişine dönüştürülür. Baseline'da crop olmadığı için `n.a.` yazıldı.
- Random üç seed ortalaması, 42/43/44 sonuçlarının aritmetik ortalamasıdır; prediction ensemble değildir. K, geçiş sayısı ve maliyet her seed'in ayrı çalışmasına aittir, üç çalışmanın toplamı değildir.

Aynı checkpoint, model ön işleme ve native-resolution değerlendirme kodu kullanıldı. Adaptive/random seçiminde crop=112, stride=28, kutu IoU eşiği=0,30 ve K=3 kullanıldı. Aynı geçiş sayısı, aynı crop birleşim alanını garanti etmez. Sliding bütün 25 crop'u sıralı işler.

Süreye preprocessing, inference, CPU logits transferi, entropy/selector gereken yöntemlerde, resize, fusion ve argmax dahildir. Dosya decode, GT, metrik hesabı ve kayıt hariçtir. Sliding optimize/batched değildir. Warm-up ve CUDA synchronization uygulandı.

CSV virgülle ayrılmış, UTF-8 BOM kodlamalıdır; ondalık ayırıcı noktadır. Excel otomatik sütunlara ayırmazsa **Veri → Metin/CSV'den** yoluyla içe aktar ve ayırıcı olarak virgül seç.

Sonuç yorumu: K=3 − random ortalaması +0,80 puan [%95 GA: +0,29; +1,33]. K=3 − baseline +0,87 puan [−0,14; +1,96], dolayısıyla baseline'a üstünlük kesinleşmedi. K=3 − sliding −1,05 puan [−1,51; −0,58]. Süre bootstrap'ında sliding/K=3 oranı 3,46× [3,33; 3,62]. Güven aralıkları CSV'deki yuvarlatılmış tablodan yeniden hesaplanmadı; kayıtlı bootstrap sonuçlarıdır.
