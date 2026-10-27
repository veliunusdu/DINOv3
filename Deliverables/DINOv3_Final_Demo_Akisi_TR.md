# DINOv3 final demo akışı

Hazırlanma: 27 Ekim 2026. Hedef süre: **8 dakika**, kabul aralığı 5–10 dakika. Bu belge gösterim sırasını ve konuşma metnini tanımlar. Kesintisiz prova henüz bu belgeyle yapılmış sayılmaz.

## Gösterimden önce

- [Final demo notebook'unu](../Notebooks/DINOv3_Adaptive_Zoom_Final_Demo.ipynb) aç; GPU, bağımlılıklar, config, checkpoint, model, VOC ve yardımcılar hazır olsun. Kurulum, giriş ve yaklaşık 2 GB veri indirmeyi sunum süresine dahil etme.
- Final config'i kullan: K=3, crop=112, stride=28, minimum entropy=0, overlap IoU=0,30, padding=0. Demo hücresinde `SOURCE="voc"`, `IMAGE_INDEX=9` seç.
- Gösterimden önce index 9 inference'ını bir kez çalıştır; indirme iletişimlerini kapat ve sonuç görselini hazır tut. Demo sırasında istenirse yalnızca tek görüntü hücresini yeniden çalıştır. Hücreye elle model/processor/run_dir tanımlama.
- [Index 9 sekiz panelli başarı görseli](Final_Visuals/success_index_9_8_panels.png), [index 10 failure görseli](Final_Visuals/failure_index_10_8_panels.png), [kalite–maliyet grafiği](Quality_Cost/quality_cost.png) ve [benchmark CSV](DINOv3_Method_Comparison.csv) ayrı olarak açık veya kolay erişilebilir olsun.
- [Temiz Colab test raporunu](../Reproducibility/clean_colab_verification_2026-10-20_TR.md) hazır tut. 20 Ekim testi kurulum/restore/inference/export'u iki örnekte doğruladı; bu 8 dakikalık anlatımın kesintisiz provası değildir.

## Açılış — 00:00–00:20

**Söyle:** “Bu proje, görüntünün tamamını tekrar tekrar işlemek yerine ek hesaplamayı modelin belirsiz bulduğu bölgelere yönlendiriyor. Dondurulmuş DINOv3 backbone ve eğitilmiş segmentasyon head'i kullanıyorum. Bölge seçiminin ne kazandırdığını ve hangi örneklerde zarar verebildiğini göstereceğim.”

## On adımlık gösterim — 00:20–06:40

| Adım | Zaman | Ekranda göster | Söyle / vurgula |
|---|---|---|---|
| 1. Input image | 00:20–00:40 | Index 9 orijinal RGB görüntü | “Giriş görüntüsü 500×375. Global giriş 224×224'e dönüştürülüyor; crop'ları daha sonra orijinal RGB'den alacağız.” |
| 2. Baseline segmentation | 00:40–01:15 | Baseline paneli; GT'yi değerlendirme referansı olarak işaret et | “Tek global geçişin tahmini bu. GT modelin seçim girdisi değil; yalnızca doğruluk kontrolü için kullanılıyor. Bu görüntünün baseline mIoU'su 0,8301.” |
| 3. Entropy map | 01:15–01:50 | Normalize entropy paneli | “Sınıf olasılıklarından hesaplanan entropy log(21) ile normalize ediliyor. Açık renk daha yüksek belirsizlik gösterir; mutlaka hata veya düzeltilebilir hata demek değildir.” |
| 4. Region Selector | 01:50–02:25 | Config veya çıktıdaki aday/score bilgisi | “112×112 pencereler, 28 piksel stride ile 224×224 haritada 25 aday oluşturuyor. Ortalama entropy'ye göre sıralayıp IoU filtresi uyguluyoruz.” |
| 5. Selected crops | 02:25–03:00 | Üç sıralı bounding box | “K=3 en fazla üç bölge demek. Burada üçü de seçildi. IoU eşiği kutuların tamamen ayrık veya farklı nesnelere ait olmasını garanti etmiyor.” |
| 6. Second-pass predictions | 03:00–03:40 | Crop 1/2/3 yerel prediction panelleri | “Her kutu native koordinatlara çevrilip orijinal RGB'den kesiliyor. Aynı model crop'u 224×224 girişle yeniden işliyor. Yeni eğitim yapılmıyor.” |
| 7. Fusion | 03:40–04:15 | Yerel paneller ile final panel arasında karşılaştır | “Yerel logits kendi native crop boyutuna döndürülüyor. Crop dışında global logits korunuyor. Çakışan yerlerde yerel logits'lerin ortalaması alınıyor; en son sınıf argmax'ı uygulanıyor.” |
| 8. Final output | 04:15–04:50 | Final fused panel | “Bu, üç yerel geçişin birleştirilmiş sonucu. Toplam bir global ve üç crop olmak üzere dört backbone geçişi var.” |
| 9. Baseline vs Adaptive | 04:50–05:40 | Baseline, GT, error map ve final panel | “Bu örnekte image mIoU 0,8301'den 0,9512'ye çıktı; fark +12,12 yüzde puan. Error map baseline hatalarını gösteriyor; final error map veya uncertainty haritası değil.” |
| 10. Benchmark tablosu | 05:40–06:40 | Aşağıdaki tablo ve kalite–maliyet grafiği | “Tek görüntü genel başarı kanıtı değil. Ana kalite 1.399 görüntüde, süre ayrı ve aynı 100 görüntülük T4 grubunda ölçüldü. Grafik bu iki kapsamı ayrı panellerde gösteriyor.” |

Bu adımlar tek pipeline'ın açıklamasıdır; anlatımın her adımında ayrı inference başlatılmaz. Canlı notebook'un çıktısı yedi panellidir; hazırlanan sekiz panelli final görsel ayrıca baseline error map içerir. Gösterim için sekiz panelli görselde ilgili panele yaklaşarak aynı sırayı izle.

### Gösterilecek kısa benchmark tablosu

| Yöntem | Ana pooled mIoU — 1.399 görüntü | Ortalama ms — aynı 100 görüntü | Geçiş |
|---|---:|---:|---:|
| Baseline | 0,7133 | 79,2 | 1 |
| Adaptive K=1 | 0,7124 | 110,8 | 2 |
| Adaptive K=3 | 0,7220 | 147,5 | 4 |
| Random K=3, üç seed mean | 0,7140 | 147,8 | Her seed için 4 |
| Sliding-window | 0,7325 | 510,3 | 25 |

**Söyle:** “K=3, ana değerlendirmede random K=3 ortalamasından +0,80 puan yüksek; %95 aralık +0,29 ile +1,33. Baseline'a farkın aralığı sıfırı içerdiği için kesin üstünlük söylemiyorum. Sliding daha doğru, fakat ölçülen seri uygulamada K=3'ten 3,46 kat daha yavaş.”

Random mean prediction ensemble'ı değildir. Relative cost, ortalama sürenin baseline süresine oranıdır; K=3 için yaklaşık 1,86×. FLOPs veya dört geçişten çıkarılmış 4× süre iddiası değildir. Aynı 100 pilotta K=1'in K=3'ten hem daha hızlı hem daha yüksek mIoU verdiğini gerektiğinde grafikten göster; K=3 her koşulda optimum değildir.

## Failure gösterimi — 06:40–07:25

[Index 10 görselini](Final_Visuals/failure_index_10_8_panels.png) aç; baseline, seçilmiş kutular ve final maskeyi göster.

**Söyle:** “Burada yüksek entropy alanını bulduk ama yerel tahmin daha iyi olmadı. Image mIoU 0,4658'den 0,3208'e düştü. Crop'lar sofa çevresinde yoğunlaşıyor; yerel tahmin sınıfları parçalıyor ve fusion bu hataları taşıyor. Dolayısıyla doğru bölge seçimi ile doğru refinement aynı problem değil.”

Ölçek ve bağlam kaybını kesin neden olarak sunma; bunlar bu örneklerden çıkan olası açıklamalardır. GT'de etiketlenmeyen gerçek bir nesnenin tahmini de benchmark bakımından yanlış pozitif sayılabilir.

## Kapanış — 07:25–08:00

**Söyle:** “Görüntüden entropy, Top-K seçim, ikinci inference, native koordinat dönüşümü ve fusion'a kadar akış otomatik. Katkı, aynı crop bütçesinde random'a göre ölçülmüş seçim avantajı. Karşılığında ek süre ve bazı örneklerde doğruluk kaybı var. Sonraki araştırma sorusu, yerel tahmine ne zaman güvenip global tahmini ne zaman korumamız gerektiği.”

Bu son soru future work'tür; confidence-gated fusion tamamlanmış sonuç olarak sunulmaz.

## Süre daralırsa: 5 dakikalık sürüm

- 00:00–00:20: Açılış.
- 00:20–03:20: On adımın ilk dokuzunu aynı index 9 görseli üzerinden anlat; kurulum ve uzun config ekranını atla, yöntem adımlarını koru.
- 03:20–04:20: Benchmark tablosu ve örneklem farkı.
- 04:20–04:50: Index 10 failure.
- 04:50–05:00: Kapanış.

## Colab veya internet aksarsa

1. Bir kısa denemeden sonra sunumda paket kurma, veri indirme veya oturum hata ayıklamaya girişme.
2. “Canlı bağlantı şu an kullanılamıyor; 20 Ekim'de temiz GPU oturumundan alınmış ve kontrol edilmiş çıktılara geçiyorum” de.
3. Aynı on adımı index 9'un kayıtlı görseli üzerinden göster; index 10 ve benchmark ile devam et.
4. Gösterilen şeyin kayıtlı çıktı olduğunu belirt; o anda yeni inference yapılmış gibi anlatma. [Temiz test raporu](../Reproducibility/clean_colab_verification_2026-10-20_TR.md) ve [çıktı ZIP doğrulaması](../Reproducibility/clean_colab_verification_2026-10-20.json) erişilebilir olsun.

## Prova kontrol listesi

- [ ] Model ve veri açılıştan önce hazır; yalnızca doğru checkpoint/config kullanılıyor.
- [ ] Index 9 giriş, baseline, entropy, selector, crop, second pass, fusion ve final sırası eksiksiz gösterildi.
- [ ] Başarı ve failure metrikleri görüntü mIoU'su olarak tanımlandı.
- [ ] 1.399 görüntülük kalite ve 100 görüntülük süre kapsamları ayrı söylendi.
- [ ] Baseline'a kesin üstünlük, sliding'e eşdeğer doğruluk veya GPU bellek tasarrufu iddiası kurulmadı.
- [ ] Canlı çıktı ve kayıtlı yedek açıkça ayrıldı.
- [ ] Kronometre ile tek seferde 5–10 dakikada tamamlandı.

Prova sonrasında gerçek süreyi ve aksayan adımı kaydet: **Tarih: ____ / Süre: ____ / Aksayan adım: ____ / Sonraki düzeltme: ____**.
