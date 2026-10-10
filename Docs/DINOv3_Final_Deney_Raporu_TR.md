# DINOv3 Adaptive Zoom — deney raporu

Güncelleme: 8 Ekim 2026. Deney ortamı: Google Colab, Tesla T4. Bu belge mevcut sonuçların birleştirilmiş raporudur; projenin tüm teslimlerinin tamamlandığı anlamına gelmez.

## Araştırma sonucu

Entropy tabanlı Top-K crop seçimi, dondurulmuş DINOv3 backbone ve eğitilmiş segmentation head ile otomatik bir adaptive zoom akışına entegre edildi. K=3, 1.399 VOC2012 validation görüntüsünde aynı crop sayısı ve fusion kullanan üç random seed ortalamasından +0,80 yüzde puan daha yüksek pooled mIoU verdi; %95 paired bootstrap aralığı [+0,29; +1,33].

Adaptive K=3 mIoU 0,7220, baseline 0,7133 ve sliding-window 0,7325 oldu. Baseline'a karşı farkın aralığı sıfırı içeriyor; sliding daha yüksek doğruluk verdi. Ayrı 100 görüntülük aynı oturum benchmark'ında K=3, sliding'den 3,46 kat hızlıydı; %95 aralık [3,33; 3,62].

Desteklenen sonuç: kullanılan sıralı uygulamada otomatik bölge seçimi, aynı K değerindeki random seçime göre doğruluk katkısı sağladı ve dense sliding'e karşı daha düşük süreyle, daha düşük doğruluk sundu. Baseline'a kesin üstünlük, sliding ile eşdeğer doğruluk, GPU bellek tasarrufu veya her küçük nesnenin yakalandığı iddiaları desteklenmiyor.

## Model ve otomatik akış

| Bileşen | Konfigürasyon |
|---|---|
| Backbone | `facebook/dinov3-vits16-pretrain-lvd1689m` |
| Backbone revision | `114c1379950215c8b35dfcd4e90a5c251dde0d32` |
| Özellik boyutu / patch / register | 384 / 16 / 4 |
| Head | 21 sınıflı 1×1 convolution; 5 epoch eğitilmiş checkpoint |
| Backbone güncellemesi | Dondurulmuş, eval modunda |
| Global ve crop model girişi | 224×224; bicubic resize, processor mean/std normalization |
| Entropy | Softmax sınıf entropy'si / log(21) |
| Selector | Ortalama crop entropy'si, azalan sıralama, greedy IoU filtresi |
| Varsayılan seçim | K=3, crop=112, stride=28, IoU eşiği=0,30, minimum entropy=0 |
| Aday sayısı | 224×224 haritada 25 |
| Koordinat dönüşümü | Orijinal görüntüye ölçekleme; başlangıç floor, bitiş ceil, bounds kontrolü |
| Second pass | Crop orijinal RGB görüntüden alınır; aynı backbone/head ile yeniden inference |
| Fusion | Crop birleşimi içinde local logits; örtüşmelerde local logits ortalaması; dışında global logits |
| Padding | 0 |
| Değerlendirme | Native çözünürlük, bilinear logits resize, ignore=255, background dahil |

Seçim inference sırasında GT kullanmaz. GT yalnızca development grubunu oluşturmak, oracle referansı ve metrik hesaplamak için kullanıldı. Minimum entropy eşiği veya NMS, gerçek seçim sayısını K'dan azaltabilir. Hiç crop seçilmezse baseline korunur; inference bütçesi gerçek crop sayısı + 1'dir.

## Eğitim ve örneklemler

Head, VOC2012 train grubunun 1.464 görüntüsünde beş epoch eğitildi; backbone donduruldu. Checkpoint tekrar yüklenerek head eşleşmesi ve referans inference çıktısı doğrulandı. Sonraki deneylerde yeniden eğitim yapılmadı.

| Grup | Kullanım | Yorum sınırı |
|---|---|---|
| 50 development görüntüsü | Oracle, selector analizi, crop/padding/fusion/K/threshold ablation | GT ile küçük hedef içeren örnekler seçildi; genel validation dağılımını temsil etmez. |
| 20 development görüntüsü | K=1 ve K=3 manuel görsel kontrol | Aynı development grubunun ilk 20 örneği; ayrı test değildir. |
| 100 pilot görüntüsü | Doğruluk pilotu ve süre ölçümü | Development ile çakışmaz; sonuçları önceden görülmüştür. |
| 1.399 görüntü | Büyük doğruluk değerlendirmesi ve bootstrap | VOC val 1.449 − 50 development. Önceden görülen 100 pilotu içerir; dokunulmamış test seti değildir. |

Pooled mIoU, görüntülerin confusion matrix toplamından hesaplandı; image mIoU ortalaması değildir. Random mean, ayrı seed'lerin mIoU ortalamasıdır; prediction ensemble'ı değildir. Image mIoU'da GT'de olmayan false-positive sınıflar da union üzerinden ortalamaya girer.

Eski 1.449 görüntülük 224×224 GT değerlendirmesi 0,7104 verdi. Native/1.399 sonuçlarla doğrudan karşılaştırılmaz. Eski belgelerdeki 0,7147 değerinin bu checkpoint ile birebir tekrarlandığı iddia edilmez.

## Ana doğruluk değerlendirmesi — 1.399 görüntü

| Yöntem | Pooled mIoU | Backbone geçişi / görüntü |
|---|---:|---:|
| Baseline | 0,7133 | 1 |
| Adaptive K=1 | 0,7124 | 2 |
| Adaptive K=3 | 0,7220 | 4 |
| Random 42 K=3 | 0,7148 | 4 |
| Random 43 K=3 | 0,7115 | 4 |
| Random 44 K=3 | 0,7157 | 4 |
| Random K=3 üç seed mean | 0,7140 | Her seed için 4 |
| Sliding | 0,7325 | 25 |

Random K=3 aynı aday listesi, crop boyutu, K ve IoU kısıtıyla seçildi. Aynı geçiş sayısı aynı crop birleşim alanını garanti etmez. Sliding bütün 25 crop'u sıralı işler; global pass kullanmaz ve örtüşmelerde local logits ortalaması alır.

5.000 paired image bootstrap; farklar yüzde puan:

| Karşılaştırma | Fark | %95 aralık |
|---|---:|---|
| K=3 − baseline | +0,87 | [−0,14; +1,96] |
| K=3 − K=1 | +0,96 | [+0,35; +1,55] |
| K=3 − random K=3 mean | +0,80 | [+0,29; +1,33] |
| K=3 − sliding | −1,05 | [−1,51; −0,58] |
| K=1 − baseline | −0,09 | [−0,89; +0,76] |
| K=1 − random K=1 mean | −0,06 | [−0,70; +0,60] |
| K=1 − sliding | −2,01 | [−2,79; −1,24] |

Aralıklar bu görüntü örneklemesindeki belirsizliği ölçer. Random seed'ler sabit üç seed'dir; seed dağılımı yeniden örneklenmedi. Karşılaştırmalar araştırma analizleridir; ayrı doğrulama seti veya çoklu karşılaştırma düzeltmesi olarak sunulmaz.

## Son süre benchmark'ı — aynı 100 görüntü, aynı oturum

| Yöntem | Bu 100 görüntüde mIoU | Mean ms | Median ms | P95 ms | Peak GPU MB | Passes |
|---|---:|---:|---:|---:|---:|---:|
| Baseline | 0,6506 | 79,2 | 65,5 | 127,6 | 98,3 | 1 |
| Adaptive K=1 | 0,6750 | 110,8 | 98,3 | 171,4 | 98,3 | 2 |
| Adaptive K=3 | 0,6702 | 147,5 | 133,0 | 216,5 | 98,3 | 4 |
| Random 42 K=3 | 0,6772 | 147,7 | 132,8 | 224,2 | 98,3 | 4 |
| Random 43 K=3 | 0,6794 | 149,2 | 133,5 | 220,4 | 98,3 | 4 |
| Random 44 K=3 | 0,6684 | 146,4 | 134,3 | 222,6 | 98,3 | 4 |
| Sliding | 0,6732 | 510,3 | 472,0 | 698,4 | 98,3 | 25 |

Warm-up ve CUDA synchronization kullanıldı; yöntemler bağımsız inference yaptı, görüntü başına çalışma sırası değiştirildi. 700 prediction'ın confusion matrix'i önceki kayıtlarla eşleşti. Preprocessing, inference, CPU logits transferi, selector, resize, fusion ve argmax ölçüme dahil; dosya decode, GT, metrik ve kayıt dışarıda. GPU MB, peak allocated bellek ölçümüdür; reserved/CPU bellek değildir.

5.000 paired latency bootstrap; oranlar ortalama sürelerin oranıdır:

| Karşılaştırma | Tahmin | %95 aralık |
|---|---:|---|
| Sliding / K=3 | 3,46× | [3,33; 3,62]× |
| K=3 / baseline | 1,86× | [1,77; 1,95]× |
| K=3 / K=1 | 1,33× | [1,26; 1,39]× |
| K=3 / random K=3 mean | 1,00× | [0,97; 1,02]× |
| K=3 − random K=3 mean | −0,31 ms | [−4,51; +3,55] ms |
| K=3 süre azalması vs sliding | %71,10 | [%69,95; %72,35] |

Adaptive ve random K=3 arasında belirgin süre farkı gösterilmedi. Bu aralıklar aynı oturumdaki görüntü değişkenliğidir; tekrar oturum/GPU değişkenliği değildir. Optimize veya batched sliding ölçülmedi. Önceki oturumların 3,35×/4,93× oranları bu son tabloyla karıştırılmaz.

100 pilotta random K=3 mean mIoU yaklaşık 0,6750 ve adaptive K=3 0,6702; 1.399 görüntüde adaptive daha yüksek. Aynı şekilde 100 pilotta K=1, K=3'ten daha yüksek mIoU verdi. Sonuç alt gruplara göre değişebilir.

## Sınıf bazlı sonuçlar — 1.399 görüntü

| Sınıf | Baseline | Adaptive K=3 | Random K=3 mean | Sliding | K=3 − baseline (puan) |
|---|---:|---:|---:|---:|---:|
| background | 0,9252 | 0,9322 | 0,9307 | 0,9363 | +0,69 |
| aeroplane | 0,7753 | 0,8398 | 0,8256 | 0,8581 | +6,45 |
| bicycle | 0,3898 | 0,5212 | 0,4818 | 0,5409 | +13,14 |
| bird | 0,8064 | 0,8683 | 0,8608 | 0,8877 | +6,19 |
| boat | 0,6972 | 0,7127 | 0,6912 | 0,7095 | +1,55 |
| bottle | 0,7404 | 0,6774 | 0,6972 | 0,6920 | −6,30 |
| bus | 0,8813 | 0,8548 | 0,8326 | 0,8564 | −2,65 |
| car | 0,8067 | 0,7946 | 0,7945 | 0,8085 | −1,21 |
| cat | 0,8681 | 0,8959 | 0,8880 | 0,9135 | +2,78 |
| chair | 0,3689 | 0,3838 | 0,3710 | 0,4040 | +1,49 |
| cow | 0,7960 | 0,7469 | 0,7581 | 0,7605 | −4,92 |
| diningtable | 0,5423 | 0,5166 | 0,5257 | 0,5056 | −2,57 |
| dog | 0,8258 | 0,8287 | 0,8257 | 0,8517 | +0,29 |
| horse | 0,7784 | 0,7698 | 0,7564 | 0,7783 | −0,86 |
| motorbike | 0,7780 | 0,7642 | 0,7645 | 0,7726 | −1,38 |
| person | 0,8135 | 0,8572 | 0,8513 | 0,8829 | +4,38 |
| pottedplant | 0,3653 | 0,4663 | 0,4440 | 0,5013 | +10,11 |
| sheep | 0,7689 | 0,7645 | 0,7300 | 0,7558 | −0,44 |
| sofa | 0,5841 | 0,5576 | 0,5638 | 0,5672 | −2,65 |
| train | 0,8531 | 0,8126 | 0,8083 | 0,8083 | −4,05 |
| tvmonitor | 0,6137 | 0,5968 | 0,5926 | 0,5908 | −1,69 |

Farklar yuvarlanmamış kayıtlı IoU'lardan hesaplandı; tablodaki dört ondalıkla yapılan fark küçük yuvarlama farkı verebilir. K=3 baseline'a karşı 10/21, random mean'e karşı 16/21 sınıfta yüksek IoU verdi. Sınıf bazlı güven aralıkları hesaplanmadı; bu sayılar betimseldir. Sınıf IoU'su küçük nesne IoU'su değildir.

## Selector kalitesi ve oracle

50 development görüntüsünde yanlış piksel entropy ortalaması 0,3274, doğru piksel 0,1129. En yüksek entropy'li %20 piksel hataların %59,88'ini kapsadı. Bu piksel sıralama ölçümüdür.

| K=1 crop metriği, 50 development | Adaptive | Random üç seed mean |
|---|---:|---:|
| Baseline hata kapsaması | %45,87 | %28,90 |
| Crop hata yoğunluğu | %17,41 | %10,63 |
| Geçerli alan kapsaması | %24,25 | %25,02 |

Adaptive hata kapsaması +16,97 puan daha yüksek. Hatalı bölgeyi seçmek, crop inference'ın o hatayı düzelteceği garantisi değildir.

Oracle zoom: aynı 50 development görüntüsünde pooled mIoU 0,4883 → 0,6180; hedef sınıf IoU değişimi ortalama +42,73 puan. 47/0/3 örnek iyileşti/kötüleşti/aynı kaldı. Oracle GT hedef kutusu kullanır; crop boyutu/alanı adaptive ile eşit değildir. Bu sonuç eşit alanlı selector üstünlüğü veya sıkı bir oracle üst sınırı olarak sunulmaz.

20 örneğin K=3 manuel kontrolünde hedef sınıf kapsaması adaptive K=1 %44,72, K=3 %73,51, random 42 K=3 %74,14. Geçerli birleşim alanları sırasıyla %24,20/%55,65/%63,46. Index 10 chair, 44 tvmonitor ve 51 bicycle hedefleri K=3'te tamamen kaçırıldı. Bu galeride adaptive küçük hedef kapsaması üstünlüğü gösterilmedi.

## Development ablation ve seçilen konfigürasyon

Önceki K=1 fusion alpha 0,25/0,50/0,75/1,00 için mIoU 0,5011/0,5090/0,5267/0,5408; padding 0/%10/%20 için 0,5408/0,5358/0,5103; crop 56/84/112 için 0,5199/0,5187/0,5408. Development adaptive K=1/2/3 için 0,5408/0,5685/0,5807. Bunlar K=3 için tüm parametrelerin genel optimumunu kanıtlamaz.

8 Ekim threshold ablation: bir seferde yalnızca bir eşik değişti; K en fazla 3, crop=112, stride=28, padding=0, aynı fusion ve aynı 50 görüntü. Önceki baseline ve K=3 confusion matrix'leri 50 görüntünün tamamında birebir eşleşti.

| Konfigürasyon | Pooled mIoU | Δ vs reference (puan) | Mean K | Mean passes | Zero crop görüntü | Mean geçerli alan |
|---|---:|---:|---:|---:|---:|---:|
| Baseline | 0,4883 | −9,24 | 0,00 | 1,00 | 50 | %0,00 |
| Reference: entropy=0, IoU=0,30 | 0,5807 | 0,00 | 3,00 | 4,00 | 0 | %56,03 |
| Minimum entropy=0,10 | 0,5712 | −0,96 | 2,74 | 3,74 | 3 | %51,31 |
| Minimum entropy=0,20 | 0,5140 | −6,68 | 1,40 | 2,40 | 15 | %28,67 |
| Minimum entropy=0,30 | 0,4718 | −10,90 | 0,32 | 1,32 | 41 | %6,43 |
| IoU threshold=0,10 | 0,5796 | −0,11 | 2,50 | 3,50 | 0 | %58,60 |
| IoU threshold=0,50 | 0,5700 | −1,07 | 3,00 | 4,00 | 0 | %45,73 |

Minimum entropy, crop'un ortalama normalized entropy skoruna uygulandı. Crop sayısı azaltılabildiği için bu deney sabit işlem bütçeli selector karşılaştırması değildir. Ortak crop inference'ları tekrar kullanıldı; latency ölçülmedi. IoU=0,10 seçeneğinin daha az geçişi olası bir maliyet seçeneğidir; süre avantajı veya doğruluk eşdeğerliği kanıtlanmadı.

En yüksek development pooled mIoU mevcut reference konfigürasyonunda kaldı. Konfigürasyon değiştirilmedi; bu nedenle 1.399 görüntülük sonuçlar yeni bir çalışma gibi tekrar hesaplanmadı. “En iyi” yalnızca denenen seçenekler arasında ve bu development metriği için kullanılır. Threshold farklarının güven aralıkları hesaplanmadı.

## Entegrasyon ve demo kanıtı

| VOC index 9 | Image mIoU | Wrong pixels | Passes |
|---|---:|---:|---:|
| Baseline | 0,8301 | 10.264 | 1 |
| K=1 | 0,9214 | 6.072 | 2 |
| K=3 | 0,9512 | 4.536 | 4 |
| K=5 | 0,9597 | 3.525 | 6 |

K=5 birleşim alanı %78,41, maksimum local coverage 4. Crop başına forward süreleri bu tek çalıştırmada 42,65/14,35/21,25/24,20/15,58 ms; preprocessing+forward+transfer/resize toplamları 77,59/22,50/33,53/32,32/24,04 ms. Fusion/metrik/disk kaydı bu crop sürelerine dahil değildir. Bunlar tek örneklik entegrasyon ölçümü; bilimsel benchmark yerine kullanılmaz.

Crop logits, native prediction ve koordinatlar dosyalara kaydedildi. Yeniden yüklenen yerel kayıtlarla aynı K=5 final prediction üretildi. K=5 için geniş veri seti doğruluk değerlendirmesi yapılmadı.

Temiz [final demo notebook'u](../Notebooks/DINOv3_Adaptive_Zoom_Final_Demo.ipynb) Colab'da VOC index 9 ve 10 üzerinde çalıştırıldı. Index 9 K=3 sonucu önceki 0,9512 ile eşleşti. Yedi panel kullanıcı ekran görüntüsünden incelendi: original, GT, global prediction, entropy, crop kutuları, local predictions, final segmentation. Yerel prediction etiketleri sunum için büyütülebilir.

Index 10 K=3 demo: 0,4658 → 0,3208, −14,50 puan. Üç crop sofa çevresinde kümelendi ve yerel tahminlerle sınıf parçalanması görüldü. Başarı ve failure örneği birlikte kaydedildi.

8 Ekim'de özel JPG upload akışı kullanıcı tarafından çalıştırıldı: görüntü 736×1041, istenen/seçilen K=3/3 ve dört backbone geçişi. GT bulunmadığı için doğruluk metrikleri hesaplanmadı; `custom_photo_demo_20261008T123108_196389Z.zip` oluşturuldu. Kullanıcı yedi panelli görseli de paylaştı ve panel düzeni incelendi; upload/inference/fusion/kayıt/görselleştirme zinciri doğrulandı.

Özel görüntü bir insan ve gitar içeren illüstrasyondur. Model insan/arka plan maskesi üretiyor; gitar VOC'nin 21 sınıfında ayrı bir kategori değildir. Final maskenin gövde/bacak şekli değişiyor; bacak çevresinde kutuya hizalı olabilecek keskin geçişler görülüyor. Bunlar görsel gözlemdir; GT olmadan final maskenin daha doğru olduğu veya mIoU artışı iddia edilmez. Bu örnek gerçek fotoğraflardaki genel başarı yerine sunulmaz.

## Failure analizi ve sınırlar

- Index 10 K=1 tanısı: 6.010 piksel düzeldi, 11.697 yeni hata oluştu; sofa IoU 0,5086 → 0,3314. Hatalar sadece crop sınırında değil iç bölgelerde de görüldü.
- Index 14 K=1 tanısı: 844 piksel düzeldi, 12.992 yeni hata oluştu; tvmonitor IoU 0,5110 → 0,2055. Ekran içeriği yerel nesne olarak yorumlanabildi; gerçek hedef şişe crop dışında kaldı.
- Index 101: hata pikseli azalırken false-positive sınıfın image mIoU ortalamasına girmesi nedeniyle image mIoU düştü. Değerlendirme tanımı sonuca göre değiştirilmedi.
- Entropy yüksekliği hata olasılığını işaret edebilir; confident yanlış tahminleri kaçırabilir. NMS kutuları tamamen ayrık yapmaz veya farklı nesneleri seçmeyi garanti etmez.
- Bağlam/ölçek kaybı olası nedenlerdir; tek tek nedensel olarak izole edilmedi. Crop sınırı artifact kontrolü görseldir; ayrı sayısal sınır metriği yoktur.
- Genel küçük/orta/büyük nesne IoU analizi yapılmadı. Classwise IoU ve development hedef kapsaması bu analiz yerine geçmez.

Yerelde incelenen kayıtlar: [K=1 galeri](../Artifacts/selector_gallery_review/review_TR.md), [refinement tanısı](../Artifacts/refinement_review/review_TR.md), [K=3 galeri](../Artifacts/topk_gallery_review/review_TR.md).

## Görev ve teslim durumu

| Görev | Durum |
|---|---|
| Region selector kalite kontrolü | Development kapsamı için tamamlandı; ölçülebilir hata kapsaması ve görsel kanıt var. |
| Second Pass entegrasyonu | Tamamlandı; K=1/3/5, native dönüş, çoklu crop, kalıcı kayıt/reload, crop başına süre. |
| Fusion entegrasyonu | Tamamlandı; global dışı korunuyor, overlap local logits mean, GT/baseline/final ve başarı/failure kayıtları. |
| Selector ablation | Random-K, crop size, entropy/overlap eşikleri yapıldı; denenen development seçeneklerinden reference seçildi. Oracle kıyası alan/GT farkıyla sınırlandırılır. |
| Büyük doğruluk ve paired bootstrap | Tamamlandı; K=1 ve K=3, 1.399 görüntü. |
| Random K=3 dahil süre benchmark'ı | Tamamlandı; aynı 100 görüntüde 7 yöntem ve paired latency bootstrap. |
| Sınıf bazlı IoU | Tamamlandı; 21 sınıf, mevcut confusion matrix'lerden. |
| Temiz notebook ve VOC demo | Notebook oluşturuldu; iki VOC örneğinde inference ve görsel çıktı doğrulandı. |
| Özel görüntü demo | JPG upload, inference, sonuç paketi ve yedi panel kontrolü tamamlandı; GT olmadığı için doğruluk değerlendirmesi yok. |
| Tez/sunum ve teslim paketi | Tamamlanmadı. |

8 Ekim'de [Türkçe yöntem ve sonuç bölüm taslağı](DINOv3_Tez_Yontem_ve_Sonuclar_Taslagi_TR.md) hazırlandı; başarı/failure ekran görüntüleri kalıcı şekil klasörüne kopyalandı. Bu taslak tamamlanmış tez veya üniversite şablonuna uyarlanmış teslim değildir. DINOv3 makalesi ve resmi VOC değerlendirme kaynağı doğrulanarak taslağa eklendi.

Mevcut bilimsel iddialar için büyük deneyleri tekrarlamak gerekmiyor. Yeni bir “küçük nesnelerde genel üstünlük” iddiası istenirse önceden tanımlı boyut gruplarıyla ayrı analiz gerekir. Alternatif region score ve confidence fusion denenmedi; mevcut yöntemin zorunlu tamamlanma şartı olarak sunulmuyor.

## Kalan teslim işleri

1. Özel görüntü demo ZIP'ini de final teslim yedeğine eklemek. Kullanıcı bu ZIP'in PC'ye kaydını henüz ayrıca onaylamadı; GT olmadan doğruluk artışı iddia edilmez.
2. Tez/sunum için yöntem akışı, ana kalite tablosu, aynı oturum süre tablosu, selector ablation, başarı ve failure görsellerini düzenlemek.
3. Raporu üniversitenin tez/sunum formatına aktarmak; kaynakça için DINOv3/VOC/uncertainty/segmentation yöntem kaynaklarını ayrı doğrulamak. Bu deney raporu literatür taraması değildir.
4. Notebook, checkpoint, tüm ayrı sonuç arşivleri, görseller, rapor ve kullanım yönergesini tek teslim dizininde toplayıp temiz oturumdan geri yükleme kontrolünü yapmak. PyTorch modül referansı hatası oturumda giderildi; kurulum sonrası temiz yeniden başlatma ve yeniden çalıştırma kontrolü teslim doğrulamasında yer almalı.

## Kanıt ve yedekleme

Nicel sonuçlar, kullanıcının Colab'dan paylaştığı çalışma çıktılarından aktarılmıştır. Tüm yeni ZIP'lerin içeriği yerelde bağımsız okunmuş veya inference yeniden çalıştırılmış değildir. Yerelde doğrudan incelenenler K=1/K=3 galerileri ve iki refinement örneğidir; final demo görselleri bu konuşmada kullanıcı tarafından paylaşıldı.

Son kullanıcı çıktısında `DINOv3_complete_20261008T122013_722899Z.zip` 54,0 MB olarak oluşturuldu; checkpoint/threshold sonuçları ve ZIP CRC kontrolü geçti. Kullanıcı paket/notebook kaydını onayladı. Bu arşiv mevcut Colab run klasörünü içerir; önceki oturumda ayrı indirilen yeni benchmark, classwise ve demo ZIP'lerinin de ayrıca korunması gerekir. Tek bir eski ZIP'in tüm sonraki sonuçları içerdiği varsayılmaz.

Önemli kayıt adları:

- `DINOv3_topk3_progress_1399.zip`
- `full_voc_topk3_v1/paired_bootstrap_k3_1399_20261006T230527_678459Z.json`
- `classwise_iou_1399_20261007T203351_459375Z.zip`
- `benchmark_all_k3_100_20261007T194714_806084Z.zip`
- `paired_latency_bootstrap_20261007T204156_744467Z.json`
- `second_pass_k5_20261007T201636_189490Z.zip`
- `final_demo_20261007T204944_753807Z.zip`
- `final_demo_index10_20261007T205216_610888Z.zip`
- `selector_threshold_ablation_dev50_v1.zip`
- `DINOv3_complete_20261008T122013_722899Z.zip`
- `custom_photo_demo_20261008T123108_196389Z.zip` (önceki tam yedekten sonra oluşturuldu; ayrıca saklanmalı)

Tekrar kullanım: [final demo yönergesi](../Notebooks/Final_Demo_Kullanim_TR.md). `.ipynb` kod dosyası sonuç ZIP'lerinden ayrı saklanmalıdır.
