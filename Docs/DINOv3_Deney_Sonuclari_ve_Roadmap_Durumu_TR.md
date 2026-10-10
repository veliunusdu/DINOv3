# DINOv3: deney sonuçları ve roadmap durumu

Bu belge 7 Ekim tarihli ara rapordur. Güncel ve birleştirilmiş sonuçlar için [8 Ekim deney raporunu](DINOv3_Final_Deney_Raporu_TR.md) kullanın; aşağıdaki tamamlanmamış durumların bazıları artık tamamlanmıştır.

Güncelleme: 7 Ekim 2026. Deneyler: Colab, Tesla T4.

## 7 Ekim ek sonuçları ve teslim durumu

İlk rapordan sonra kullanıcı Colab'da şu çalışmaları tamamladı:

- Second-pass K=5 entegrasyonu: VOC index 9 için baseline/K=1/K=3/K=5 image mIoU 0,8301/0,9214/0,9512/0,9597. Crop başına süreler ve yerel logits/prediction/koordinatlar kaydedildi; yerel kayıtlar yeniden yüklenerek aynı final prediction üretildi. Bu, tek örneklik entegrasyon kontrolüdür; genel K=5 üstünlüğü değildir.
- 1.399 görüntünün sınıf bazlı IoU tablosu kayıtlı matrislerden üretildi. K=3 baseline'a karşı 10/21, random K=3 ortalamasına karşı 16/21 sınıfta daha yüksek IoU verdi. Baseline'a karşı en büyük kazançlar bicycle +13,14, pottedplant +10,11; en büyük kayıplar bottle −6,30, cow −4,92 puan. Sınıf bazlı güven aralıkları hesaplanmadı.
- Random K=3 dahil yedi yöntemin aynı 100 görüntüde yeni süre benchmark'ı tamamlandı; 700 tahmin önceki kayıtlarla eşleşti. Aşağıdaki yeni tablo, önceki oturumun süre tablosundan ayrı değerlendirilmelidir.

| Yöntem | 100 görüntü mIoU | Mean ms | Median ms | P95 ms | Peak allocated GPU MB |
|---|---:|---:|---:|---:|---:|
| Baseline | 0,6506 | 79,2 | 65,5 | 127,6 | 98,3 |
| Adaptive K=1 | 0,6750 | 110,8 | 98,3 | 171,4 | 98,3 |
| Adaptive K=3 | 0,6702 | 147,5 | 133,0 | 216,5 | 98,3 |
| Random 42 K=3 | 0,6772 | 147,7 | 132,8 | 224,2 | 98,3 |
| Random 43 K=3 | 0,6794 | 149,2 | 133,5 | 220,4 | 98,3 |
| Random 44 K=3 | 0,6684 | 146,4 | 134,3 | 222,6 | 98,3 |
| Sliding | 0,6732 | 510,3 | 472,0 | 698,4 | 98,3 |

Yeni paired latency bootstrap: sliding/K=3 3,46× [3,33; 3,62]; K=3/baseline 1,86× [1,77; 1,95]; K=3/K=1 1,33× [1,26; 1,39]. K=3−random mean −0,31 ms [−4,51; +3,55]; belirgin süre farkı gösterilmiyor. K=3 süre azalması sliding'e göre %71,10 [%69,95; %72,35]. Bunlar aynı oturumdaki görüntü örneklemesi aralıklarıdır.

Bu ek sonuçlarla sınıf bazlı tablo ve Random K=3 timing görevleri tamamlandı. Genel küçük nesne alan grubu analizi, tez/sunum ve nihai teslim henüz tamamlanmadı. Aşağıdaki eski durum/iş listesi ilk rapor anını belgeler; bu ek bölüm güncel durum için önceliklidir.

Temiz demo notebook'u [Notebooks/DINOv3_Adaptive_Zoom_Final_Demo.ipynb](../Notebooks/DINOv3_Adaptive_Zoom_Final_Demo.ipynb) olarak oluşturuldu; kullanım yönergesi [Final_Demo_Kullanim_TR.md](../Notebooks/Final_Demo_Kullanim_TR.md). Sözdizimi ve koordinat kontrolleri yerelde geçti; birleşik demo için Colab inference/görsel kontrolü bekleniyor.

## Araştırmanın şu anki sonucu

Entropy ile seçilen üç crop'u yeniden işlemek, 1.399 görüntüde tek crop ve aynı K değerindeki random seçimin üç seed ortalamasından daha yüksek pooled mIoU verdi. Baseline'a karşı farkın %95 bootstrap aralığı sıfırı içeriyor. Sliding-window daha yüksek mIoU verdi; adaptive K=3 ise ayrı 100 görüntülük süre ölçümünde sliding'den yaklaşık 3,35 kat hızlıydı.

Bu sonuç, ölçülen uygulama için bir doğruluk–süre dengesi gösteriyor. “Baseline'dan kesin daha iyi”, “sliding ile eşdeğer doğruluk” veya “her küçük nesneyi buluyor” iddiaları desteklenmiyor.

## Protokol ve örneklemler

- Backbone: `facebook/dinov3-vits16-pretrain-lvd1689m`; dondurulmuş ViT-S/16, hidden size 384, patch size 16, dört register token.
- Head: 21 sınıflı 1×1 convolution. VOC2012 train grubunda 1.464 görüntü, beş epoch; checkpoint geri yüklemesi ve inference eşleşmesi Colab'da doğrulandı. Deneyler için yeniden eğitim yapılmadı.
- Her global/crop girişi 224×224'e boyutlandırılıyor. Entropy, sınıf olasılıklarından hesaplanıp log(21)'e bölünüyor.
- Selector: 224×224 entropy haritasında 112×112 crop, stride 28, 25 aday, ortalama entropy skoru, azalan sıralama, Top-K ve greedy IoU filtresi (eşik 0,30).
- Crop koordinatları orijinal görüntüye taşınıyor, sınırlar içinde tutuluyor. Crop orijinal görüntüden alınıyor ve aynı modelle yeniden işleniyor.
- Fusion: crop birleşimi içinde yerel logits; örtüşen crop'larda logits ortalaması; birleşim dışında global logits. Padding 0.
- Native değerlendirme: logits orijinal çözünürlüğe bilinear taşınıyor; GT=255 yok sayılıyor; background dahil. Pooled mIoU, görüntülerin confusion matrix toplamından hesaplanıyor.

| Örneklem | Amaç | Sınır |
|---|---|---|
| 50 development görüntüsü | Selector, crop/fusion/padding/K ablation ve failure analizi | GT ile küçük hedef içeren görüntüler seçildi; genel VOC dağılımını temsil etmiyor. Selector inference sırasında GT kullanmıyor. |
| 100 pilot görüntüsü | İlk doğruluk karşılaştırmaları ve süre benchmark'ları | Development ile çakışmıyor; sonuçları önceden görüldü. |
| 1.399 görüntü | Büyük değerlendirme ve paired bootstrap | VOC val 1.449 görüntüden 50 development örneği çıkarıldı. Önceden görülen 100 pilot bu gruba dahil; bağımsız, dokunulmamış test seti değildir. |
| 20 development görüntüsü | Manuel görsel kontrol | Development grubunun ilk 20 örneği; nicel genelleme için ayrı test değildir. |

## Ana doğruluk sonuçları

1.399 görüntü, aynı native değerlendirme protokolü:

| Yöntem | Pooled mIoU | Tek görüntü için backbone geçişi |
|---|---:|---:|
| Baseline | 0,7133 | 1 |
| Adaptive K=1 | 0,7124 | 2 |
| Adaptive K=3 | 0,7220 | 4 |
| Random K=3, seed 42 | 0,7148 | 4 |
| Random K=3, seed 43 | 0,7115 | 4 |
| Random K=3, seed 44 | 0,7157 | 4 |
| Random K=3, üç seed ortalaması | 0,7140 | Her seed için 4 |
| Sliding-window | 0,7325 | 25 |

Random mean, üç ayrı yöntemin pooled mIoU ortalamasıdır; tahmin ensemble'ı değildir. Random K=3 aynı adaylar, K ve NMS kısıtıyla seçiliyor. Aynı crop sayısı, aynı crop birleşim alanını garanti etmiyor.

5.000 paired görüntü bootstrap örneklemesi; farklar yüzde puan:

| Karşılaştırma | Fark | %95 aralık | Desteklenen yorum |
|---|---:|---|---|
| K=3 − baseline | +0,87 | [−0,14; +1,96] | Nokta tahmini pozitif; aralık sıfırı içeriyor. |
| K=3 − K=1 | +0,96 | [+0,35; +1,55] | Bu değerlendirmede K=3 lehine fark. |
| K=3 − random K=3 mean | +0,80 | [+0,29; +1,33] | Bu üç random seed ortalamasına karşı K=3 lehine fark. |
| K=3 − sliding | −1,05 | [−1,51; −0,58] | Sliding daha yüksek mIoU veriyor. |
| K=1 − baseline | −0,09 | [−0,89; +0,76] | K=1 üstünlüğü gösterilmiyor. |
| K=1 − random K=1 mean | −0,06 | [−0,70; +0,60] | K=1 üstünlüğü gösterilmiyor. |
| K=1 − sliding | −2,01 | [−2,79; −1,24] | Sliding daha yüksek mIoU veriyor. |

Random seed sayısı üçtür; bootstrap görüntü örneklemesindeki belirsizliği değerlendirir, tüm olası random seed'lerin dağılımını kapsamaz.

Önceki 1.449 görüntülük **224×224 GT değerlendirmesi** 0,7104 verdi. Bu değer yukarıdaki native/1.399 sonuçlarla doğrudan karşılaştırılmaz. Eski belgelerdeki 0,7147 sonucu bu checkpoint ile birebir yeniden üretildiği şeklinde sunulmamalıdır.

## Süre ve maliyet

K=3 eklendikten sonra aynı oturumda yeniden ölçülen 100 pilot görüntüsü:

| Yöntem | Bu 100 görüntüde mIoU | Mean ms | Median ms | P95 ms | Peak allocated GPU MB |
|---|---:|---:|---:|---:|---:|
| Baseline | 0,6506 | 92,4 | 72,0 | 146,6 | 98,3 |
| Adaptive K=1 | 0,6750 | 141,5 | 107,5 | 244,7 | 98,3 |
| Adaptive K=3 | 0,6702 | 187,0 | 146,9 | 335,2 | 98,3 |
| Sliding-window | 0,6732 | 625,7 | 492,4 | 919,7 | 98,3 |

| Ortalama sürelerin oranı | Oran | %95 paired bootstrap aralığı |
|---|---:|---|
| Sliding / K=3 | 3,35× | [3,17; 3,58]× |
| K=3 / baseline | 2,02× | [1,91; 2,16]× |
| K=3 / K=1 | 1,32× | [1,29; 1,36]× |

K=3'ün sliding'e göre ortalama süre azalması %70,11; aralık [%68,46; %72,03]. Bu aralık aynı oturumdaki görüntü değişkenliğini yansıtır; oturumlar veya farklı GPU'lar arasındaki değişkenliği ölçmez.

Ölçüm öncesi warm-up ve CUDA synchronization yapıldı. Görüntü decode, GT okuma ve metrik hesaplama süre dışında; preprocessing, inference, CPU logits aktarımı, selector, resize ve fusion süre içinde. Tahminler kayıtlı değerlendirmeyle eşleşti.

Sliding 25 crop'u sıralı işler; batched/optimize sliding ölçülmedi. CPU belleği ölçülmedi. Eşit GPU peak değerleriyle GPU bellek tasarrufu iddia edilemez. Random K=3'ün aynı oturum/protokolde süre ölçümü henüz yoktur.

100 pilotta K=3, K=1'den düşük mIoU verdi; 1.399 görüntüde daha yüksek verdi. K artırmanın her görüntü veya her alt grupta fayda sağlamadığı raporlanmalıdır.

## Selector kanıtı ve tanısal deneyler

50 development görüntüsünde yanlış piksellerin ortalama entropy'si 0,3274, doğru piksellerinki 0,1129. En yüksek entropy'li %20 piksel, hataların yaklaşık %59,88'ini içeriyor. Bu piksel sıralama bulgusudur; crop hata kapsamasıyla aynı metrik değildir.

| K=1 selector metriği, 50 development görüntüsü | Adaptive | Random üç seed mean |
|---|---:|---:|
| Baseline hata piksellerini kapsama | %45,87 | %28,90 |
| Crop içindeki hata yoğunluğu | %17,41 | %10,63 |
| Geçerli görüntü alanı kapsaması | %24,25 | %25,02 |

Hata kapsaması farkı +16,97 yüzde puan. Selector bu development grubunda daha fazla hata içeren bölgeleri seçiyor; crop modelinin bu hataları düzelteceği garantisi yok.

Oracle zoom 50 örnekte pooled mIoU'yu 0,4883'ten 0,6180'a çıkardı; hedef sınıf IoU değişimi ortalama +42,73 puan, 47/0/3 iyileşen/kötüleşen/aynı. Oracle GT hedef kutusu kullanır ve alanı adaptive ile eşit değildir; uygulanabilir selector veya eşit maliyet/alanlı karşılaştırma değildir.

Development ablation sonuçları:

| Deney | Sonuç |
|---|---|
| K=1 fusion alpha 0,25 / 0,50 / 0,75 / 1,00 | mIoU 0,5011 / 0,5090 / 0,5267 / 0,5408; replacement seçildi. |
| K=1 padding 0 / %10 / %20 | 0,5408 / 0,5358 / 0,5103; padding 0 seçildi. |
| K=1 crop size 56 / 84 / 112 | 0,5199 / 0,5187 / 0,5408; 112 seçildi. |
| Adaptive K=1 / 2 / 3 | 0,5408 / 0,5685 / 0,5807. |
| Eşleşen Top-K random mean, K=1 / 2 / 3 | 0,5125 / 0,5312 / 0,5469. |

Bunlar development bulgularıdır; K=3 için tüm parametrelerin veya genel optimumun doğrulandığı anlamına gelmez. Top-K deneyinin random seçim protokolü önceki K=1 random deneyinden farklıdır; iki random sonuç tablosu karıştırılmamalıdır.

## Manuel inceleme ve failure case'ler

K=1 ve K=3 için 20'şer görsel yerelde incelendi. Kaydedilmiş K=3 koordinatları, skor sırası, K=1 prefix ve çiftler arası IoU ≤0,30 ayrıca kontrol edildi.

| 20 development örneğinde metrik | Adaptive K=1 | Adaptive K=3 | Random K=3 seed 42 |
|---|---:|---:|---:|
| Hedef sınıf piksel kapsaması | %44,72 | %73,51 | %74,14 |
| Geçerli crop birleşim alanı | %24,20 | %55,65 | %63,46 |

K=3, 14 örnekte K=1'e göre hedef kapsamasını artırdı, altısında aynı kaldı. Index 10 chair, 44 tvmonitor ve 51 bicycle hedefleri K=3'te tamamen dışarıda kaldı. Random daha geniş birleşim alanıyla biraz daha fazla hedef pikseli kapsadı. Bu 20 örnek hedef kapsamasında adaptive üstünlüğünü göstermiyor.

NMS kutuları ayrık yapmaz ve farklı nesnelerin seçilmesini garanti etmez; aynı belirsiz bölgede üç crop görülebiliyor.

Refinement tanısı:

- Index 10: sofa IoU 0,5086 → 0,3314. 6.010 piksel düzelirken 11.697 yeni hata oluştu. Yerel sınıf değişimleri yalnızca crop kenarında değil, iç bölgelerde de var.
- Index 14: tvmonitor IoU 0,5110 → 0,2055. 844 piksel düzelirken 12.992 yeni hata oluştu. Ekran içeriğinin yerel nesne olarak yorumlanması ve monitor/background karışması görüldü; gerçek hedef şişe seçimin dışında kaldı.
- Index 101: yanlış piksel sayısı azalmasına rağmen yeni false-positive sınıfın image mIoU ortalamasına girmesiyle image mIoU düştü. Bu metrik davranışıdır; değerlendirme tanımı sonuca göre değiştirilmedi.

Bağlam kaybı ve ölçek değişimi olası açıklamalardır; nedenleri tek tek izole eden deney yapılmadı.

İnceleme kayıtları:

- [K=1 selector galerisi incelemesi](../Artifacts/selector_gallery_review/review_TR.md)
- [Refinement hata tanısı](../Artifacts/refinement_review/review_TR.md)
- [Top-K galerisi incelemesi](../Artifacts/topk_gallery_review/review_TR.md)

## Roadmap durumu

| Milestone / görev | Durum | Kalan iş |
|---|---|---|
| M1 — Oracle zoom hipotezi | Tamamlandı | Bulguyu GT ve alan farkı sınırıyla raporlamak. |
| M2 — Entropy → otomatik Top-K koordinatları | Tamamlandı | Configurable K/crop/overlap, bounds, skor kaydı ve 20 görsel incelemesi var. |
| M3 — Global → selector → crop inference → multi-fusion | Colab deney akışı çalışıyor | Temiz, tek giriş noktalı tam notebook/API; final demo çıktıları. |
| M4 — Bilimsel değerlendirme | Ana doğruluk ve süre bulguları hazır | Kayıtlı matrislerden sınıf bazlı IoU; Random K=3 süre ölçümü; genel küçük nesne/alan grubu analizi henüz yok. |
| M5 — Teslim paketi, demo, tez ve sunum | Tamamlanmadı | Birleşik kod, şekiller, demo, yöntem/sonuç metni, sunum. |

Oracle, entropy, selector, Top-K/NMS, orijinal crop çıkarma, yeniden inference, çoklu fusion, matched random karşılaştırması ve bootstrap görevlerini tekrar çalıştırmak gerekmiyor.

K=1 crop/padding/fusion ve development K=1/2/3 ablation yapıldı. Alternatif region score, farklı NMS eşikleri ve K=3 confidence fusion denenmedi. Bunlar yapılmamış seçenekler olarak belirtilmeli; yeni hipotez olmadan deney kapsamı sürekli genişletilmemeli. Roadmap'teki örnek K değerlerinin tamamını denemek tek başına zorunlu bir bitirme ölçütü değildir.

Mevcut başlangıç notebook'u yalnızca oturum/checkpoint/veri kurtarma içindir; bütün selector ve deney kodlarını içeren nihai notebook değildir. Entropy API'si ve görsel çıktı sözleşmesi final paketinde birleştirilmeli.

## Sıradaki işler

1. **Sınıf bazlı IoU raporu:** 1.399 görüntünün kayıtlı confusion matrix'lerinden baseline, K=1, K=3, random K=3 ve sliding için 21 sınıf IoU tablosu; K=3 kazanç/kayıpları. Yeni eğitim veya GPU inference gerektirmez. Eksik sınıf sıfır yerine tanımlı kurala göre işlenmeli.
2. **Benchmark tablosunu tamamlama:** Random K=3'ü aynı 100 görüntü, aynı timing sınırları ve bağımsız geçişlerle ölçmek. Önceki oturumun süreleriyle kesin eş oturum iddiası kurmadan, karşılaştırılan yöntemleri birlikte ölçmek.
3. **Teslim edilebilir akış:** Geri yükleme → tek görüntü → entropy → configurable Top-K → crop inference → fusion → kayıt. Original, GT varsa GT, baseline, entropy, crop kutuları, local prediction ve final mask içeren demo.
4. **Yazım ve son paket:** Deney tablosu/şekiller, failure örnekleri, sınırlar, yeniden çalıştırma yönergesi ve tez/sunum. Küçük nesneye özel iddia gerekiyorsa önceden tanımlı alan gruplarıyla ayrı analiz; mevcut hedef kapsamasını küçük nesne IoU diye sunmamak.

## Kanıt kaynakları ve kayıt sınırı

Ana kalite/süre/bootstrap sayıları kullanıcının Colab'dan paylaştığı çıktılardır. Bu rapor, tüm deney ZIP'lerinin içeriğinin yerelde yeniden doğrulandığı iddiasını taşımaz. Yerelde doğrudan incelenen arşivler K=1 selector galerisi, iki örneklik refinement tanısı ve Top-K 20 görsel galerisidir. Canlı Colab inference bu raporu hazırlarken tekrar çalıştırılmadı.

İlgili Colab çıktı adları:

- `DINOv3_full_voc_progress_1399.zip`
- `full_voc_fixed_v1/paired_bootstrap_1399_20261006T221201_050013Z.json`
- `DINOv3_topk3_progress_1399.zip`
- `full_voc_topk3_v1/paired_bootstrap_k3_1399_20261006T230527_678459Z.json`
- `benchmark_k3_100_20261006T231004_663101Z.zip`
- `paired_latency_bootstrap_20261006T231148_724488Z.json`
- `topk_gallery_20_20261006T231350_145105Z.zip`

Colab geçici diski sonuçların kalıcı kaydı değildir. Güncel sonuç/checkpoint ZIP'leri ve notebook'un `.ipynb` kopyası PC'de korunmalıdır.
