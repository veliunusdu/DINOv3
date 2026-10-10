# Top-K selector — 20 görselin incelemesi

Kaynak: topk_gallery_20_20261006T231350_145105Z.zip. 20 PNG'nin tamamı doğrudan incelendi. Kaydedilmiş 20 örneğin kutuları, sıralı skorları, K=1 prefix'i ve çiftler arası entropy-haritası box IoU <= 0.30 koşulu ayrıca programatik olarak doğrulandı. Canlı Colab inference burada tekrar çalıştırılmadı.

## Bulgular

Ortalama hedef sınıf piksel kapsaması K=1 için %44.72, adaptive K=3 için %73.51, random K=3 seed 42 için %74.14. Bunlar aynı geliştirme grubunun ilk 20 görüntüsüdür; bütün görüntünün hata kapsaması veya segmentasyon başarısı değildir. GT sadece bu tanısal ölçümde kullanılmıştır.

| Index | Hedef | K=1 / K=3 / random K=3 hedef kapsaması (%) | Görsel gözlem |
|---|---|---|---|
| 9 | bottle | 83.2 / 100.0 / 100.0 | Ek crop'lar şişenin üst kısmını tamamlıyor; merkez üç kez işleniyor. |
| 10 | chair | 0.0 / 0.0 / 72.9 | Üç adaptive kutu soldaki sofa çevresinde; sağdaki hedef hâlâ dışarıda. |
| 11 | cow | 99.9 / 100.0 / 100.0 | Hedef zaten K=1'de neredeyse tam; ek kutular aynı bölgeyi tekrar işliyor. |
| 13 | boat | 89.9 / 100.0 / 89.9 | Ek alt kutu hedefi tamamlıyor; diğer crop geniş arka plan içeriyor. |
| 14 | bottle | 0.0 / 25.3 / 35.6 | Monitör çevresindeki seçim sürüyor; şişenin yalnızca sağdaki bir kısmı kapsanıyor. |
| 18 | pottedplant | 13.3 / 100.0 / 13.3 | İkinci crop bitkiyi kapsıyor; K=1'deki eksik gideriliyor. |
| 21 | bus | 0.0 / 92.3 / 98.8 | İlk kutu tabelada, ek alt kutular otobüsün büyük kısmını kapsıyor. |
| 22 | bottle | 51.6 / 58.8 / 7.3 | Ek crop üst bölgeyi artırıyor; şişenin alt/sol bölümü hâlâ dışarıda. |
| 23 | person | 75.1 / 100.0 / 98.4 | Üst crop binicinin başını tamamlıyor; tek tek kutular nesneyi kesmeye devam ediyor. |
| 27 | boat | 92.5 / 100.0 / 100.0 | Üst crop teknenin direk bölgesini tamamlıyor. |
| 31 | bottle | 0.0 / 100.0 / 97.0 | İkinci alt crop, ilk kutunun kaçırdığı şişeyi kapsıyor. |
| 33 | tvmonitor | 0.0 / 48.7 / 46.2 | Sağ kutu monitörün alt kısmını kapsıyor; üstü dışarıda. Ignore alanı geniş. |
| 37 | bird | 100.0 / 100.0 / 100.0 | Hedef K=1'de tam; ek kutular kuşun aynı bölgelerini tekrar kapsıyor. |
| 42 | pottedplant | 0.0 / 53.5 / 71.7 | Üst kutu bitkinin altını kapsıyor; üst yapraklar kesiliyor. |
| 44 | tvmonitor | 0.0 / 0.0 / 58.5 | Üç adaptive kutu masa/insan çevresinde; üstteki monitör dışarıda. |
| 45 | chair | 0.0 / 93.0 / 100.0 | İkinci sağ crop sandalye hedefinin büyük bölümünü kapsıyor. |
| 49 | person | 90.2 / 100.0 / 98.6 | Alt-sağ crop ayakları tamamlıyor. |
| 50 | person | 98.6 / 98.6 / 35.0 | K=3 hedef sınıf kapsamasını artırmıyor; at çevresine genişliyor. |
| 51 | bicycle | 0.0 / 0.0 / 59.5 | Üç kutu duvar/pencere çevresinde kümeleniyor; bisiklet hâlâ tamamen dışarıda. |
| 52 | dog | 100.0 / 100.0 / 100.0 | Köpek ilk kutuda tam; ek kutular koyun/insan çevresine genişliyor. |

## NMS yorumu

IoU filtresi seçilen kutu çiftleri arasındaki toplam kesişim/birleşim oranını sınırlar. Kutuları tamamen ayrık yapmaz, üçlü piksel örtüşmesini engellemez ve farklı nesnelere yönelmeyi garanti etmez. Görsellerde birden fazla örnekte aynı belirsiz nesne çevresinde üç crop ve coverage=3 bölgeleri var. Bu, kaydedilmiş <=0.30 IoU koşuluyla uyumludur.

## Kapsam ve tamamlanma

Bu notebook protokolü için entropy -> adaylar -> skor sıralaması -> configurable Top-K -> NMS -> orijinal koordinatları zinciri çalışıyor. K/crop boyutu/overlap eşiği ayarlanabilir; koordinat/skorlar kaydediliyor; 20 örneğin görsel incelemesi tamamlandı. Bu durum küçük nesnelerin her zaman seçildiği veya bütün projenin tamamlandığı anlamına gelmez.

Random K=3'ün bu 20 örnekte hedef sınıf kapsaması biraz daha yüksek. Hedef kapsaması açısından adaptive üstünlüğü iddia edilemez. Ayrı 1399 görüntülük değerlendirmede adaptive K=3'ün random üç seed ortalamasına karşı pooled mIoU farkı +0.80 yüzde puan (%95 bootstrap aralığı [+0.29,+1.33]) idi; bu farklı bir metrik ve farklı bir örneklem kapsamıdır.
