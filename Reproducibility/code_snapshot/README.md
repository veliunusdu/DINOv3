# DINOv3 Adaptive Zoom

DINOv3 tabanlı semantik segmentasyonda, global tahminin belirsiz bölgelerini otomatik seçip yakınlaştırarak yeniden işleyen bitirme projesi. Deneyler Google Colab üzerinde Tesla T4 ile VOC2012 verisinde yürütüldü.

## Yöntem

1. Görüntüyü 224×224 boyutunda işleyerek global logits üret.
2. Softmax olasılıklarından `log(21)` ile normalize edilmiş entropy haritası hesapla.
3. Aday crop'ları ortalama entropy skoruna göre sırala ve IoU filtresiyle Top-K seç.
4. Seçilen bölgeleri **orijinal RGB görüntüden** al, model giriş boyutuna getir ve aynı modelle yeniden işle.
5. Yerel logits'i orijinal koordinatlara taşı. Crop dışında global logits'i koru, crop içinde yerel logits kullan, örtüşmelerde yerel logits'lerin ortalamasını al.

Selector inference sırasında ground truth kullanmaz. Model, dondurulmuş `facebook/dinov3-vits16-pretrain-lvd1689m` backbone ve VOC2012 train grubunda 5 epoch eğitilmiş 21 sınıflı 1×1 convolution head kullanır. Demo sırasında yeniden eğitim yapılmaz.

Varsayılan seçim: **K=3, crop=112, stride=28, kutu IoU eşiği=0,30, padding=0**. Crop boyutu 224×224 seçim haritasındaki pencere boyutudur. Gerçek seçim sayısı IoU filtresinden sonra K'dan az olabilir.

## Sonuçlar

| Yöntem | Pooled mIoU | Ortalama süre (ms/görüntü) | Backbone geçişi |
|---|---:|---:|---:|
| Baseline | 0,7133 | 79,2 | 1 |
| Adaptive K=1 | 0,7124 | 110,8 | 2 |
| Adaptive K=3 | 0,7220 | 147,5 | 4 |
| Random K=3, üç seed ortalaması | 0,7140 | 147,8 | Her seed için 4 |
| Sliding-window | 0,7325 | 510,3 | 25 |

**mIoU 1.399 görüntüde**, orijinal çözünürlükte, background dahil ve ignore=255 ile hesaplandı. **Süre aynı oturumdaki 100 görüntüde** ölçüldü. Bu 100 görüntü ana değerlendirme grubunun içindedir. Random ortalaması 42/43/44 seed sonuçlarının ortalamasıdır, prediction ensemble değildir.

- K=3 − random K=3: **+0,80 yüzde puan**, %95 paired bootstrap aralığı **[+0,29; +1,33]**.
- K=3 − baseline: **+0,87 puan**, aralık **[−0,14; +1,96]**. Baseline'a üstünlük kesinleşmedi.
- K=3 − sliding: **−1,05 puan**, aralık **[−1,51; −0,58]**. Sliding daha yüksek doğruluk sağladı.
- Süre benchmark'ında sliding/K=3 oranı **3,46×**, aralık **[3,33; 3,62]**. Sliding sıralı, optimize edilmemiş uygulamadır.

Development grubundaki 50 görüntü konfigürasyon seçimi için kullanıldı ve ana değerlendirmeden çıkarıldı. Ana grup daha önce incelenmiş validation verisidir; dokunulmamış test seti değildir. Aynı crop sayısı, adaptive ve random yöntemlerinin aynı görüntü alanını kapsadığı anlamına gelmez. Yerel işleme bazı görüntüleri kötüleştirebilir.

Tam protokol, ablation, sınıf sonuçları ve sınırlılıklar: [final deney raporu](Docs/DINOv3_Final_Deney_Raporu_TR.md).

Checkpoint ve kaynak arşiv hash'leri, split ID listeleri, seed'ler ve güncel kod snapshot'ı: [deney manifest’i](Reproducibility/experiment_manifest.json) ve [doğrulama kapsamı](Reproducibility/README_TR.md). Geçmiş deneylerin Git commit’i, training ID listesi ve tam ortam dökümü mevcut kaynaklarda eksiktir; güncel Git commit eski deneylerin orijinal sürümü olarak sunulmaz.

## Colab'da demo çalıştırma

1. [Final demo notebook'unu](Notebooks/DINOv3_Adaptive_Zoom_Final_Demo.ipynb) indir ve Google Colab'da **Dosya → Notebook yükle** ile aç.
2. Çalışma zamanı türünü **T4 GPU** olarak seç.
3. Hücreleri sırayla çalıştır: bağımlılıklar, deney ZIP'ini geri yükleme, Hugging Face giriş işlemi ve model yükleme.
   Bağımlılıklardan sonra **1b Final config** hücresini çalıştır. [Açıklamalı TOML](Config/DINOv3_Final_Config.toml) notebook'a gömülüdür; kendi config'in için dosyayı `/content/DINOv3_Final_Config.toml` yoluna yükle. K, crop size, stride, minimum entropy ve overlap eşiği bu dosyadan okunur. Kullanılan config çıktı ZIP'lerine eklenir; checkpoint hash'i ve model revision doğrulanır.
4. VOC örneği için veri hücresini çalıştır. Kendi JPG/PNG görüntün için demo hücresinde `SOURCE="upload"` seç.
5. Demo hücresini çalıştır ve üretilen sonuç ZIP'ini indir. Oturum sonunda tam yedek hücresini de çalıştır.

**Gerekli checkpoint bu Git deposunda bulunmuyor.** Checkpoint içeren deney paketi, örneğin `DINOv3_complete_20261008T122013_722899Z.zip`, ayrıca temin edilip notebook'a yüklenmelidir. Paket için bu depoda bir indirme bağlantısı yayımlanmadı. Yalnızca bu depoyu klonlamak eğitilmiş head'i sağlamaz. Hugging Face hesabının kullanılan backbone'a erişimi olması gerekir.

Varsayılan demo VOC index 9'u K=3 ile işler. Kendi görüntünde ground truth yoksa doğruluk metrikleri hesaplanmaz. Model yalnızca VOC'nin 21 sınıfını tahmin eder. Bu notebook demo içindir; eğitim ve büyük benchmark deneylerini yeniden çalıştırmaz.

Ayrıntılı hücre sırası ve çıktı tanımları: [demo kullanım kılavuzu](Notebooks/Final_Demo_Kullanim_TR.md).

## Görsel örnekler

### İyileşen örnek

VOC index 9: image mIoU **0,8301 → 0,9512**, K=3 ve toplam 4 geçiş.

![Index 9: baseline, entropy, seçilen crop'lar ve final segmentasyon](Artifacts/Thesis_Figures/voc_index9_success_k3.png)

### Kötüleşen örnek

VOC index 10: image mIoU **0,4658 → 0,3208**. Belirsiz bölgenin seçilmesi, yerel tahminin doğru olmasını garanti etmez.

![Index 10: second-pass tahminlerinin final segmentasyonu kötüleştirdiği örnek](Artifacts/Thesis_Figures/voc_index10_failure_k3.png)

## Depo içeriği

| Konum | İçerik |
|---|---|
| `Notebooks/` | Final Colab demo, Python yardımcıları ve kullanım kılavuzu |
| `Docs/` | Final deney raporu, tez yöntem/sonuç taslağı ve roadmap belgeleri |
| `Deliverables/` | Yöntem karşılaştırması CSV, açıklaması ve Türkçe savunma sunumu |
| `Artifacts/Thesis_Figures/` | Başarı, hata ve özel görüntü demo görselleri |
| `Artifacts/*_review/` | Selector ve refinement görsel incelemeleri |
| `Artifacts/presentation_build/` | Sunum üretim kaynakları ve doğrulama çıktıları |

CSV: [yöntem karşılaştırması](Deliverables/DINOv3_Method_Comparison.csv) ve [kolon/birim açıklaması](Deliverables/DINOv3_Method_Comparison_Aciklama_TR.md). `Delta mIoU` yüzde puandır, `Inference Time` ms/görüntüdür, `Relative Cost` baseline'a göre süre oranıdır; FLOPs değildir.

Sunum: [10 slaytlık Türkçe savunma sunumu](Deliverables/DINOv3_Savunma_Sunumu_TR.pptx). Tez taslağı: [yöntem ve sonuçlar](Docs/DINOv3_Tez_Yontem_ve_Sonuclar_Taslagi_TR.md).

`Downloads/` Git takibinden çıkarılmıştır. Checkpoint ve büyük deney ZIP'leri ayrı yedekte tutulmalıdır. Eski ara raporlardaki değerler yerine güncel final raporu esas alınmalıdır.

## DINOv3 ve veri kaynakları

- [DINOv3 teknik raporu](https://arxiv.org/abs/2508.10104)
- [DINOv3 resmî deposu](https://github.com/facebookresearch/dinov3)
- [VOC2012 resmî sayfası](https://www.robots.ox.ac.uk/~vgg/projects/pascal/VOC/voc2012/)

Bu depodaki deney sonuçları proje kapsamında ölçülmüştür; DINOv3 yayınının benchmark sonuçları değildir. Model ve veri kullanımı için ilgili kaynakların lisans ve erişim koşulları geçerlidir.
