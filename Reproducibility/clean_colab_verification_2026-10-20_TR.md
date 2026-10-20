# Temiz Colab demo testi — 20 Ekim 2026

**Sonuç: PASS.** Kullanıcı yeni ve boş GPU runtime bildirdi; kurulum, config, checkpoint restore, model ve VOC veri yükleme çıktılarını paylaştı. Son iki örnek testi ZIP olarak indirildi ve yerelde doğrulandı.

| Örnek | Baseline image mIoU | Adaptive K=3 image mIoU | Durum |
|---|---:|---:|---|
| Index 9 | 0.8300664277 | 0.9512478780 | PASS |
| Index 10 | 0.4658147988 | 0.3207856414 | PASS |

İki örnekte de seçilen K=3 ve toplam geçiş sayısı 4. Index 10, yöntemin kötüleştiği beklenen örnektir; testin geçmesi performansın her örnekte arttığı anlamına gelmez.

## Doğrulanan kanıtlar

- ZIP CRC kontrolü ve 32 çıktı dosyasının SHA-256 değeri geçti.
- Skorlar PNG sınıf maskeleri ve GT'den, ignore=255 ve background dahil olarak yeniden hesaplandı.
- Crop birleşimi dışında baseline ve final prediction aynı kaldı.
- Runtime config hash'i raporla, config anlamı yerel TOML ile eşleşti. Ham config baytları satır sonu nedeniyle farklılaşabilir.
- Checkpoint hash'i ve model revision doğru; pipeline hücre hash'leri yerel final notebook ile eşleşti.
- İki yedi panelli final görseli yerelde incelendi.

## Ortam

Python 3.13.15; Torch 2.11.0+cu130; CUDA 13.0; GPU Tesla T4.
Paketler: {"transformers": "5.17.0", "huggingface_hub": "1.33.0", "torchvision": "0.26.0+cu130", "numpy": "2.1.3", "Pillow": "11.3.0", "matplotlib": "3.10.0"}.

## Kaynak ve kapsam

Arşiv: `Downloads/DINOv3_clean_smoke_20261020T073625_596320Z.zip`. SHA-256: `ac041bbdfc9350d3f7e3b6a9ecd8d0a206c34702985768dd4fe821b5a98466ae`.
Makine-okunur doğrulama: `clean_colab_verification_2026-10-20.json`. Orijinal test raporu: `clean_colab_smoke_report_2026-10-20.json`.
Bu test config bağlantılı inference/export zincirini iki referans görüntüde doğrular. Eğitimin yeniden üretimi veya 1.399 görüntülük kalite / 100 görüntülük latency benchmark'ının yeniden çalıştırılması değildir. Oturumun yeni olduğu bilgisi kullanıcı beyanı ve paylaşılan kurulum adımlarına dayanır.
Eski experiment_manifest.json tarihsel snapshot olarak korunmuştur; bu 20 Ekim kanıtı onu tamamlar. Kaynak ZIP Downloads altında Git dışında kalır ve ayrı yedeklenmelidir.
