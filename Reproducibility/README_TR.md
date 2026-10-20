# Deney manifest’i

`experiment_manifest.json`, kayıtlı deneylerin checkpoint, veri, sonuç ve kod kaynaklarını birleştirir. Yeni inference yapılmadı.

## İçerik

- Checkpoint SHA-256 ve model revision arşivle doğrulandı.
- `splits/`: validation 1.449, development 50, ana değerlendirme 1.399 ve latency 100 için sıralı görüntü ID listeleri ve indeks eşleşmeleri.
- `code_snapshot/`: manifest oluşturulurken kullanılan güncel notebook, config ve yardımcı kodların birebir kopyaları. Manifest’te SHA-256 değerleri bulunur.
- `historical_model_config.json`: arşivdeki model/eğitim parametrelerinin kopyası.
- ZIP kaynaklarının dosya boyutları ve SHA-256 değerleri; okunan arşiv üyelerinin ayrı hash’leri.

## Önemli sınırlar

Güncel Git HEAD, geçmiş deneylerin çalıştırıldığı commit değildir. Config bağlantılı güncel kod henüz commit edilmemiştir; bu nedenle exact dosya kopyaları ve hash’leri saklandı. Eski notebook’un hash’i de kayıtlıdır ancak bütün deneylerin eksiksiz orijinal kod sürümü olduğu iddia edilmez.

Training 1.464 görüntünün tam ID listesi incelenen arşivlerde bulunmadı. Bu alan `null` olarak işaretlendi; tahminle doldurulmadı. Tarihsel ortamdan Tesla T4, Torch ve Transformers sürümleri timing protokolünden doğrulandı; Python/CUDA/torchvision ve tam pip freeze bu kaynakta mevcut değil. Yerel manifest üreticisinin sürümleri Colab ortamı olarak gösterilmez.

Latency 100, evaluation 1.399 içinde yer alır. Development 50 dışarıdadır. Bu gruplar dokunulmamış test olarak sunulmaz. Manifest ve hash’ler kaynakları tanımlar; tek başına yeni inference veya eğitim reproduksiyonunun kanıtı değildir.

## Yeniden üretme

Proje kökünden, NumPy içeren Python 3.11+ ile `python Notebooks/build_experiment_manifest.py` çalıştır. `Downloads/` arşivleri ve manifest’teki kaynak dosyalar yerelde bulunmalıdır. Kaynaklar veya kod değiştiğinde manifest’i yenilemeden önce önceki manifest paketini ayrı sakla. ZIP’ler GitHub’da değildir; ayrı yedek gerekir.
