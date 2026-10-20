# Kayıtlı mIoU doğrulama raporu

**Durum: PASS.** Arşivdeki confusion matrix'lerden pooled mIoU yeniden hesaplandı; yeni inference yapılmadı.

Kapsam: aynı sıradaki 1.399 görüntü, 21 sınıf ve background dahil. Native çözünürlük ve ignore=255 protokol kaydından doğrulandı; ham GT maskeleri tekrar işlenmedi.

| Yöntem | Yeniden hesaplanan mIoU | Baseline farkı (puan) |
|---|---:|---:|
| baseline | 0.7132528531 | +0.0000 |
| adaptive_k1 | 0.7123846722 | -0.0868 |
| adaptive_k3 | 0.7219864665 | +0.8734 |
| random_42_k3 | 0.7147672956 | +0.1514 |
| random_43_k3 | 0.7115173819 | -0.1735 |
| random_44_k3 | 0.7157299539 | +0.2477 |
| random_k3_mean | 0.7140048772 | +0.0752 |
| sliding | 0.7324650975 | +1.9212 |

Arşiv summary dosyalarıyla tam hassasiyet karşılaştırması 1e-12 toleransında geçti. Final rapor ve CSV'nin dört ondalıklı mIoU değerleri yuvarlama toleransında eşleşti. CSV delta değerleri yuvarlatılmış rapordan üretildiği için tam hassasiyetli delta ile küçük yuvarlama farkı olabilir.

İki state dosyasının görüntü indeksleri, tamamlanma işaretleri, method listeleri ve yöntem başına geçerli piksel sayıları kontrol edildi. Random mean üç seed'in ayrı mIoU değerlerinin ortalamasıdır; ensemble değildir.

**0,7104 ile karşılaştırma yapılmadı:** bu değer ayrı 1.449 görüntü/224×224 değerlendirmesine aittir. Bootstrap güven aralıkları ve süre ölçümleri bu görevde yeniden hesaplanmadı.

Kaynak üyeleri ve hash'ler `metrics_verification.json` içinde. Tekrar çalıştırmak için proje kökünde `python Notebooks/verify_saved_metrics.py` kullan; NumPy ve yerel kaynak arşiv gerekir.
