# Kesintisiz demo prova kaydı

Durum: **Dosya ön kontrolü geçti; süreli anlatım henüz yapılmadı.**

Bu kayıt, hazırlanmış 8 dakikalık akışın gerçek anlatım süresini ve sorunlarını belgelemek içindir. Temiz Colab testi veya dosyaların açılması tek başına prova tamamlandı anlamına gelmez.

## Başlamadan aç

1. [Anlatım akışı](DINOv3_Final_Demo_Akisi_TR.md).
2. [Index 9: başarı ve pipeline](Final_Visuals/success_index_9_8_panels.png).
3. [Kalite–maliyet grafiği](Quality_Cost/quality_cost.png) ve [benchmark tablosu](DINOv3_Method_Comparison.csv).
4. [Index 10: failure](Final_Visuals/failure_index_10_8_panels.png).

Canlı Colab hazırsa aynı index 9 çıktısını kullan; model yükleme/indirme işlemlerini prova sırasında başlatma. Oturum hazır değilse kayıtlı çıktı provası yap ve aşağıda modu açıkça kaydet. Bu, yeni canlı inference çalıştırıldığı anlamına gelmez.

## Tek seferlik deneme

Kronometreyi sıfırla, başlat ve yüksek sesle anlat. Takıldığında yeniden başlama; devam et ve sorun yaşanan adımı not et. Hedef akış:

| Kronometre | Bölüm |
|---|---|
| 00:00–00:20 | Problem ve amaç |
| 00:20–03:00 | Input, baseline, entropy, selector, crop'lar |
| 03:00–05:40 | Second pass, fusion, final çıktı ve karşılaştırma |
| 05:40–06:40 | Benchmark ve farklı örneklem kapsamları |
| 06:40–07:25 | Failure örneği |
| 07:25–08:00 | Sonuç ve sınırlılıklar |

Colab aksarsa kronometreyi durdurmadan kayıtlı index 9 görseline geç ve “Kayıtlı, doğrulanmış çıktıya geçiyorum” de. Uzun hata ayıklama yapma. Geçiş süresini sorun kaydına yaz.

## Gerçek deneme kaydı

- Tarih: ____
- Mod: canlı Colab / kayıtlı çıktılar / canlı başlayıp kayıtlı çıktıya geçiş
- Başlangıç: ____
- Bitiş: ____
- Kronometrede toplam süre (dk:sn): ____
- Tek seferde tamamlandı mı: ____
- On pipeline adımı gösterildi mi: ____
- Benchmark'ta 1.399 kalite ve 100 süre ayrımı söylendi mi: ____
- Failure ve sınırlılık açıklandı mı: ____

| Sorunun zamanı | Adım | Yaşanan sorun veya “yok” | Sonraki düzeltme |
|---|---|---|---|
| ____ | ____ | ____ | ____ |

## Kapatma ölçütü

Görev ancak gerçek deneme tamamlandığında, süre **05:00–10:00** aralığında olduğunda ve sorunlar (yoksa “yok”) kaydedildiğinde kapatılır. Süre dışındaysa kayıt korunur; akış kısaltılır/genişletilir ve ayrı bir deneme yapılır. Yukarıdaki planlanan 08:00 süresi gerçekleşmiş prova süresi yerine yazılmaz.

Prova sonunda paylaşılacak kısa çıktı:

```text
Süre: __ dakika __ saniye
Mod: __
Kesintisiz tamamlandı: evet/hayır
Sorun yaşanan adım: __ / sorun yok
```
