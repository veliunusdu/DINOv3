# Refinement görsel incelemesi — index 10 ve 14

Kaynak: refinement_diagnosis_20261006T222248_223281Z.zip. İki PNG doğrudan incelendi; sayısal sonuçlar kullanıcının paylaştığı diagnosis çıktısıyla birlikte değerlendirildi.

## Index 10 — 2007_000452

Yeni hatalar yalnızca crop sınırında değil, sofa olarak etiketlenmiş nesnenin iç kısmında geniş bölgelerde oluşuyor. Yerel tahmin bu nesneyi sofa (yeşil), chair (kırmızı), background (siyah) ve cat (koyu kırmızı) bölgelerine ayırıyor. Sofa IoU 0.5086'dan 0.3314'e düşüyor. 6010 piksel düzelirken 11697 yeni hata oluşuyor; toplam yanlış piksel sayısı 5687 artıyor.

Orijinal görüntüde kedi görülebiliyor, fakat GT'de cat etiketli geçerli piksel yok. Bu nedenle yerel cat tahmini benchmark açısından hatalı ve görüntü mIoU'suna sıfır IoU'lu yeni bir sınıf ekliyor. Bu, gerçek nesnenin mutlaka uydurulduğu anlamına gelmez; tahmin ile kullanılan anotasyonun uyuşmadığını gösterir.

## Index 14 — 2007_000559

Yerel tahmin ekranın içindeki şişe benzeri şekli bottle olarak etiketliyor; çevresindeki ekranın geniş bölümünü background yapıyor. GT ise ekran yüzeyini tvmonitor olarak etiketliyor. Yeni hatalar ekranın içinde yayılıyor; salt crop kenarı artefaktı olarak açıklanamaz. Monitör IoU 0.5110'dan 0.2055'e düşüyor. 844 piksel düzelirken 12992 yeni hata oluşuyor; toplam yanlış piksel sayısı 12148 artıyor.

Masadaki GT bottle nesnesi crop dışında kalıyor ve tespit edilmiyor. Ekranda üretilen bottle tahmini GT şişesini düzeltmiyor; yanlış pozitif oluşturuyor.

## Sonuç ve sınırlar

Bu iki örnekte global prediction'ı yerel prediction ile tamamen değiştirmek geniş iç bölgelerde semantik sınıf değişimleri üretiyor. Görüntüler, başarısızlığın yalnızca crop sınırlarında oluşmadığını gösteriyor. Ölçek ve bağlam değişiminin etkisi olasıdır, fakat bunların nedensel etkisini bu iki görselden ayrı ayrı doğrulayamayız.

Selector'ın hata bölgesini kapsaması, yerel tahminin doğru olacağını garanti etmiyor. Genel Top-K/NMS özelliklerinin tamamlanması kalite artışını garanti etmez. Kaliteyi iyileştirmek için geliştirilecek yöntemlerde global/local tahmin uyuşmazlığı ve yeni yanlış pozitifler ayrıca değerlendirilmelidir. Mevcut 1399 görüntülük sabit deney değiştirilmemelidir.
