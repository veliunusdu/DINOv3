# Selector görsel incelemesi — 20 geliştirme görüntüsü

Kaynak: selector_gallery_20_20261006T221453_694935Z.zip. İlk üç görsel sohbet eklerinden, kalan 17 görsel ZIP'ten incelendi. Aşağıdaki oranlar ZIP içindeki gallery_metadata.json kayıtlarından alınmıştır. Karşılaştırılan random seed 42'dir; üç seed ortalaması değildir.

## Bulgular

Adaptive hata kapsaması 20 örneğin 18'inde random_42'den yüksek, 2'sinde düşüktür. Bu örnekler 50 görüntülük geliştirme grubunun ilk 20 kaydıdır; bağımsız değerlendirme değildir. Oracle GT hedef sınıfına göre oluşturulmuş değişken boyutlu referans kutusudur; eşit alanlı karşılaştırma değildir.

| Index | Hedef | Adaptive / random hata kapsaması (%) | Görsel gözlem |
|---|---|---|---|
| 9 | bottle | 58.5 / 14.3 | Şişe gövdesindeki hatalar seçiliyor; üst kısmı kesiliyor. |
| 10 | chair | 76.9 / 22.1 | Soldaki büyük nesne seçiliyor; sağ kenardaki küçük hedef dışarıda. |
| 11 | cow | 99.7 / 60.2 | İki inek ve neredeyse bütün hatalar kapsanıyor. |
| 13 | boat | 87.0 / 57.9 | Global tahminde kaybolan gemi bölgesi seçiliyor; alt kısmı kesiliyor. |
| 14 | bottle | 59.8 / 45.1 | Monitör hataları seçiliyor; küçük şişe dışarıda. |
| 18 | pottedplant | 69.6 / 43.8 | Kanepe çevresi seçiliyor; bitkinin büyük kısmı dışarıda. |
| 21 | bus | 4.0 / 0.0 | Yüksek entropy tabela bölgesine yöneliyor; otobüsü kaçırıyor. |
| 22 | bottle | 65.1 / 58.9 | Masa çevresi ve şişenin bir kısmı kapsanıyor; şişe kesiliyor. |
| 23 | person | 60.1 / 51.7 | At ve binicinin alt bölgeleri seçiliyor; binicinin başı dışarıda. |
| 27 | boat | 90.6 / 50.3 | Tekne gövdesi seçiliyor; direğin üstü dışarıda. |
| 31 | bottle | 76.2 / 9.9 | Baş ve şapka sınır hataları seçiliyor; şişe dışarıda. |
| 33 | tvmonitor | 42.2 / 40.0 | Oturma alanı ve bacak çevresi seçiliyor; monitör dışarıda. Ignore alanı geniş. |
| 37 | bird | 100.0 / 53.2 | Kuşun tamamı ve bütün geçerli hatalar kapsanıyor. |
| 42 | pottedplant | 11.7 / 14.5 | Ön şişe bölgesi seçiliyor; bitki dışarıda. Random daha fazla hata kapsıyor. |
| 44 | tvmonitor | 24.0 / 8.3 | Masa ve insan çevresi seçiliyor; monitör dışarıda. |
| 45 | chair | 37.7 / 22.8 | Kanepe seçiliyor; sağdaki küçük sandalye dışarıda. |
| 49 | person | 60.9 / 32.5 | İnsan ve kanepe çevresi kapsanıyor; ayaklar kesiliyor. |
| 50 | person | 79.8 / 17.0 | Binici ve atın ön bölgesindeki hatalar kapsanıyor; nesneler kesiliyor. |
| 51 | bicycle | 0.0 / 39.9 | Duvar seçiliyor; tamamen kaçırılmış bisiklet dışarıda. |
| 52 | dog | 96.5 / 75.1 | Köpek ve koyunlar kapsanıyor; insanın bir kısmı kesiliyor. |

## Yorum ve sınırlar

Görseller ve 50 görüntülük hata kapsama ölçümü, entropy seçiminin bu geliştirme grubunda rastgele seçime göre anlamlı hata bölgelerine yöneldiğini destekliyor. Küçük nesneyi seçmek ve nesneyi bütünüyle kapsamak garanti edilmiyor. Index 51, düşük entropy ile kaçırılan nesne örneğidir; index 21 yüksek entropy'nin yoğun GT hatasına karşılık gelmediği örnektir.

Bu galeri global tahminleri gösteriyor; crop sonrası prediction içermiyor. Dolayısıyla refinement başarısızlığının nedeni veya bağlam kaybının etkisi bu görsellerden doğrulanamaz. 1399 görüntülük deneyde adaptive'ın baseline/random'a üstünlüğü gösterilemedi ve sliding'e karşı mIoU farkı -2.01 puandı (bootstrap %95 aralık [-2.79, -1.24]).

Sonraki analiz: index 10 ve 14 gibi önceki refinement başarısızlıklarında global, crop sonrası tahmin, düzeltilen pikseller ve yeni hataları karşılaştırmak; index 21 ve 51'i selector başarısızlık örnekleri olarak korumak.
