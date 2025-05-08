# Şerit Tespiti (Lane Detection)

Bu proje, video üzerinde şerit tespiti yapan bir Python uygulamasıdır. OpenCV ve NumPy kütüphaneleri kullanılarak geliştirilmiştir.

## Ana Fonksiyonlar

1. **Gri Tonlama**: Görüntüyü gri tonlamaya çevirir
2. **Gaussian Blur**: Gürültüyü azaltmak için bulanıklaştırma uygular
3. **Canny Kenar Tespiti**: Kenarları tespit eder
4. **ROI (İlgi Alanı) Maskesi**: Sadece yol şeritlerinin olduğu bölgeyi maskeleme
5. **Hough Çizgileri**: Şeritleri tespit eder ve çizer
   - Sol ve sağ şeritleri ayrı ayrı gruplar
   - Her grup için ortalama çizgi hesaplar
   - Şeritleri görüntü üzerine çizer
6. **Görselleştirme**: 
   - Tespit edilen şeritleri yeşil çizgilerle gösterir
   - Şeritler arası alanı yarı saydam mavi ile doldurur
   - ROI köşe noktalarını kırmızı noktalarla işaretler
