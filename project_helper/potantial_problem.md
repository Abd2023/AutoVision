# Araba Gövde Tipi Sınıflandırma Projesi - Potansiyel Problemler ve Riskler

Proje dokümanını (Yazlab 2- Proje 3) inceledikten sonra, geliştirme sürecinde karşılaşabileceğimiz en kritik ve önemli potansiyel problemler aşağıda listelenmiştir:

## 1. Veri Seti Dengesizliği ve "Domain Bias" (Arka Plan Yanlılığı)
*   **Problem:** Sizin de belirttiğiniz gibi, hazır veri setlerinde **Micro** ve **Açık Tekerlekli (F1)** sınıfları eksik. Bu sınıfları farklı veri setlerinden topladığımızda, fotoğrafların çekildiği ortamlar arasında uçurum olabilir. Örneğin; F1 araçları her zaman yarış pistlerinde, SUV'ler ise orman/arazi ortamında olabilir. Model, arabanın gövdesini öğrenmek yerine arka planı (pist, orman, otoyol) ezberleyebilir (Overfitting).
*   **Risk:** Tamamen yeni test verisi geldiğinde, arka plan farklıysa model yanlış tahmin yapacaktır. Test verisinin "hiç görülmemiş" olacağı özellikle vurgulandığı için genelleme (generalization) çok kritiktir.

## 2. Model Boyutu Kısıtlaması (Maksimum 95 MB)
*   **Problem:** Projenin nihai teslim boyutunun **95 MB'ı aşmaması** gerektiği belirtilmiş. Geleneksel güçlü modeller (örneğin ResNet50, Vision Transformer - ViT) tek başlarına 100 MB ile 350 MB arasında yer kaplayabilmektedir.
*   **Risk:** Doğruluğu yüksek tutmak için büyük bir model seçersek, e-destek sistemine yükleme aşamasında sınırın üstünde kalırız ve modeli küçültmek için zamanımız kalmayabilir. Lightweight (hafif) modeller seçilmeli (örn: MobileNetV3, EfficientNet-B0) veya Quantization (küçültme) teknikleri uygulanmalıdır.

## 3. Görsel Benzerlikleri Yüksek Olan Sınıfların Karışması
*   **Problem:** Sınıflar arasında görsel olarak birbirine çok benzeyen tipler mevcut. Özellikle:
    *   **Hatchback** vs. **Micro** (İkisi de küçük ve arkası kesik).
    *   **SUV** vs. **Station Wagon** (Uzun ve geniş yapıları benziyor).
*   **Risk:** F1-Skoru en önemli metrik olarak belirlenmiş. Bu benzer sınıflar yüzünden Confusion Matrix'te (Karışıklık Matrisi) ciddi çapraz hatalar görebiliriz. Modelin bu ince detayları (fine-grained classification) öğrenebilmesi için "Data Augmentation" (veri çoğaltma) adımlarının çok dikkatli yapılması gerekecek.

## 4. Test Scripti Entegrasyonu ve Arayüz Gecikmesi (Latency)
*   **Problem:** Hocaların sunum sırasında kendi **test script'lerini** kullanacakları ve arayüze entegre edecekleri belirtilmiş. Ayrıca, sınıflandırma süresinin kabul edilebilir süreyi aşması durumunda "puan kırılacağı" söylenmiş. React + Gradio kullanacağımız için, tahmin API'sinin yanıt süresi düşük olmalıdır.
*   **Risk:** Model ağır olursa veya test scripti arayüz API'mizle uyumlu tasarlanmazsa, canlı sunumda tahminler yavaş dönebilir veya script patlayabilir. Hocaların scriptinin beklediği input/output formatlarına tam uyum sağlamak zorundayız.

## 5. Gradio ve React Entegrasyonu (Mimari Karmaşıklık)
*   **Problem:** Modern bir arayüz için Gradio ve React'i birlikte kullanmak istiyoruz. Ancak Gradio kendi başına bir Frontend sunarken, React de bir Frontend kütüphanesidir.
*   **Risk:** İkisini birleştirmek (örneğin Gradio'yu sadece Backend/API (Gradio Client) olarak kullanıp Frontend'i React ile yazmak) mimariyi karmaşıklaştırabilir. 95 MB sınırına, React derlenmiş dosyaları (build) ve Python kütüphanelerinin de dahil edilme ihtimalini göz önünde bulundurmalıyız.
