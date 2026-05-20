# Araba Gövde Tipi Sınıflandırma Projesi - Araştırmacı Çözümleri (Solutions)

Araştırmacı ajandan gelen rapor (Car Classification Project Challenges.pdf) incelenmiş ve karşılaştığımız potansiyel problemlere yönelik sunulan en iyi, modern ve pratik çözümler aşağıda özetlenmiştir:

## 1. Veri Seti Toplama ve Eksik Sınıflar
Standart veri setlerinde bulunmayan **F1 (Açık Tekerlekli)** ve **Micro** sınıfları için şu kaynaklar önerilmiştir:
*   **F1 Sınıfı:** Kaggle'daki `f1-image-classification-updated` veri seti. İçerisinde Ferrari, Mercedes vb. takımlara göre ayrılmış yüksek çözünürlüklü binlerce güncel F1 aracı görseli bulunmaktadır.
*   **Micro Sınıfı:** HuggingFace üzerindeki `Unit293/car_models_3887` veri seti (içerisindeki CSV dosyasından Fiat 500, Smart Fortwo gibi Micro modeller filtrelenebilir) ve özellikle arka planları halihazırda YOLOv11 ve `rembg` ile temizlenmiş olan `DamianBoborzi/car_images` veri seti.

## 2. Arka Plan Yanlılığı (Domain Bias) Çözümleri
Modelin arabayı değil de arka planı (pist, orman vb.) öğrenmesini engellemek için:
*   **Arka Plan Silme (Background Subtraction):** Çevrimdışı (offline) veri hazırlığı sırasında sadece 4.7 MB boyutunda olan ultra-hafif `U2NetP` modeli ile arka planlar silinerek yerine düz renkler veya Perlin gürültüsü (noise) konulmalıdır.
*   **Gelişmiş Veri Çoğaltma (Augmentation):** Araçları kesip birbiriyle alakasız arka planlara (örneğin F1 aracını çamurlu yola) yapıştırmak gibi yöntemler (background swapping) kullanılmalıdır.

## 3. 95 MB Sınırı ve Hızlı CPU Çıkarımı (Inference)
Modelin çok yer kaplamaması ve sunum sırasında standart bir CPU üzerinde anında sonuç verebilmesi için:
*   **Hafif Mimariler:** ResNet gibi ağır modeller yerine `MobileNetV4-Conv` veya `FastViT-S12` kullanılmalıdır. Bu modeller ham (FP32) halleriyle bile 20-30 MB yer kaplar ve çok hızlı çalışır.
*   **ONNX ve Kuantizasyon (Quantization):** PyTorch veya TensorFlow gibi devasa kütüphaneleri projeye dahil etmek 95 MB sınırını hemen aşar. Bunun yerine model `ONNX` formatına dönüştürülmeli ve sadece `onnxruntime` kütüphanesi ile çalıştırılmalıdır.
*   **Dynamic Quantization (Dinamik Kuantizasyon):** Model INT8 formatına dönüştürülerek 7 MB seviyelerine kadar küçültülmelidir. (Not: MobileNet modellerinde çökme olmaması için Statik yerine Dinamik Kuantizasyon ve `per_channel=True` parametresi kullanılmalıdır).

## 4. Birbirine Benzeyen Sınıfların Karışmasını Önleme (Fine-Grained Classification)
*   **Focal Loss:** Modelin kolay tahmin ettiği sınıflarla uğraşmayı bırakıp, sürekli karıştırdığı "Hatchback vs. Micro" veya "Station Wagon vs. SUV" gibi zor (hard) örneklere odaklanmasını sağlamak için standart Cross-Entropy yerine Focal Loss kullanılmalıdır.
*   **Bilgi Damıtma Tabanlı Etiket Düzeltme (Knowledge-Distillation-Based Label Smoothing):** Standart Label Smoothing yerine, modelin Station Wagon ile SUV'un birbirine benzediğini (ama F1'e benzemediğini) mantıksal olarak öğrenmesini sağlayan bu yaklaşım uygulanmalıdır.

## 5. Gradio ve React Entegrasyonu (Mimari Çözüm)
Modern bir arayüz geliştirirken 95 MB sınırında kalmak ve sunucuyu yormamak için:
*   **Headless Gradio ve FastAPI:** Gradio'nun kendi hantal arayüzü yerine sadece `gradio.Server` sınıfı çağrılarak arka uç (backend) tamamen "Headless" (arayüzsüz) bir FastAPI sunucusu olarak çalıştırılmalıdır.
*   **Vite ve Statik React Dağıtımı:** React uygulaması `Vite` ile derlenmeli (build). Çıkan `dist` klasörü sadece 2-5 MB boyutunda olacaktır. Bu statik dosyalar doğrudan FastAPI (`StaticFiles`) üzerinden sunulmalıdır.
*   **@gradio/client Kullanımı:** React içinden backend'e istek atarken ağır kütüphaneler (Axios vb.) yerine doğrudan `@gradio/client` kullanılmalıdır.

Bu çözümler projedeki her bir darboğazı ortadan kaldıracak, test setinde yüksek doğruluk (F1-score) verecek ve 95 MB kısıtlamasına rahatlıkla uyulmasını sağlayacaktır.
