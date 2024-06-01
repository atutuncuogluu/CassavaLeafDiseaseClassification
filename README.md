# Rapor

**Ahmet Tütüncüoğlu**  
21120205057

## Adım 1 - Probleme yaklaşım ve veri setini inceleme

Kaynak veri seti incelendiğinde 21397 veri varken etiketli olan sadece 17.938 veri bulunmaktadır. Literatür ve kaynaklar tarandığında eksik olan verilerin de etiketlendiği veri setine [1] Kaggle platformu üzerinden erişilmiştir.

Veri seti incelendiğinde sınıfların dengesiz (unbalanced) dağılımı tespit edilmiş ve [Resim 1] sınıfların bar grafiği çizdirilmiş olup tüm süreç boyunca araştırılan ve uygulanan yöntemler bu dengesizlik göze alınarak yapılmıştır.

![Resim 1](https://www.kaggle.com/competitions/cassava-leaf-disease-classification)

Son olarak veri setindeki resimlerin boyutunu ve genel bitki yapısını gözlemlemek amacıyla 4 adet sample alınıp görselleştirilmiştir [Resim 2].

![Resim 2](https://www.kaggle.com/competitions/cassava-leaf-disease-classification)

## Adım 2 – Probleme dair literatür taraması

Araştırmalarım sonucu probleme ait birçok çözüm içeren makale ve uygulama bulsam da çözüm noktasında ele aldığım temel kaynaklardan birisinde [2] probleme herhangi başka data eklemeden Cutmix, Cutout ve Mixup preproccess yöntemleriyle eğitilen modellerin accuracy skorunda belirli bir artış elde edildiği gözlemleniyor. Bu sebeple ben de modellerimde bu yöntemlerden Cutout ve Cutmix işlemine yer verdim.

Ayrıca veri arttırma noktasında (data augmentation) Torchvision kütüphanesinin transforms metodu probleme dair birçok uygulama ve makalede yer aldığı için projemde ben de bu kütüphaneyi kullandım. Bu metodla birlikte veriye randomflip, randomrotation, gaussianblur, colorjitter, randomresizedcrop ve ImageNet veri setindeki RGB değerlerin ortalaması ve standart sapması normalize fonksiyonu kullanılarak uygulanmıştır [Resim 3].

![Resim 3](https://sensors.myu-group.co.jp/sm_pdf/SM2743.pdf)

## Adım 3 - Eğitim için gerekli fonksiyonlar ve dosyaların hazırlanması

Kullanacağım yöntemde verileri dataframe üzerinde pathleri ekleyip kullanmak yerine her class için train ve test adlı 2 ayrı klasör oluşturup Shutil kütüphanesiyle bu klasörlere classlara ait verileri ekledim. Böylece daha nizami ve belirli bir formata uygun veri seti elde etmiş oldum.

Ardından torch.utils kütüphanesi içerisinde bulunan Dataset class’ını ImageFolderCustom class’ım içerisinde inherit ederek yine aynı kütüphane içerisinde DataLoader class’ı formatına uygun ve transform fonksiyonu uygulanabilecek bir düzene getirdim.

Tüm bu yöntemler sayesinde veri setimi batchlere ayırıp hız, hafıza verimliliği ve paralel çalışma noktasında avantaj elde ettim.

## Adım 4 – GoogleNet, AlexNet, VGGNet ve RestNet mimarilerinin araştırılması

GoogleNet için literatür taraması yapılıp orijinal makalesi [3] bulunmuş olup bu makaleye uygun mimari, modelde birebir uygulanmıştır [Resim 4].

![Resim 4](https://arxiv.org/pdf/1409.4842)

AlexNet için kullanılan mimari ders kaynaklarından alınmış olup modelde birebir uygulanmıştır [Resim 5].

![Resim 5](https://arxiv.org/pdf/1409.4842)

VGGNet için birçok alternatif versiyon araştırmalarım sonucu bulunduysa da kullandığım mimari versiyonu ders kaynaklarından alınmıştır [Resim 6].

![Resim 6](https://arxiv.org/pdf/1409.4842)

RestNet mimarisi için literatür taraması sonucu orijinal makaleye [4] ulaşılmış olup makalede yer alan block tipine göre çeşitli RestNet varyasyonlarından basic block ve bottleneck block kullanılarak sırasıyla modelde RestNet34 ve RestNet152 varyasyonları kullanılmıştır [Resim 7].

![Resim 7](https://arxiv.org/pdf/1512.03385v1)

## Adım 5 – Model, eğitim ve test fonksiyonlarının oluşturulması

Her model pytorch temelli torch.nn modülünü inherit ederek belirtilen CNN mimarilerini oluşturmaktadır. Bazı modellerde ek olarak batch normalization, dropout kullanılarak model, eğitim ve test verimliliği sağlanmıştır. Her model için Adam optimizer ve learning rate değerini güncellemek için ise ReduceLROnPlateau scheduler kullanılmıştır.

Train ve test aşaması için oluşturduğum train_step ve test_step fonksiyonları ile modelin eğitimi sırasında elde edilen verileri kaydedip görsel olarak sunumu için hazırlık yapılıyor. Ayrıca train_step fonksiyonunda önceden oluşturduğum cutout ya da cutmix fonksiyonlarından birisi 0.5 ihtimalle veriye uygulanıyor. Tüm bunlar dışında her iki fonksiyon da standart olarak şu işlemleri yapar:

1. Modeli kullanarak raw tahminleri al
2. Gerçek değerler ve raw tahminleri CrossEntropyLoss kullanarak kıyasla
3. Optimizer türevini sıfırla
4. Backpropagation uygula
5. Optimizer’ı uygula

Son olarak train fonksiyonu belirtilen epoch sayısı kadar train_step ve test_step fonksiyonlarını kullanarak eğitim ve test işlemlerini yapıp belirtilen scheduler ile learning rate güncellenir ardından results değişkenine metrikleri kaydeder.

## Adım 6 – Modeller ve ortalama metrikleri tablosu

| Model       | Epoch | Loss | Accuracy | Precision | Recall | F1-Score |
|-------------|-------|------|----------|-----------|--------|----------|
| VGGNet      | 35    | 0.71 | 0.74     | 0.61      | 0.58   | 0.57     |
| AlexNet     | 25    | 0.96 | 0.66     | 0.44      | 0.35   | 0.35     |
| GoogleNet   | 40    | 0.81 | 0.70     | 0.56      | 0.51   | 0.51     |
| RestNet34   | 40    | 0.74 | 0.73     | 0.60      | 0.53   | 0.54     |
| RestNet152  | 40    | 0.85 | 0.69     | 0.52      | 0.45   | 0.46     |

### AlexNet

### GoogleNet

### RestNet34

### RestNet152

### VGGNet
