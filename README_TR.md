# P-Core / E-Core Performans Analizi

## Intel Raptor Lake Üzerinde OpenMP Paralel Görüntü Filtreleme ve VTune Mikromimari Karşılaştırması

Bu depo, yüksek lisans tez çalışmam kapsamında geliştirilen OpenMP tabanlı paralel görüntü filtreleme algoritmalarını, kıyaslama altyapısını ve Intel VTune mikromimari analizlerini içermektedir.

Çalışmada, Intel hibrit işlemci mimarisindeki Performance-Core (P-Core) ve Efficient-Core (E-Core) çekirdeklerinin performans davranışları karşılaştırılmıştır.

---

## Tez Konusu

### P-CORE / E-CORE PERFORMANS ANALİZİ:
### VTUNE MİKROMİMARİ KARŞILAŞTIRMASI

Bu tez çalışmasında:

- Ortalama (Mean) filtre
- Gaussian filtre
- OpenMP paralelleştirme
- İş parçacığı afinite yönetimi
- Hızlanma ve verimlilik analizi
- Intel VTune Profiler mikromimari incelemeleri

gerçekleştirilmiştir.

Amaç, Intel Raptor Lake mimarisinde E-Core’ların görüntü işleme iş yüklerinde P-Core’lara yakın performans sağlayıp sağlayamadığını incelemektir.

---

## Donanım ve Yazılım Ortamı

| Bileşen | Özellik |
|---|---|
| İşlemci | Intel Core i7-14700KF |
| Mimari | Raptor Lake-DT |
| P-Core | 8 çekirdek / 16 thread |
| E-Core | 12 çekirdek |
| Toplam Mantıksal CPU | 28 |
| İşletim Sistemi | Ubuntu 24.04 LTS |
| Derleyici | g++ 14 |
| Paralelleştirme | OpenMP 5.2 |
| Profilleyici | Intel VTune Profiler 2024 |
| Görüntü Kütüphanesi | OpenCV4 |

---

## Test Yapılandırmaları

### Filtreler
- Mean Filter
- Gaussian Filter

### Kernel Boyutları
- k ∈ {3, 7, 11, 15, 19}

### İş Parçacığı Sayıları
- p ∈ {2, 4, 6, 8, 10, 12}

### OpenMP Affinity Yapılandırmaları
- P-Core
- E-Core
- Unset (İşletim sistemi zamanlayıcısı)

Her yapılandırma 100 kez bağımsız olarak çalıştırılmıştır.

---

## Temel Bulgular

Deney sonuçları göstermektedir ki:

- P-Core ve E-Core çalışma süreleri istatistiksel olarak neredeyse aynıdır.
- Maksimum performans farkı %2’nin altındadır.
- İş yükü büyük ölçüde L1/L2 önbellek içerisinde çalışmaktadır.
- DRAM bant genişliği kullanımı %5’in altında kalmaktadır.
- Derleyici otomatik SIMD vektörleştirmesi gerçekleştirmemiştir.
- Ölçeklenebilirliği sınırlayan temel faktör OpenMP senkronizasyon maliyetleridir.

---

## VTune Mikromimari Analizi

Çalışmada aşağıdaki VTune analizleri gerçekleştirilmiştir:

- HPC Performance
- Threading Analysis
- Memory Access
- Top-Down Microarchitecture Analysis (TMA)

Önemli gözlemler:

- LLC cache miss oranı ≈ 0
- Ortalama bellek gecikmesi ≈ 5 cycle
- Fiziksel çekirdek kullanımı <%30
- İş yükü bellek-bağımlı değil, hesaplama-bağımlıdır

---

## Paralelleştirme Stratejisi

Görüntü, karo tabanlı (grid decomposition) şekilde bölünmektedir.

Her OpenMP iş parçacığı bir görüntü karosunu işlemektedir.

Karo yapısı:

```math
⌈√p⌉ × ⌈p / √p⌉
```

şeklinde oluşturulmaktadır.

---

## Derleme

```bash
g++ main.cpp -O3 -march=native -fopenmp -fopenmp-simd -o benchmark
```

---

## Çalıştırma

```bash
./benchmark input.jpg
```

---

## Depo Yapısı

```text
.
├── common/               # Filtre kodları
├── gaussian/             # Gauss filtre main kodu
├── mean/                 # Ortalama filtre main kodu
├── vtune_results/        # VTune profiler raporları
├── logs/                 # Çıktı raporları
├── run_gaussian_vtune.sh # VTune profiler ile birlikte Gauss filtresi çalıştırma scripti
├── run_mean_vtune.sh     # VTune profiler ile birlikte Ortalama filtresi çalıştırma scripti
└── README.md
```

---

## Araştırma Katkıları

Bu çalışma:

- Hibrit işlemci mimarilerinde OpenMP davranışını incelemektedir.
- P-Core ve E-Core performans eşitliğini göstermektedir.
- SIMD vektörleştirme eksikliğinin performans üzerindeki etkisini ortaya koymaktadır.
- Enerji/verimlilik açısından E-Core kullanımının avantajlarını tartışmaktadır.

---

## Atıf

Türkan, A. (2026). Görüntü İyileştirme Yöntemlerinin GPU Tabanlı Sistemlerde Paralelleştirilmesi. 
Yüksek Lisans Tezi, Bolu Abant İzzet Baysal Üniversitesi, Bilgisayar Mühendisliği Ana Bilim Dalı.
