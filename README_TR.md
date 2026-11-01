# Yüksek Performanslı Benzerlik Arama Motoru

**YZM 513 – Yapay Zekâ için Yüksek Performanslı Programlama · Proje 1** · [English](README.md)

Güncel 20.000 arXiv makale özeti (cs.AI / cs.CL / cs.LG, Ocak 2025 – Eylül 2026) ve 200 sorgu veriliyor; her sorgu için en benzer
10 dokümanı döndür ve bunu hızlı yap. Saf Python bir baseline ile başlıyor, onu profilliyor ve darboğazları tek tek gideriyoruz
(vektörizasyon, Numba, boyut indirgeme, paralellik). Her sürüm baseline ile 1e-5 toleransla doğrulanıyor.

* Rapor: [`report/RAPOR.md`](report/RAPOR.md) (PDF: `report/RAPOR.pdf`) · İngilizce: [`report/REPORT.md`](report/REPORT.md)
* Bütün figürler ve ham benchmark verisi: [`results/`](results/) (her proje fazı için bir klasör)

Ortak raporun yazarları Ahmet Ozan Yıldırım ve Mazhar Selvitopi'dir. Yeni ARM64 ölçümleri aynı raporun Ek F bölümündedir.
Bu ek, 128-d TF-IDF/SVD temsiliyle yapılan 32 karşılaştırmayı içerir; Windows/MPNet sonuçlarıyla hızlanma hesabında karıştırılmaz.

| | |
|---|---|
| Veri seti | arXiv API, 20.000 özet (doküman) + 200 makale başlığı (sorgu); doğru cevap başlığın kendi özeti |
| Embedding | `sentence-transformers/all-mpnet-base-v2`, 768 boyut, float32 (üretim süresi benchmark'a dahil değil) |
| Son sistem | bir kez normalize → TruncatedSVD ile 128 boyut → toplu BLAS arama → paralel Numba top-k (100 aday) → kesin 768-d rerank → top-10 |
| Donanım | AMD Ryzen 7 9800X3D (8 çekirdek / 16 thread, 96 MB L3), 32 GB RAM, Windows 11, Python 3.12 |

## Klasör yapısı

```
├── README.md / README_TR.md
├── requirements.txt, pyproject.toml     # `pip install -e .` ile `hpss` her yerden (worker process'ler dahil) import edilebilir
├── src/hpss/
│   ├── config.py, data.py, bench.py, plotting.py     # yollar, veri seti, ölçüm (warm-up + ≥5 tekrar), figür stili
│   ├── naive.py          # Faz 1: saf Python baseline (+ Python düzeyinde düzeltmeler)
│   ├── sparse_dense.py   # Faz 2A: TF-IDF/CSR ve kendi sparse·sparse / sparse·dense / dense·dense çekirdeklerimiz
│   ├── svd.py            # Faz 2B: kendi blok power-iteration SVD'miz + kütüphane yöntemleri
│   ├── kernels.py        # Faz 2C: kosinüs matrisi beş yolla; birleşik top-k arama; paralel top-k
│   ├── parallel.py       # Faz 3: BLAS, Numba, thread, process (+paylaşımlı bellek), asyncio, Dask, Ray
│   ├── search.py         # Faz 4: SearchEngine (son sistem)
│   └── experiments/      # her faz için bir modül — rapordaki bütün figür ve sayıları üretir
├── scripts/              # prepare_data, run_all, collect_results, demo, profile_naive
├── data/processed/       # corpus.jsonl, queries.jsonl (embedding'ler prepare_data.py ile yeniden üretilir)
├── results/<faz>/        # figürler (PNG + PDF), ham CSV/JSON, temel sayılar ve başlıklarla results.json
└── report/               # REPORT.md (EN), RAPOR.md (TR), results_overview.md
```

## Yeniden üretme

```bash
python -m venv .venv
.venv\Scripts\activate                 # Linux/macOS/Colab: source .venv/bin/activate
pip install -r requirements.txt
pip install -e .

python scripts/prepare_data.py         # ~10 dk indirme + dizüstü CPU'da ~40 dk embedding (GPU'da 1-2 dk); önbelleğe alınır
python scripts/run_all.py              # bütün fazlar, test makinesinde ~70 dk (tam iş yükleri, tahmin yok); ya da ör. `python scripts/run_all.py 1 2a`
python scripts/collect_results.py      # bütün sayı ve başlıkların özeti
python scripts/demo.py --compare-naive "speculative decoding for LLM inference"
```

Faz sırası önemli: sonraki fazlar sonuçlarını Faz 1'in yazdığı naif baseline referansıyla doğruluyor. Her fazın komutu ve çıktı
klasörü [README.md](README.md#reproduce) tablosunda.

**İsteğe bağlı - GIL'siz Python (Faz 3, GIL deneyi):** aynı saf Python thread testi GIL'siz CPython 3.14t (PEP 703) ile de çalıştırılıyor.
Kurulum: `uv python install 3.14t` (ya da `HPSS_FT_PYTHON` değişkenini herhangi bir `python3.14t`'ye yönlendirin); `scripts/gil_free_threading.py`
sadece standart kütüphaneyi kullanıyor. Kurulu değilse o kısım bir not yazıp atlanıyor.

## Ölçüm yöntemi

* Her ölçüm: warm-up (JIT derlenen her şey için 3) + **en az 5 ölçüm**; **medyan** raporlanıyor (ortalama, std ve ham süreler `results/` içinde).
* Her sürüm için aynı makine, aynı veri ve sorgular. Bir karşılaştırma gerektirdiğinde BLAS / Numba thread sayıları açıkça ayarlanıyor
  (ör. Faz 2A'da tek thread).
* Doğruluk: her optimize sürüm, naif baseline ile (`results/1_baseline/baseline_topk.npz`) sıra sıra, 1e-5 toleransla karşılaştırılıyor.
* Embedding üretimi ve indeks kurulumu (normalizasyon, SVD, ters indeks) bir kez yapılıp ayrı raporlanıyor; sadece arama süresi ölçülüyor.
* `import hpss`, `OMP_WAIT_POLICY=PASSIVE` / `KMP_BLOCKTIME=0` ayarlıyor: boştaki Numba (OpenMP) thread'leri boşta dönüp OpenBLAS'ın
  çekirdeklerini çalmak yerine uyuyor (Faz 4'te ölçüldü; kapatmak için `HPSS_NO_THREAD_FIX=1`).
* Ölçüm sırasında başka ağır programları kapatın.

## Ek doğrulamayı çalıştırma

```bash
python -m pytest tests -q
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 NUMBA_NUM_THREADS=4 \
python scripts/validate_independent.py --output results/validation_arm64/new_run
```

Bu komut kayıtlı JSONL metinlerini kullanır; model indirmez ve eski sonuçları ezmez. Her yapılandırma için 3 ısınma ve
5 zamanlı tekrar yapar. Çıktı dizini mevcutsa yeni bir dizin seçilmesini ister. Ham süreler, doğruluk ve SHA-256
kayıtları `summary.json` içinde tutulur. Ana MPNet deneylerinin kayıtlı metinlerden tekrar üretimi için
`python scripts/prepare_data.py --from-processed` kullanılabilir; bu komut embedding üretir.
