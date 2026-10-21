# Sustainability Analyzer

Sürdürülebilirlik raporlarını PDF olarak işleyen, metin ve tablolarından kanıt
getiren ve soruları dil modelleriyle yanıtlayan Python komut satırı uygulaması.
Soru kataloğu üzerinden kategori bazında analiz ve Excel çıktısı üretir.

## Kurulum

Python 3.11 veya 3.12 kullanın. OCR için Tesseract ve İngilizce/Türkçe dil
verileri (`eng`, `tur`) gereklidir. Yerel kurulumda `tesseract` komutu PATH
üzerinden erişilebilir olmalıdır. Docker imajı bu sistem bağımlılıklarını kurar.

Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
Copy-Item .env.example .env
.\.venv\Scripts\python.exe main.py --help
```

Linux/macOS (Tesseract ayrıca kurulmalıdır):

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
cp .env.example .env
.venv/bin/python main.py --help
```

Aşağıdaki örneklerde sanal ortamın Python yorumlayıcısını kullanın veya ortamı
etkinleştirin.

## Yapılandırma

`.env` içine yalnızca kullanacağınız servislerin anahtarlarını yazın:

| Ayar | Kullanım |
| --- | --- |
| `LLM_PROVIDER` | `groq` (varsayılan), `gemini`, `openrouter`, `ollama`, `openai_compatible` |
| `GROQ_API_KEY`, `GEMINI_API_KEY`, `OPENROUTER_API_KEY`, `OPENAI_API_KEY` | Seçilen sağlayıcının anahtarı |
| `LLM_API_KEY` | Sağlayıcıya özel anahtarın yerine kullanılabilen ortak ayar |
| `LLM_MODEL` | Hesabınızda erişebildiğiniz model; boşsa uygulama varsayılanı |
| `LLM_BASE_URL` | Özel OpenAI uyumlu sunucu adresi |
| `EXA_API_KEY` | Kaynak verilmediğinde otomatik rapor arama |
| `VISION_*` | İsteğe bağlı multimodal yanıt ayarları |
| `STORAGE_BACKEND` | `local` veya `azure_blob` |

Yerel PDF veya doğrudan URL ile çalışırken Exa anahtarı gerekmez. Yanıt
üretmek için seçtiğiniz model servisini yapılandırın. Ollama kullanıyorsanız
sunucuyu ayrıca çalıştırın ve kurulu modelin adını `LLM_MODEL` ile belirtin.
Görsel fallback seçenekleri varsayılan olarak kapalıdır; ingestion sırasında
`--enable-vlm-fallback` Groq anahtarı ve uygun vision modeli gerektirir.

API anahtarlarını komut satırı argümanları yerine `.env` veya ortam değişkenleri
ile sağlayın. `.env` Git ve Docker build bağlamından dışlanır.

## Kullanım

Yerel raporu işleme:

```bash
python main.py ingest --company "Example Company" --year 2023 --pdf-path "data/report.pdf" --report-id example_2023
```

İşlenmiş rapora soru sorma:

```bash
python main.py ask --report-id example_2023 --question "Toplam su tüketimi nedir?"
```

Katalogdaki su sorularını yanıtlama ve Excel çıktısı üretme:

```bash
python main.py batch --company "Example Company" --year 2023 --report-id example_2023 --skip-ingestion --questions-file config_data/questions.json --groups water --output-directory outputs
```

Yeni bir raporda `batch` komutuna `--pdf-path` veya `--url` verip
`--skip-ingestion` seçeneğini kaldırabilirsiniz. Kaynak verilmezse otomatik
arama yapılır. Diğer seçenekler: `python main.py <komut> --help`.

## Docker

```bash
docker build -t sustainability-analyzer .
docker run --rm sustainability-analyzer
```

Varsayılan komut yardım metnini gösterir. Yerel PDF ile analiz örneği
(komutu proje klasöründe çalıştırın):

```powershell
docker run --rm --env-file .env --mount "type=bind,source=$($PWD.Path)/data,target=/app/data" --mount "type=bind,source=$($PWD.Path)/outputs,target=/app/outputs" sustainability-analyzer python main.py batch --company "Example Company" --year 2023 --pdf-path /app/data/report.pdf --report-id example_2023 --output-directory /app/outputs
```

Önce yerel `data` ve `outputs` klasörlerini oluşturun. Container içindeki
`localhost`, ana bilgisayarı göstermez; yerel model sunucusu kullanırken
adresini Docker ağınıza göre ayarlayın.

## Azure örnekleri

`job-template-v6.yaml` ve `azure-minimal-restore-v6.json` dağıtım örnekleridir.
`exampleacr` registry adresini, imaj etiketini, container adını ve rapor
prefix'lerini kendi ortamınıza uyarlayın. `--restore-from-blob` örnekleri,
önceden yüklenmiş rapor durumu gerektirir.

`secretRef` alanları anahtar içermez; Azure ortamında ayrıca oluşturacağınız
`storage-connection`, `groq-api-key` ve `exa-api-key` adlı secret'lara başvurur.
Gerçek değerleri YAML/JSON dosyalarına yazmayın. Azure senkronizasyonu için
`STORAGE_BACKEND=azure_blob`, `AZURE_STORAGE_CONNECTION_STRING` ve
`AZURE_STORAGE_CONTAINER` ayarlarını sağlayın.

## Kontroller ve paylaşım

```bash
python -m unittest discover -s tests -v
python scripts/check_secrets.py
python main.py --help
```

Hassas bilgi tarayıcısı yaygın anahtar biçimlerini ve dosyaya yazılmış kimlik
bilgilerini arar; eşleşen değerleri ekrana basmaz. Bu kontrol tüm olası sırları
tespit etme garantisi vermez. GitHub Actions aynı taramayı ve CLI kontrollerini
çalıştırır; gerçek API anahtarı gerektirmez.

Yerel test çıktıları, veritabanları, `.env`, sertifika dosyaları ve sanal ortam
Git'ten dışlanır. Yükleme öncesi `git status --short` ve `git diff --cached`
ile eklenecek dosyaları kontrol edin. PDF ve üretilmiş raporların paylaşım
izinlerini ayrıca değerlendirin. Model yanıtlarını rapordaki kaynaklarla
doğrulayın; analiz, uzak model servislerine metin/görsel gönderebilir.

## Lisans

Bu proje [MIT lisansı](LICENSE) ile paylaşılır.
