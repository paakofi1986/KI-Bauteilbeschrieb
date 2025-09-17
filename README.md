# KI-Bauteilbeschrieb (EbicBauteilApp)

Diese Streamlit-App unterstützt die **Zustandsanalyse von Bauprojekten** mit Bildern, Metadaten und KI-gestützten Bauteilbeschreibungen.

## 🚀 Features
- Projekte anlegen, speichern und verwalten (SQLite-Datenbank `projects.db`).
- Bilder importieren (JPG, PNG oder ZIP).
- Automatische Miniaturbilder (Thumbnails) mit EXIF-Datumserkennung.
- Tabellen-Editor für Bauteil-Beschreibungen (Bearbeiten von Bauteile, Risiken, Empfehlungen etc.).
- **KI-Analyse** über OpenAI (ChatGPT) oder Microsoft Copilot:
  - Füllt automatisch Felder wie *Bauteile, Besonderheiten, Risiken, Empfehlung*.
  - Custom-Prompt-Eingabe in der Sidebar für eigene Abfragen.
- Exporte:
  - **PDF** (ReportLab-basiert)
  - **Word (DOCX)** (python-docx)
  - **Excel (XLSX)** (xlsxwriter)
  - **ZIP**-Paket (alle Formate + Bilder)
- Galerieansicht mit Detailansicht pro Bild.
- Inline-PDF-Vorschau (Edge-sicher via Blob-URL).
- Konfigurierbares **Theme** über `.streamlit/config.toml`.

## 📦 Installation

1. Repository klonen oder Dateien herunterladen.
2. Virtuelle Umgebung erstellen und aktivieren:

   ```bash
   python -m venv .venv
   source .venv/bin/activate   # Linux/macOS
   .venv\Scripts\activate    # Windows
   ```

3. Abhängigkeiten installieren:

   ```bash
   pip install -r requirements.txt
   ```

## ▶️ Starten der App

```bash
streamlit run EbicBauteilApp.py
```

Die App ist danach unter [http://localhost:8501](http://localhost:8501) erreichbar.

## ⚙️ Konfiguration

### Theme
Passe das Aussehen an, indem du `.streamlit/config.toml` erstellst:

```toml
[theme]
base = "light"
primaryColor = "#31333f"
backgroundColor = "#ffffff"
secondaryBackgroundColor = "#f0f2f6"
textColor = "#000000"
font = "sans serif"
```

### API-Keys
API-Keys für OpenAI oder Copilot werden in `config.json` gespeichert oder können über `st.secrets` gesetzt werden.

Beispiel `config.json`:
```json
{
  "ki_provider": "OpenAI (ChatGPT)",
  "ki_model": "gpt-4o",
  "api_keys": {
    "OpenAI (ChatGPT)": {
      "default": "sk-..."
    }
  }
}
```

## ☁️ Deployment

### Streamlit Community Cloud
1. Push das Projekt in ein GitHub-Repo.
2. Gehe zu [share.streamlit.io](https://share.streamlit.io).
3. Wähle dein Repo und die App-Datei `EbicBauteilApp.py` aus.
4. Füge `requirements.txt` hinzu, um Abhängigkeiten automatisch zu installieren.

### Docker
Dockerfile-Beispiel:
```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
EXPOSE 8501
CMD ["streamlit", "run", "EbicBauteilApp.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

Build & Run:
```bash
docker build -t bauteilapp .
docker run -p 8501:8501 bauteilapp
```

## 🛠️ Tech-Stack
- Python 3.11
- Streamlit
- Pandas
- Pillow
- ReportLab
- python-docx
- OpenAI Python SDK
- SQLite

## ❤️ Autor
Made with ♥️ by Elvis Owusu