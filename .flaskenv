# Configurazione Flask per evitare problemi di ricaricamento
FLASK_APP=server.py
FLASK_DEBUG=False
FLASK_ENV=production

# Disabilita il watchdog per file system esterni
PYTHONDONTWRITEBYTECODE=1
