Human Review Mobile (React Native)
=================================

Seconda interfaccia mobile in React Native, separata dalla web UI esistente. Replica le principali funzionalità dell’interfaccia corrente per l’uso da cellulare.

Funzionalità incluse (prima versione)
- Selezione tenant e caricamento tenants
- Review queue: lista casi, dettagli caso, risoluzione (decisione umana)
- Statistiche review di base
- Gestione parametri di clustering (lettura/aggiornamento)

Struttura
- `App.tsx`: entry con navigazione e TenantProvider
- `src/navigation/`: stack + tabs
- `src/screens/`: ReviewDashboard, CasesList, CaseDetail, ClusteringParameters
- `src/services/`: api.ts con chiamate REST al backend esistente
- `src/types/`: Tipi condivisi (ReviewCase, Tenant)
- `src/contexts/`: TenantContext mobile
- `src/config.ts`: configurazione `API_BASE_URL`

Requisiti
- Node.js >= 18
- Expo CLI (consigliato): `npm i -g expo` oppure `npx expo`
- App "Expo Go" installata su iPhone (App Store)

Configurazione
1) Imposta l’URL del backend in `src/config.ts` (esempio sotto). Se usi il device reale, imposta l’IP della tua macchina.

```
export const API_BASE_URL = 'http://192.168.1.100:5000/api';
```

2) Installazione pacchetti (sulla tua macchina):

```
cd human-review-mobile
npm install
```

3) Avvio in sviluppo:

```
# Opzione 1: stessa Wi‑Fi (LAN)
npx expo start --lan

# Opzione 2: rete diversa / restrizioni (Tunnel)
npx expo start --tunnel
```

Poi apri l'app "Expo Go" su iPhone e scansiona il QR code.

Note su rete
- Per test su dispositivo fisico usa la modalità LAN e un `API_BASE_URL` raggiungibile (IP locale + porta backend).
- Verifica l’endpoint `/health` del backend: deve essere raggiungibile dal telefono.
- Se la LAN è filtrata, usa `--tunnel` e assicurati che il backend sia comunque raggiungibile via IP.

Compatibilità Expo Go (iOS)
- Il progetto usa solo librerie compatibili con Expo Go (nessun modulo nativo custom).
- È già configurato `react-native-gesture-handler` e `react-native-reanimated`:
  - import iniziale in `index.js`
  - plugin Babel `react-native-reanimated/plugin` in `babel.config.js`
- Dopo `npm install`, se riscontri warning di versione, esegui:
  - `npx expo install react-native-gesture-handler react-native-reanimated react-native-screens react-native-safe-area-context`

Mappatura endpoint
- Review list: `GET /api/review/{tenant_id}/cases`
- Case detail: `GET /api/review/{tenant_id}/cases/{case_id}`
- Resolve case: `POST /api/review/{tenant_id}/cases/{case_id}/resolve`
- Review stats: `GET /api/review/{tenant_id}/stats`
- Clustering params: `GET/POST /api/clustering/{tenant_id}/parameters`

Prossimi passi (TODO)
- Porting completo delle schermate rimanenti (LLM configuration, Examples, Batch, Scheduler)
- Grafici (statistiche) con libreria RN compatibile
- Localizzazione e dark mode
