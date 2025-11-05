#!/bin/bash
# Script per ricostruire completamente il container frontend
# Autore: Valerio Bignardi
# Data: 2025-11-05

set -e  # Exit on error

echo "🔄 ======================================"
echo "🔄 REBUILD COMPLETO FRONTEND"
echo "🔄 ======================================"

echo ""
echo "1️⃣ Fermo il container frontend..."
docker-compose stop frontend || echo "⚠️ Container già fermo"

echo ""
echo "2️⃣ Rimuovo il container frontend..."
docker-compose rm -f frontend || echo "⚠️ Container già rimosso"

echo ""
echo "3️⃣ Rimuovo l'immagine frontend..."
docker rmi classificatore-frontend 2>/dev/null || echo "⚠️ Immagine già rimossa"

echo ""
echo "4️⃣ Rebuild immagine frontend (con cache Docker)..."
docker-compose build frontend

echo ""
echo "5️⃣ Avvio nuovo container frontend..."
docker-compose up -d frontend

echo ""
echo "6️⃣ Attendo avvio container..."
sleep 3

echo ""
echo "7️⃣ Verifico stato container..."
docker ps | grep frontend

echo ""
echo "8️⃣ Controllo health..."
sleep 2
docker inspect classificatore-frontend --format='{{.State.Health.Status}}' 2>/dev/null || echo "Health check non ancora disponibile"

echo ""
echo "✅ ======================================"
echo "✅ REBUILD COMPLETATO!"
echo "✅ ======================================"
echo ""
echo "📋 Comandi utili:"
echo "   - Logs: docker logs -f classificatore-frontend"
echo "   - Logs React: docker logs classificatore-frontend | grep 'webpack'"
echo "   - Health: docker inspect classificatore-frontend --format='{{.State.Health.Status}}'"
echo "   - URL: http://localhost:3000"
echo ""