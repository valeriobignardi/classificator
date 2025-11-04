#!/bin/bash
# Script per ricostruire completamente il container backend
# Autore: Valerio Bignardi
# Data: 2025-11-04

set -e  # Exit on error

echo "🔄 ======================================"
echo "🔄 REBUILD COMPLETO BACKEND"
echo "🔄 ======================================"

echo ""
echo "1️⃣ Fermo il container backend..."
docker-compose stop backend || echo "⚠️ Container già fermo"

echo ""
echo "2️⃣ Rimuovo il container backend..."
docker-compose rm -f backend || echo "⚠️ Container già rimosso"

echo ""
echo "3️⃣ Rimuovo l'immagine backend..."
docker rmi classificatore-backend 2>/dev/null || echo "⚠️ Immagine già rimossa"

echo ""
echo "4️⃣ Rebuild immagine backend (con cache Docker)..."
docker-compose build backend

echo ""
echo "5️⃣ Avvio nuovo container backend..."
docker-compose up -d backend

echo ""
echo "6️⃣ Attendo avvio container..."
sleep 3

echo ""
echo "7️⃣ Verifico stato container..."
docker ps | grep backend

echo ""
echo "8️⃣ Controllo health..."
sleep 2
docker inspect classificatore-backend --format='{{.State.Health.Status}}' 2>/dev/null || echo "Health check non ancora disponibile"

echo ""
echo "✅ ======================================"
echo "✅ REBUILD COMPLETATO!"
echo "✅ ======================================"
echo ""
echo "📋 Comandi utili:"
echo "   - Logs: docker logs -f classificatore-backend"
echo "   - Logs GPT-5: docker logs classificatore-backend | grep 'GPT-5 DEBUG'"
echo "   - Health: docker inspect classificatore-backend --format='{{.State.Health.Status}}'"
echo ""
