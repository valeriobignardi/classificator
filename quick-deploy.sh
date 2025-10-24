#!/bin/bash
# Script di deploy rapido per il sistema di classificazione
# Autore: GitHub Copilot
# Data: 2025-09-22
# Scopo: Deploy rapido con build automatico

set -e

# Colori
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "${BLUE}🚀 Deploy Sistema di Classificazione${NC}"
echo ""

# Verifica che Docker sia disponibile
if ! command -v docker &> /dev/null; then
    echo -e "${YELLOW}❌ Docker non trovato!${NC}"
    exit 1
fi

# Ferma container esistenti se in esecuzione
echo -e "${BLUE}📦 Fermando container esistenti...${NC}"
docker-compose down --remove-orphans

# Build delle immagini
echo -e "${BLUE}🔨 Build delle immagini...${NC}"
docker-compose build --no-cache

# Avvio dei servizi
echo -e "${BLUE}🚀 Avvio servizi...${NC}"
docker-compose up -d

# Attesa che i servizi siano pronti
echo -e "${BLUE}⏳ Attendo che i servizi siano pronti...${NC}"
sleep 15

# Verifica stato
echo -e "${BLUE}🔍 Verifica stato servizi...${NC}"

# Test backend
if curl -f -s http://localhost:5000/health > /dev/null; then
    echo -e "${GREEN}✅ Backend: OK (http://localhost:5000)${NC}"
else
    echo -e "${YELLOW}⚠️  Backend: Non ancora pronto${NC}"
fi

# Test frontend
if curl -f -s http://localhost:3005/ > /dev/null; then
    echo -e "${GREEN}✅ Frontend: OK (http://localhost:3005)${NC}"
else
    echo -e "${YELLOW}⚠️  Frontend: Non ancora pronto${NC}"
fi

echo ""
echo -e "${GREEN}🎉 Deploy completato!${NC}"
echo ""
echo "📍 Servizi disponibili:"
echo "   • Frontend: http://localhost:3005"
echo "   • Backend API: http://localhost:5000"
echo ""
echo "📍 Servizi esterni utilizzati:"
echo "   • MongoDB: localhost:27017"
echo "   • MySQL: localhost:3306"
echo "   • Ollama: localhost:11434"
echo ""
echo "🔧 Gestione container:"
echo "   • Stato: docker-compose ps"
echo "   • Logs: docker-compose logs -f"
echo "   • Stop: docker-compose down"
echo ""