#!/bin/bash
# Quick setup per diversi tipi di progetto

set -e

GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${BLUE}🚀 Quick Setup Universal Deploy${NC}"
echo ""

# Auto-detect project type
PROJECT_TYPE=""
if [ -f "package.json" ]; then
    PROJECT_TYPE="nodejs"
elif [ -f "requirements.txt" ] || [ -f "pyproject.toml" ]; then
    PROJECT_TYPE="python"
elif [ -f "docker-compose.yml" ]; then
    PROJECT_TYPE="docker"
else
    PROJECT_TYPE="generic"
fi

echo "� Progetto rilevato: $PROJECT_TYPE"
echo ""

PROJECT_NAME=$(basename "$PWD")

# Crea i file di deploy necessari
if [ "$PROJECT_TYPE" = "nodejs" ] && [ ! -f "Dockerfile" ]; then
    echo "� Creando Dockerfile per Node.js..."
    cp universal-deploy-standalone/Dockerfile.nodejs Dockerfile
    sed -i.bak "s/myapp/$PROJECT_NAME/g" Dockerfile && rm Dockerfile.bak
fi

if [ "$PROJECT_TYPE" = "python" ] && [ ! -f "Dockerfile" ]; then
    echo "📄 Creando Dockerfile per Python..."
    cp universal-deploy-standalone/Dockerfile.python Dockerfile
    sed -i.bak "s/myapp/$PROJECT_NAME/g" Dockerfile && rm Dockerfile.bak
fi

if [ ! -f "docker-compose.yml" ]; then
    echo "🐳 Creando docker-compose.yml..."
    if [ "$PROJECT_TYPE" = "nodejs" ]; then
        cp universal-deploy-standalone/docker-compose.nodejs.yml docker-compose.yml
    elif [ "$PROJECT_TYPE" = "python" ]; then
        cp universal-deploy-standalone/docker-compose.python.yml docker-compose.yml
    else
        cp universal-deploy-standalone/docker-compose.template.yml docker-compose.yml
    fi
    
    # Sostituisci il nome del progetto
    sed -i.bak "s/myapp/$PROJECT_NAME/g" docker-compose.yml && rm docker-compose.yml.bak
fi

if [ ! -f "deploy-config.env" ]; then
    echo "⚙️  Creando deploy-config.env..."
    cp universal-deploy-standalone/ec2-config.env.template deploy-config.env
    sed -i.bak "s/webScrapingMultiTenant/$PROJECT_NAME/g" deploy-config.env && rm deploy-config.env.bak
fi

echo ""
echo -e "${GREEN}✅ Quick setup completato!${NC}"
echo ""
echo "📋 Files creati:"
[ -f "Dockerfile" ] && echo "   ✅ Dockerfile"
[ -f "docker-compose.yml" ] && echo "   ✅ docker-compose.yml"
[ -f "deploy-config.env" ] && echo "   ✅ deploy-config.env"
echo ""
echo "🚀 Prossimi passi:"
echo "   1. Modifica deploy-config.env con i tuoi dati server"
echo "   2. Esegui: ./universal-deploy-standalone/universal-deploy.sh config"
echo "   3. Esegui: ./universal-deploy-standalone/universal-deploy.sh deploy"
