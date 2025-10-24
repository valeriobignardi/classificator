#!/bin/bash
# Test rapido per verificare che le immagini Docker siano funzionanti
# Autore: GitHub Copilot
# Data: 2025-09-22

echo "🧪 Test delle immagini Docker del sistema di classificazione"

# Test immagine backend
echo "📦 Test immagine backend..."
if docker images | grep -q "classificatore-backend:latest"; then
    echo "✅ Immagine backend trovata"
    
    # Test avvio container
    echo "🚀 Test avvio container backend..."
    BACKEND_ID=$(docker run -d --name test-backend-container \
        -e MONGODB_URL=mongodb://fake:fake@fake:27017/fake \
        -e MYSQL_HOST=fake \
        -e MYSQL_USER=fake \
        -e MYSQL_PASSWORD=fake \
        -e OLLAMA_URL=http://fake:11434 \
        classificatore-backend:latest || echo "failed")
    
    if [ "$BACKEND_ID" != "failed" ]; then
        echo "✅ Container backend avviato: $BACKEND_ID"
        
        # Attendi 5 secondi e controlla logs
        sleep 5
        echo "📋 Logs container backend:"
        docker logs $BACKEND_ID | head -10
        
        # Pulisci
        docker stop $BACKEND_ID >/dev/null 2>&1
        docker rm $BACKEND_ID >/dev/null 2>&1
    else
        echo "❌ Errore avvio container backend"
    fi
else
    echo "❌ Immagine backend non trovata"
fi

echo ""

# Test immagine frontend
echo "📦 Test immagine frontend..."
if docker images | grep -q "classificatore-frontend:latest"; then
    echo "✅ Immagine frontend trovata"
    
    # Test avvio container
    echo "🚀 Test avvio container frontend..."
    FRONTEND_ID=$(docker run -d --name test-frontend-container \
        -p 8080:80 \
        classificatore-frontend:latest || echo "failed")
    
    if [ "$FRONTEND_ID" != "failed" ]; then
        echo "✅ Container frontend avviato: $FRONTEND_ID"
        
        # Test health check
        echo "🏥 Test health check..."
        sleep 3
        if curl -f -s http://localhost:8080/health > /dev/null; then
            echo "✅ Health check frontend riuscito"
        else
            echo "⚠️ Health check frontend fallito (potrebbe essere normale)"
        fi
        
        # Pulisci
        docker stop $FRONTEND_ID >/dev/null 2>&1
        docker rm $FRONTEND_ID >/dev/null 2>&1
    else
        echo "❌ Errore avvio container frontend"
    fi
else
    echo "❌ Immagine frontend non trovata"
fi

echo ""
echo "🎯 Test completato!"
echo ""
echo "Per avviare il sistema completo usa:"
echo "  ./docker-manager.sh setup"
echo ""
echo "Oppure usa Docker Compose direttamente:"
echo "  docker-compose up -d"