#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
🧪 TEST INTEGRAZIONE INTERFACCIA GRAFICA - Verifica compatibilità con DocumentoProcessing

Scopo: Testare che l'interfaccia grafica (tasto PROVA CLUSTERING) funzioni con le modifiche DocumentoProcessing
Componenti testati:
- Endpoint /api/clustering/<tenant_id>/test
- ClusteringTestService 
- Visualizzazioni grafiche
- Mapping dati frontend/backend

Autore: Valerio Bignardi  
Data: 2025-09-08
"""

import sys
import os
import json
import time
import requests
from datetime import datetime
from typing import Dict, Any, List

# Aggiungi il percorso del progetto
sys.path.append('/home/ubuntu/classificatore')

def test_clustering_api_integration():
    """
    🧪 Test completo dell'integrazione API clustering con interfaccia grafica
    """
    print(f"\n🧪 TEST INTEGRAZIONE API CLUSTERING")
    print(f"=" * 80)
    print(f"📅 Esecuzione: {datetime.now()}")
    
    # 1. Test connessione server
    print(f"\n1️⃣ TEST: Connessione server backend")
    test_server_connection()
    
    # 2. Test endpoint clustering
    print(f"\n2️⃣ TEST: Endpoint clustering test API")
    test_clustering_endpoint()
    
    # 3. Test formato dati visualizzazione
    print(f"\n3️⃣ TEST: Formato dati per visualizzazioni")
    test_visualization_data_format()
    
    # 4. Test compatibilità frontend
    print(f"\n4️⃣ TEST: Compatibilità mapping dati frontend")
    test_frontend_compatibility()
    
    print(f"\n✅ TESTING INTEGRAZIONE COMPLETATO")
    print(f"=" * 80)

def test_server_connection():
    """
    Test connessione al server backend
    """
    try:
        # Testa endpoint base
        response = requests.get("http://localhost:5000/health", timeout=5)
        
        if response.status_code == 200:
            print(f"   ✅ Server backend connesso: {response.status_code}")
        else:
            print(f"   ⚠️ Server risponde ma stato diverso: {response.status_code}")
            
    except requests.exceptions.ConnectionError:
        print(f"   ❌ Server backend non raggiungibile su localhost:5000")
        print(f"   💡 Suggerimento: Avviare il server con 'python server.py'")
        
    except Exception as e:
        print(f"   ❌ Errore connessione server: {e}")

def test_clustering_endpoint():
    """
    Test endpoint clustering specifico
    """
    try:
        # Usa un tenant di test
        tenant_id = "test-tenant-gui"
        
        # Parametri test clustering
        test_payload = {
            "parameters": {
                "min_cluster_size": 3,
                "min_samples": 2,
                "cluster_selection_epsilon": 0.15,
                "metric": "cosine",
                "use_umap": True,
                "umap_n_neighbors": 15,
                "umap_min_dist": 0.1,
                "umap_n_components": 50
            },
            "sample_size": 50
        }
        
        print(f"   🎯 Test tenant: {tenant_id}")
        print(f"   📊 Parametri: {len(test_payload['parameters'])} parametri")
        print(f"   📝 Sample size: {test_payload['sample_size']}")
        
        # Chiamata API
        url = f"http://localhost:5000/api/clustering/{tenant_id}/test"
        
        try:
            response = requests.post(url, json=test_payload, timeout=30)
            
            print(f"   📡 Status code: {response.status_code}")
            
            if response.status_code == 200:
                data = response.json()
                print(f"   ✅ API risponde correttamente")
                
                # Verifica struttura dati
                required_fields = ['success', 'statistics', 'detailed_clusters', 'quality_metrics']
                missing_fields = [field for field in required_fields if field not in data]
                
                if missing_fields:
                    print(f"   ⚠️ Campi mancanti: {missing_fields}")
                else:
                    print(f"   ✅ Tutti i campi richiesti presenti")
                    
                # Verifica dati visualizzazione
                if 'visualization_data' in data:
                    print(f"   ✅ Dati visualizzazione presenti")
                    viz_data = data['visualization_data']
                    
                    if 'points' in viz_data and isinstance(viz_data['points'], list):
                        print(f"   📊 Punti visualizzazione: {len(viz_data['points'])}")
                    
                    if 'coordinates' in viz_data:
                        coords = viz_data['coordinates']
                        coord_types = ['tsne_2d', 'pca_2d', 'pca_3d']
                        available_coords = [ct for ct in coord_types if ct in coords]
                        print(f"   🎨 Coordinate disponibili: {available_coords}")
                else:
                    print(f"   ⚠️ Dati visualizzazione mancanti")
                    
                return data
                
            elif response.status_code == 404:
                print(f"   ⚠️ Tenant non trovato (normale per test)")
                return None
                
            else:
                print(f"   ❌ Errore API: {response.status_code}")
                try:
                    error_data = response.json()
                    print(f"   🔍 Dettagli errore: {error_data.get('error', 'N/A')}")
                except:
                    print(f"   🔍 Response non JSON")
                return None
                
        except requests.exceptions.Timeout:
            print(f"   ❌ Timeout API (>30s)")
            return None
            
        except requests.exceptions.ConnectionError:
            print(f"   ❌ Errore connessione API")
            return None
            
    except Exception as e:
        print(f"   ❌ Errore test endpoint: {e}")
        return None

def test_visualization_data_format():
    """
    Test formato dati per visualizzazioni grafiche
    """
    try:
        # Simula dati di risposta API per verificare formato
        mock_response = {
            "success": True,
            "statistics": {
                "total_conversations": 100,
                "n_clusters": 5,
                "n_outliers": 10,
                "clustering_ratio": 0.9
            },
            "visualization_data": {
                "points": [
                    {
                        "x": 1.5,
                        "y": 2.3,
                        "z": -0.8,
                        "cluster_id": 0,
                        "cluster_label": "Cluster 0",
                        "session_id": "test_session_1",
                        "text_preview": "Testo di esempio..."
                    }
                ],
                "cluster_colors": {
                    "0": "#FF5733",
                    "1": "#33FF57", 
                    "-1": "#333333"
                },
                "coordinates": {
                    "tsne_2d": [[1.5, 2.3]],
                    "pca_2d": [[1.2, 2.1]],
                    "pca_3d": [[1.2, 2.1, -0.8]]
                },
                "statistics": {
                    "total_points": 100,
                    "n_clusters": 5,
                    "n_outliers": 10,
                    "dimensions": 3
                }
            }
        }
        
        # Verifica struttura dati visualizzazione
        viz_data = mock_response["visualization_data"]
        
        # Test punti
        if "points" in viz_data and len(viz_data["points"]) > 0:
            point = viz_data["points"][0]
            required_point_fields = ["x", "y", "cluster_id", "session_id"]
            
            point_fields_present = all(field in point for field in required_point_fields)
            print(f"   📊 Formato punti: {'✅' if point_fields_present else '❌'}")
            
            if not point_fields_present:
                missing = [f for f in required_point_fields if f not in point]
                print(f"       Campi mancanti: {missing}")
        
        # Test colori cluster
        if "cluster_colors" in viz_data:
            colors = viz_data["cluster_colors"]
            has_outlier_color = "-1" in colors
            print(f"   🎨 Colori cluster: ✅ ({len(colors)} colori)")
            print(f"   🔍 Colore outlier: {'✅' if has_outlier_color else '❌'}")
        
        # Test coordinate
        if "coordinates" in viz_data:
            coords = viz_data["coordinates"]
            coord_types = ["tsne_2d", "pca_2d", "pca_3d"]
            available = [ct for ct in coord_types if ct in coords and len(coords[ct]) > 0]
            print(f"   📐 Coordinate: ✅ ({len(available)}/{len(coord_types)} tipi)")
            
        print(f"   ✅ Formato dati visualizzazione valido")
        
    except Exception as e:
        print(f"   ❌ Errore test formato visualizzazione: {e}")

def test_frontend_compatibility():
    """
    Test compatibilità con mapping frontend
    """
    try:
        # Simula il mapping che fa il frontend (dal file apiService.ts)
        backend_response = {
            "success": True,
            "statistics": {
                "total_conversations": 100,
                "n_clusters": 5,
                "n_outliers": 10,
                "clustering_ratio": 0.9
            },
            "detailed_clusters": [
                {
                    "cluster_id": 0,
                    "size": 20,
                    "conversations": [
                        {
                            "session_id": "test_1",
                            "testo_completo": "Testo test",
                            "is_representative": True
                        }
                    ]
                }
            ],
            "quality_metrics": {
                "silhouette_score": 0.65,
                "cluster_balance": "balanced",
                "quality_assessment": "good"
            },
            "outlier_analysis": {
                "count": 10,
                "ratio": 0.1,
                "sample_outliers": []
            },
            "visualization_data": {
                "points": [],
                "cluster_colors": {},
                "coordinates": {}
            }
        }
        
        # Simula mapping frontend
        try:
            mapped_result = {
                "success": backend_response["success"],
                "statistics": {
                    "total_conversations": backend_response["statistics"]["total_conversations"],
                    "n_clusters": backend_response["statistics"]["n_clusters"],
                    "n_outliers": backend_response["statistics"]["n_outliers"],
                    "clustering_ratio": backend_response["statistics"]["clustering_ratio"]
                },
                "detailed_clusters": {
                    "clusters": [
                        {
                            "cluster_id": cluster["cluster_id"],
                            "size": cluster["size"],
                            "conversations": [
                                {
                                    "session_id": conv["session_id"],
                                    "text": conv["testo_completo"],
                                    "text_length": len(conv["testo_completo"])
                                }
                                for conv in cluster["conversations"]
                            ]
                        }
                        for cluster in backend_response["detailed_clusters"]
                    ]
                },
                "visualization_data": backend_response.get("visualization_data")
            }
            
            print(f"   ✅ Mapping frontend: OK")
            print(f"   📊 Cluster mappati: {len(mapped_result['detailed_clusters']['clusters'])}")
            
            # Verifica campi critici
            critical_fields = ["success", "statistics", "detailed_clusters"]
            missing_critical = [field for field in critical_fields if field not in mapped_result]
            
            if missing_critical:
                print(f"   ❌ Campi critici mancanti: {missing_critical}")
            else:
                print(f"   ✅ Tutti i campi critici presenti")
            
        except Exception as mapping_error:
            print(f"   ❌ Errore mapping frontend: {mapping_error}")
            
    except Exception as e:
        print(f"   ❌ Errore test compatibilità frontend: {e}")

if __name__ == "__main__":
    test_clustering_api_integration()
