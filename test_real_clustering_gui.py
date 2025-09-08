#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
🧪 TEST ENDPOINT CLUSTERING REALE - Con tenant esistente

Scopo: Testare l'endpoint /clustering/test con tenant reale per verificare compatibilità DocumentoProcessing
Autore: Valerio Bignardi
Data: 2025-09-08
"""

import requests
import json
import time
from typing import Dict, Any

def test_real_clustering_endpoint():
    """
    Test con tenant che dovrebbe esistere nel sistema
    """
    print(f"\n🧪 TEST ENDPOINT CLUSTERING REALE")
    print(f"=" * 60)
    
    # Tenant comuni che potrebbero esistere
    test_tenants = ["humanitas", "test", "demo", "sviluppo"]
    
    for tenant_id in test_tenants:
        print(f"\n🎯 Test tenant: {tenant_id}")
        
        success = test_tenant_clustering(tenant_id)
        
        if success:
            print(f"   ✅ Test completato con successo per {tenant_id}")
            break
        else:
            print(f"   ⚠️ Tenant {tenant_id} non disponibile")
    
    print(f"\n📊 Test completato")

def test_tenant_clustering(tenant_id: str) -> bool:
    """
    Testa clustering per un tenant specifico
    """
    try:
        # Parametri test più conservativi per dati reali
        test_payload = {
            "parameters": {
                "min_cluster_size": 5,
                "min_samples": 3,
                "cluster_selection_epsilon": 0.1,
                "metric": "cosine",
                "use_umap": False,  # Disabilito UMAP per test più veloce
                "sample_size": 30   # Campione piccolo per test rapido
            }
        }
        
        print(f"   📡 Chiamata API...")
        url = f"http://localhost:5000/api/clustering/{tenant_id}/test"
        
        start_time = time.time()
        response = requests.post(url, json=test_payload, timeout=60)
        execution_time = time.time() - start_time
        
        print(f"   ⏱️ Tempo risposta: {execution_time:.2f}s")
        print(f"   📡 Status code: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            
            # Verifica struttura risposta
            if validate_response_structure(data):
                print_response_summary(data)
                
                # Test specifico per visualizzazioni
                test_visualization_compatibility(data)
                
                return True
            else:
                print(f"   ❌ Struttura risposta non valida")
                return False
                
        elif response.status_code == 404:
            print(f"   ⚠️ Tenant non trovato")
            return False
            
        elif response.status_code == 400:
            error_data = response.json()
            print(f"   ⚠️ Errore parametri: {error_data.get('error', 'N/A')}")
            return False
            
        else:
            print(f"   ❌ Errore HTTP: {response.status_code}")
            return False
            
    except requests.exceptions.Timeout:
        print(f"   ❌ Timeout (>60s)")
        return False
        
    except requests.exceptions.ConnectionError:
        print(f"   ❌ Connessione fallita")
        return False
        
    except Exception as e:
        print(f"   ❌ Errore: {e}")
        return False

def validate_response_structure(data: Dict[str, Any]) -> bool:
    """
    Valida che la risposta abbia la struttura corretta per il frontend
    """
    required_fields = [
        'success',
        'statistics', 
        'detailed_clusters',
        'quality_metrics',
        'outlier_analysis'
    ]
    
    missing = [field for field in required_fields if field not in data]
    
    if missing:
        print(f"   ❌ Campi mancanti: {missing}")
        return False
    
    # Verifica campi statistiche
    if 'statistics' in data:
        stats = data['statistics']
        stats_fields = ['total_conversations', 'n_clusters', 'n_outliers']
        missing_stats = [field for field in stats_fields if field not in stats]
        
        if missing_stats:
            print(f"   ❌ Statistiche mancanti: {missing_stats}")
            return False
    
    return True

def print_response_summary(data: Dict[str, Any]) -> None:
    """
    Stampa riassunto della risposta
    """
    if data.get('success'):
        stats = data.get('statistics', {})
        
        print(f"   📊 Conversazioni: {stats.get('total_conversations', 0)}")
        print(f"   🔗 Cluster: {stats.get('n_clusters', 0)}")
        print(f"   🔍 Outlier: {stats.get('n_outliers', 0)}")
        
        if 'execution_time' in data:
            print(f"   ⏱️ Tempo esecuzione: {data['execution_time']:.2f}s")
        
        # Verifica quality metrics
        if 'quality_metrics' in data:
            qm = data['quality_metrics']
            if 'silhouette_score' in qm:
                print(f"   📈 Silhouette score: {qm['silhouette_score']:.3f}")
            if 'quality_assessment' in qm:
                print(f"   🎯 Qualità: {qm['quality_assessment']}")
    else:
        error = data.get('error', 'Errore sconosciuto')
        print(f"   ❌ Errore: {error}")

def test_visualization_compatibility(data: Dict[str, Any]) -> None:
    """
    Testa specificamente la compatibilità dei dati di visualizzazione
    """
    print(f"   🎨 Test dati visualizzazione:")
    
    viz_data = data.get('visualization_data')
    
    if not viz_data:
        print(f"      ⚠️ Dati visualizzazione assenti")
        return
    
    # Test punti per grafici
    points = viz_data.get('points', [])
    print(f"      📊 Punti disponibili: {len(points)}")
    
    if points and len(points) > 0:
        # Verifica struttura primo punto
        first_point = points[0]
        required_point_fields = ['x', 'y', 'cluster_id', 'session_id']
        
        missing_point_fields = [f for f in required_point_fields if f not in first_point]
        
        if not missing_point_fields:
            print(f"      ✅ Struttura punti valida")
            
            # Test se ha coordinate 3D
            has_z = 'z' in first_point
            print(f"      📐 Coordinate 3D: {'✅' if has_z else '❌'}")
        else:
            print(f"      ❌ Campi punto mancanti: {missing_point_fields}")
    
    # Test colori
    colors = viz_data.get('cluster_colors', {})
    print(f"      🎨 Colori cluster: {len(colors)} definiti")
    
    # Test coordinate alternative
    coords = viz_data.get('coordinates', {})
    coord_types = ['tsne_2d', 'pca_2d', 'pca_3d']
    available_coords = [ct for ct in coord_types if ct in coords and len(coords[ct]) > 0]
    
    print(f"      📐 Coordinate multiple: {available_coords}")
    
    # Compatibilità frontend
    frontend_compatible = (
        len(points) > 0 and 
        'x' in points[0] and 'y' in points[0] and
        len(colors) > 0
    )
    
    print(f"      🖥️ Compatibilità frontend: {'✅' if frontend_compatible else '❌'}")

if __name__ == "__main__":
    test_real_clustering_endpoint()
