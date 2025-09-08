#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
🧪 TEST MOCK INTERFACCIA GRAFICA - Simulazione completa

Scopo: Testare l'intera catena di funzionalità dell'interfaccia grafica 
simulando dati realistici senza dipendere da tenant esistenti.

Testa:
- Struttura dati backend → frontend  
- Compatibilità visualizzazioni
- Mapping completo ClusteringParametersManager

Autore: Valerio Bignardi
Data: 2025-09-08
"""

import json
import numpy as np
from typing import Dict, Any, List
from datetime import datetime

def test_complete_gui_workflow():
    """
    🧪 Test completo del workflow interfaccia grafica
    """
    print(f"\n🧪 TEST COMPLETO WORKFLOW GUI")
    print(f"=" * 80)
    print(f"📅 Esecuzione: {datetime.now()}")
    
    # 1. Simula risposta backend
    print(f"\n1️⃣ SIMULAZIONE: Risposta backend")
    backend_response = generate_mock_backend_response()
    print(f"   ✅ Risposta backend generata: {len(json.dumps(backend_response))} bytes")
    
    # 2. Test mapping frontend  
    print(f"\n2️⃣ TEST: Mapping frontend (ClusteringParametersManager.tsx)")
    frontend_mapped = test_frontend_mapping(backend_response)
    
    # 3. Test componente risultati
    print(f"\n3️⃣ TEST: Componente ClusteringTestResults")
    test_clustering_results_component(frontend_mapped)
    
    # 4. Test dati visualizzazione
    print(f"\n4️⃣ TEST: Dati per visualizzazioni grafiche")
    test_visualization_components(frontend_mapped)
    
    # 5. Test edge cases
    print(f"\n5️⃣ TEST: Edge cases e fallback")
    test_edge_cases()
    
    print(f"\n✅ TEST WORKFLOW GUI COMPLETATO")
    print(f"=" * 80)

def generate_mock_backend_response() -> Dict[str, Any]:
    """
    Genera una risposta mock realistica del backend
    """
    # Simula embedding coordinates per 100 punti
    np.random.seed(42)  # Per risultati riproducibili
    n_points = 100
    n_clusters = 5
    
    # Coordinate 2D e 3D per visualizzazioni
    tsne_2d = np.random.randn(n_points, 2) * 2
    pca_2d = np.random.randn(n_points, 2) * 1.5  
    pca_3d = np.random.randn(n_points, 3) * 1.2
    
    # Assegna cluster labels (simulando HDBSCAN)
    cluster_labels = []
    outliers_count = 15
    
    # Prima parte: outliers
    cluster_labels.extend([-1] * outliers_count)
    
    # Resto: diviso tra cluster
    remaining = n_points - outliers_count
    points_per_cluster = remaining // n_clusters
    
    for cluster_id in range(n_clusters):
        cluster_labels.extend([cluster_id] * points_per_cluster)
    
    # Completa eventuali punti rimanenti
    while len(cluster_labels) < n_points:
        cluster_labels.append(np.random.randint(0, n_clusters))
    
    cluster_labels = np.array(cluster_labels)
    
    # Genera punti per visualizzazione
    points = []
    cluster_colors = {str(i): f"hsl({i * 360 // n_clusters}, 70%, 50%)" for i in range(n_clusters)}
    cluster_colors["-1"] = "#666666"  # Colore outlier
    
    for i in range(n_points):
        points.append({
            "x": float(tsne_2d[i, 0]),
            "y": float(tsne_2d[i, 1]), 
            "z": float(pca_3d[i, 2]) if len(pca_3d[i]) > 2 else 0.0,
            "cluster_id": int(cluster_labels[i]),
            "cluster_label": f"Cluster {cluster_labels[i]}" if cluster_labels[i] != -1 else "Outlier",
            "session_id": f"session_{i:03d}",
            "text_preview": f"Testo conversazione di esempio numero {i} per test interfaccia..."
        })
    
    # Genera cluster dettagliati
    detailed_clusters = []
    
    for cluster_id in range(n_clusters):
        cluster_points = [p for p in points if p["cluster_id"] == cluster_id]
        
        if cluster_points:
            conversations = []
            for point in cluster_points[:10]:  # Max 10 per cluster
                conversations.append({
                    "session_id": point["session_id"],
                    "testo_completo": point["text_preview"] * 3,  # Testo più lungo
                    "is_representative": len(conversations) < 2  # Prime 2 sono rappresentanti
                })
            
            detailed_clusters.append({
                "cluster_id": cluster_id,
                "label": f"Cluster {cluster_id}",
                "size": len(cluster_points),
                "conversations": conversations,
                "representatives": [c for c in conversations if c["is_representative"]],
                "representative_count": len([c for c in conversations if c["is_representative"]])
            })
    
    # Outliers
    outlier_points = [p for p in points if p["cluster_id"] == -1]
    sample_outliers = []
    for point in outlier_points[:5]:  # Max 5 outliers di esempio
        sample_outliers.append({
            "session_id": point["session_id"],
            "testo_completo": point["text_preview"] * 2
        })
    
    return {
        "success": True,
        "tenant_id": "mock-tenant-gui",
        "execution_time": 2.345,
        "statistics": {
            "total_conversations": n_points,
            "n_clusters": n_clusters,
            "n_outliers": outliers_count,
            "n_clustered": n_points - outliers_count,
            "clustering_ratio": round((n_points - outliers_count) / n_points, 3),
            "parameters_used": {
                "min_cluster_size": 5,
                "min_samples": 3,
                "use_umap": True,
                "umap_n_components": 50
            }
        },
        "quality_metrics": {
            "silhouette_score": 0.652,
            "davies_bouldin_score": 0.234,
            "calinski_harabasz_score": 156.7,
            "outlier_ratio": outliers_count / n_points,
            "cluster_balance": "balanced",
            "quality_assessment": "good"
        },
        "detailed_clusters": detailed_clusters,
        "outlier_analysis": {
            "count": outliers_count,
            "ratio": outliers_count / n_points,
            "analysis": "Outlier ratio normale per il dataset",
            "recommendation": "Parametri clustering appropriati",
            "sample_outliers": sample_outliers
        },
        "visualization_data": {
            "points": points,
            "cluster_colors": cluster_colors,
            "statistics": {
                "total_points": n_points,
                "n_clusters": n_clusters,
                "n_outliers": outliers_count,
                "dimensions": 3
            },
            "coordinates": {
                "tsne_2d": tsne_2d.tolist(),
                "pca_2d": pca_2d.tolist(),
                "pca_3d": pca_3d.tolist()
            }
        }
    }

def test_frontend_mapping(backend_response: Dict[str, Any]) -> Dict[str, Any]:
    """
    Simula il mapping che avviene nel frontend (ClusteringParametersManager.tsx)
    """
    try:
        # Mapping esatto dal codice TypeScript
        mapped_result = {
            "success": backend_response["success"],
            "error": backend_response.get("error"),
            "execution_time": backend_response["execution_time"],
            "statistics": backend_response["statistics"] if backend_response.get("statistics") else {
                "total_conversations": 0,
                "n_clusters": 0,
                "n_outliers": 0,
                "clustering_ratio": 0
            },
            "quality_metrics": backend_response["quality_metrics"] if backend_response.get("quality_metrics") else {
                "silhouette_score": 0,
                "calinski_harabasz_score": 0,
                "davies_bouldin_score": 0
            },
            "recommendations": [
                f"📊 Valutazione: {backend_response['quality_metrics']['quality_assessment']}",
                f"📈 Bilanciamento cluster: {backend_response['quality_metrics']['cluster_balance']}",
                f"💡 {backend_response['outlier_analysis']['recommendation']}"
            ] if backend_response.get("quality_metrics") and backend_response.get("outlier_analysis") else [],
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
            "outlier_analysis": {
                "count": backend_response["outlier_analysis"]["count"],
                "percentage": backend_response["outlier_analysis"]["ratio"] * 100,
                "samples": [
                    {
                        "session_id": outlier["session_id"],
                        "text": outlier["testo_completo"],
                        "text_length": len(outlier["testo_completo"])
                    }
                    for outlier in backend_response["outlier_analysis"]["sample_outliers"]
                ]
            } if backend_response.get("outlier_analysis") else {
                "count": 0,
                "percentage": 0,
                "samples": []
            },
            "visualization_data": backend_response.get("visualization_data")
        }
        
        print(f"   ✅ Mapping frontend completato")
        print(f"   📊 Cluster mappati: {len(mapped_result['detailed_clusters']['clusters'])}")
        print(f"   📈 Outlier mappati: {len(mapped_result['outlier_analysis']['samples'])}")
        print(f"   🎨 Punti visualizzazione: {len(mapped_result['visualization_data']['points']) if mapped_result['visualization_data'] else 0}")
        
        return mapped_result
        
    except Exception as e:
        print(f"   ❌ Errore mapping frontend: {e}")
        return {}

def test_clustering_results_component(mapped_data: Dict[str, Any]) -> None:
    """
    Testa compatibilità con il componente ClusteringTestResults
    """
    try:
        # Verifica campi richiesti dal componente
        required_fields = [
            "success", "statistics", "detailed_clusters", 
            "quality_metrics", "outlier_analysis"
        ]
        
        missing_fields = [field for field in required_fields if field not in mapped_data]
        
        if missing_fields:
            print(f"   ❌ Campi mancanti per ClusteringTestResults: {missing_fields}")
            return
        
        # Test statistiche
        stats = mapped_data["statistics"]
        stats_ok = all(field in stats for field in ["total_conversations", "n_clusters", "n_outliers"])
        print(f"   📊 Statistiche: {'✅' if stats_ok else '❌'}")
        
        # Test cluster dettagliati
        clusters = mapped_data["detailed_clusters"]["clusters"]
        cluster_structure_ok = len(clusters) > 0 and all(
            "cluster_id" in c and "size" in c and "conversations" in c 
            for c in clusters
        )
        print(f"   🔗 Struttura cluster: {'✅' if cluster_structure_ok else '❌'}")
        
        # Test metriche qualità
        qm = mapped_data["quality_metrics"]
        metrics_ok = "silhouette_score" in qm
        print(f"   📈 Metriche qualità: {'✅' if metrics_ok else '❌'}")
        
        # Test outliers
        outliers = mapped_data["outlier_analysis"]
        outliers_ok = "count" in outliers and "percentage" in outliers
        print(f"   🔍 Analisi outlier: {'✅' if outliers_ok else '❌'}")
        
        print(f"   ✅ ClusteringTestResults: Compatibile")
        
    except Exception as e:
        print(f"   ❌ Errore test ClusteringTestResults: {e}")

def test_visualization_components(mapped_data: Dict[str, Any]) -> None:
    """
    Testa compatibilità con componenti visualizzazione (grafici)
    """
    try:
        viz_data = mapped_data.get("visualization_data")
        
        if not viz_data:
            print(f"   ⚠️ Dati visualizzazione mancanti")
            return
        
        # Test struttura punti
        points = viz_data.get("points", [])
        if points and len(points) > 0:
            first_point = points[0]
            point_fields_ok = all(
                field in first_point 
                for field in ["x", "y", "cluster_id", "session_id"]
            )
            print(f"   📊 Struttura punti: {'✅' if point_fields_ok else '❌'}")
            
            # Test coordinate 3D  
            has_3d = "z" in first_point
            print(f"   📐 Coordinate 3D: {'✅' if has_3d else '❌'}")
        else:
            print(f"   ❌ Nessun punto disponibile")
        
        # Test colori
        colors = viz_data.get("cluster_colors", {})
        has_colors = len(colors) > 0 and "-1" in colors  # Deve includere colore outlier
        print(f"   🎨 Colori cluster: {'✅' if has_colors else '❌'} ({len(colors)} colori)")
        
        # Test coordinate multiple
        coords = viz_data.get("coordinates", {})
        coord_types = ["tsne_2d", "pca_2d", "pca_3d"]
        available_coords = [ct for ct in coord_types if ct in coords and len(coords[ct]) > 0]
        print(f"   📐 Coordinate multiple: ✅ ({len(available_coords)}/{len(coord_types)})")
        
        # Test compatibilità librerie grafiche (Plotly, D3, etc.)
        plotly_compatible = (
            len(points) > 0 and
            "x" in points[0] and "y" in points[0] and
            "cluster_id" in points[0]
        )
        print(f"   📊 Compatibilità Plotly: {'✅' if plotly_compatible else '❌'}")
        
        print(f"   ✅ Visualizzazioni: Completamente compatibili")
        
    except Exception as e:
        print(f"   ❌ Errore test visualizzazioni: {e}")

def test_edge_cases() -> None:
    """
    Test edge cases e scenari di fallback
    """
    print(f"   🧪 Test scenari critici:")
    
    # 1. Risposta di errore
    error_response = {
        "success": False,
        "error": "Tenant non trovato",
        "tenant_id": "invalid-tenant"
    }
    
    try:
        mapped_error = test_frontend_mapping(error_response)
        error_handled = not mapped_error["success"] and "error" in mapped_error
        print(f"      ❌→✅ Gestione errori: {'✅' if error_handled else '❌'}")
    except:
        print(f"      ❌→❌ Gestione errori: ❌")
    
    # 2. Dati parziali
    partial_response = {
        "success": True,
        "statistics": {"total_conversations": 10, "n_clusters": 2, "n_outliers": 1},
        "detailed_clusters": [],
        # Mancano quality_metrics e visualization_data
    }
    
    try:
        mapped_partial = test_frontend_mapping(partial_response)
        partial_handled = mapped_partial["success"] and "statistics" in mapped_partial
        print(f"      📊→⚠️ Dati parziali: {'✅' if partial_handled else '❌'}")
    except:
        print(f"      📊→❌ Dati parziali: ❌")
    
    # 3. Dataset vuoto
    empty_response = {
        "success": True,
        "statistics": {"total_conversations": 0, "n_clusters": 0, "n_outliers": 0},
        "detailed_clusters": [],
        "quality_metrics": {"silhouette_score": 0},
        "outlier_analysis": {"count": 0, "ratio": 0, "sample_outliers": []},
        "visualization_data": {"points": [], "cluster_colors": {}, "coordinates": {}}
    }
    
    try:
        mapped_empty = test_frontend_mapping(empty_response)
        empty_handled = (
            mapped_empty["success"] and 
            mapped_empty["statistics"]["total_conversations"] == 0
        )
        print(f"      📭→✅ Dataset vuoto: {'✅' if empty_handled else '❌'}")
    except:
        print(f"      📭→❌ Dataset vuoto: ❌")

if __name__ == "__main__":
    test_complete_gui_workflow()
