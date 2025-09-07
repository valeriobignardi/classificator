        try:
            # Usa clustering fornito come parametri
            n_clusters = len([l for l in cluster_labels if l != -1])
            n_outliers = sum(1 for l in cluster_labels if l == -1)
            
            print(f"   📈 Cluster trovati: {n_clusters}")
            print(f"   🔍 Outliers: {n_outliers}")
            
            # STEP 2: Selezione rappresentanti per ogni cluster 
            print(f"👥 STEP 2: Selezione rappresentanti per classificazione...")
            representatives = {}
            suggested_labels = {}
