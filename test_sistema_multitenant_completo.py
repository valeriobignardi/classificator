#!/usr/bin/env python3
"""
Test Sistema Multi-Tenant Completo
=================================

Script di test end-to-end per verificare che tutto il sistema multi-tenant
funzioni correttamente con tenant isolation e naming convention.

Test Coverage:
1. Backend Components (MongoDB, MySQL, API)
2. Pipeline Integration (EndToEndPipeline, SemanticMemory)
3. Fine-tuning Models (MistralFineTuningManager)
4. Frontend API Endpoints

Autore: Pipeline Multi-Tenant
Data: 21 Agosto 2025
"""

import os
import sys
import json
import time
import traceback
from datetime import datetime
from typing import Dict, List, Any, Optional

# Add project paths
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.extend([
    project_root,
    os.path.join(project_root, 'Pipeline'),
    os.path.join(project_root, 'FineTuning')
])

class MultiTenantSystemTester:
    """Tester completo per sistema multi-tenant"""
    
    def __init__(self):
        """Inizializza il tester"""
        self.results = {
            'backend_components': {},
            'pipeline_integration': {},
            'finetuning_models': {},
            'api_endpoints': {},
            'overall_success': False,
            'start_time': datetime.now().isoformat(),
            'errors': []
        }
        self.tenants_to_test = ['Humanitas', 'Alleanza', 'Boots']
        
    def print_header(self, title: str, level: int = 1):
        """Print formatted header"""
        symbols = ['🌐', '🧪', '🔧', '📊', '✅']
        symbol = symbols[min(level-1, len(symbols)-1)]
        separator = '=' * (50 - len(title)) if level == 1 else '-' * (40 - len(title))
        print(f"\n{symbol} {title} {separator}")
    
    def print_test(self, name: str, status: str, details: str = ""):
        """Print test result"""
        status_symbols = {
            'PASS': '✅',
            'FAIL': '❌', 
            'WARN': '⚠️',
            'INFO': 'ℹ️'
        }
        symbol = status_symbols.get(status, '❓')
        print(f"   {symbol} {name}: {status}")
        if details:
            print(f"      {details}")
    
    def test_backend_components(self) -> Dict[str, bool]:
        """Test 1: Backend Components"""
        self.print_header("Test Backend Components", 1)
        results = {}
        
        # Test MongoDB Reader
        try:
            from mongo_classification_reader import MongoClassificationReader
            reader = MongoClassificationReader()
            
            # Test tenant list
            tenants = reader.get_available_tenants()
            if len(tenants) > 0:
                self.print_test("MongoDB Tenant List", "PASS", f"Found {len(tenants)} tenants")
                results['mongo_tenant_list'] = True
            else:
                self.print_test("MongoDB Tenant List", "FAIL", "No tenants found")
                results['mongo_tenant_list'] = False
            
            # Test tenant mapping
            test_tenant = tenants[0] if tenants else None
            if test_tenant:
                tenant_name = test_tenant['nome']
                tenant_id = reader.get_tenant_id_from_name(tenant_name)
                tenant_name_back = reader.get_tenant_name_from_id(tenant_id)
                
                if tenant_name == tenant_name_back:
                    self.print_test("Tenant Mapping", "PASS", f"{tenant_name} ↔ {tenant_id[:8]}...")
                    results['tenant_mapping'] = True
                else:
                    self.print_test("Tenant Mapping", "FAIL", f"Mapping inconsistent")
                    results['tenant_mapping'] = False
            
            # Test naming generation
            for tenant in tenants[:2]:
                tenant_name = tenant['nome']
                
                # Test model naming
                model_name = reader.generate_model_name(tenant_name, 'test')
                expected_parts = [tenant_name.lower(), 'test']
                if all(part in model_name.lower() for part in expected_parts):
                    self.print_test(f"Model Naming ({tenant_name})", "PASS", f"{model_name}")
                    results[f'model_naming_{tenant_name}'] = True
                else:
                    self.print_test(f"Model Naming ({tenant_name})", "FAIL", f"Invalid format: {model_name}")
                    results[f'model_naming_{tenant_name}'] = False
                
                # Test cache path generation  
                cache_path = reader.generate_semantic_cache_path(tenant_name, 'test')
                if tenant_name.lower() in cache_path.lower() and 'semantic_cache' in cache_path:
                    self.print_test(f"Cache Path ({tenant_name})", "PASS", f"{cache_path}")
                    results[f'cache_path_{tenant_name}'] = True
                else:
                    self.print_test(f"Cache Path ({tenant_name})", "FAIL", f"Invalid path: {cache_path}")
                    results[f'cache_path_{tenant_name}'] = False
                    
        except Exception as e:
            self.print_test("MongoDB Reader", "FAIL", f"Error: {e}")
            results['mongo_reader'] = False
            self.results['errors'].append(f"MongoDB Reader: {e}")
        
        # Test API Server Endpoints
        try:
            import requests
            
            # Test tenants endpoint
            response = requests.get('http://localhost:5000/api/tenants', timeout=5)
            if response.status_code == 200:
                data = response.json()
                if data.get('success') and 'tenants' in data:
                    self.print_test("API Tenants Endpoint", "PASS", f"Returned {len(data['tenants'])} tenants")
                    results['api_tenants'] = True
                else:
                    self.print_test("API Tenants Endpoint", "FAIL", f"Invalid response format")
                    results['api_tenants'] = False
            else:
                self.print_test("API Tenants Endpoint", "FAIL", f"Status {response.status_code}")
                results['api_tenants'] = False
                
        except requests.exceptions.RequestException as e:
            self.print_test("API Server", "WARN", f"Server not running: {e}")
            results['api_server'] = False
        except Exception as e:
            self.print_test("API Server", "FAIL", f"Error: {e}")
            results['api_server'] = False
            self.results['errors'].append(f"API Server: {e}")
        
        self.results['backend_components'] = results
        return results
    
    def test_pipeline_integration(self) -> Dict[str, bool]:
        """Test 2: Pipeline Integration"""
        self.print_header("Test Pipeline Integration", 1)
        results = {}
        
        # Test EndToEndPipeline per diversi tenant
        for tenant_name in self.tenants_to_test[:2]:  # Test solo primi 2 per velocità
            try:
                from end_to_end_pipeline import EndToEndPipeline
                
                self.print_header(f"Pipeline Test: {tenant_name}", 2)
                
                pipeline = EndToEndPipeline(tenant_slug=tenant_name)
                
                # Test inizializzazione
                if hasattr(pipeline, 'semantic_memory') and hasattr(pipeline, 'tenant_slug'):
                    self.print_test(f"Pipeline Init ({tenant_name})", "PASS", 
                                  f"Tenant: {pipeline.tenant_slug}")
                    results[f'pipeline_init_{tenant_name}'] = True
                else:
                    self.print_test(f"Pipeline Init ({tenant_name})", "FAIL", "Missing attributes")
                    results[f'pipeline_init_{tenant_name}'] = False
                    continue
                
                # Test SemanticMemoryManager
                cache_path = pipeline.semantic_memory.cache_path
                if tenant_name.lower() in cache_path.lower():
                    self.print_test(f"SemanticMemory ({tenant_name})", "PASS", 
                                  f"Cache: {cache_path}")
                    results[f'semantic_memory_{tenant_name}'] = True
                else:
                    self.print_test(f"SemanticMemory ({tenant_name})", "FAIL", 
                                  f"Wrong cache path: {cache_path}")
                    results[f'semantic_memory_{tenant_name}'] = False
                
                # Test directory creation
                if os.path.exists(cache_path):
                    self.print_test(f"Cache Directory ({tenant_name})", "PASS", "Directory exists")
                    results[f'cache_dir_{tenant_name}'] = True
                else:
                    self.print_test(f"Cache Directory ({tenant_name})", "FAIL", "Directory not created")
                    results[f'cache_dir_{tenant_name}'] = False
                
                # Cleanup pipeline to free GPU memory
                del pipeline
                
            except Exception as e:
                self.print_test(f"Pipeline ({tenant_name})", "FAIL", f"Error: {e}")
                results[f'pipeline_{tenant_name}'] = False
                self.results['errors'].append(f"Pipeline {tenant_name}: {e}")
        
        self.results['pipeline_integration'] = results
        return results
    
    def test_finetuning_models(self) -> Dict[str, bool]:
        """Test 3: Fine-tuning Models"""
        self.print_header("Test Fine-tuning Models", 1)
        results = {}
        
        # Test MistralFineTuningManager
        for tenant_name in self.tenants_to_test[:2]:  # Test primi 2
            try:
                from mistral_finetuning_manager import MistralFineTuningManager
                
                self.print_header(f"Fine-tuning Test: {tenant_name}", 2)
                
                # Initialize con tenant awareness
                manager = MistralFineTuningManager(tenant_slug=tenant_name)
                
                # Test inizializzazione
                if hasattr(manager, 'tenant_slug') and manager.tenant_slug == tenant_name:
                    self.print_test(f"Manager Init ({tenant_name})", "PASS", 
                                  f"Tenant-aware: {manager.tenant_slug}")
                    results[f'manager_init_{tenant_name}'] = True
                else:
                    self.print_test(f"Manager Init ({tenant_name})", "FAIL", 
                                  f"Not tenant-aware")
                    results[f'manager_init_{tenant_name}'] = False
                    continue
                
                # Test model naming
                model_name = manager.generate_model_name(tenant_name, 'test')
                expected_parts = [tenant_name.lower(), 'test']
                if all(part in model_name.lower() for part in expected_parts):
                    self.print_test(f"Model Naming ({tenant_name})", "PASS", 
                                  f"{model_name}")
                    results[f'finetuning_naming_{tenant_name}'] = True
                else:
                    self.print_test(f"Model Naming ({tenant_name})", "FAIL", 
                                  f"Invalid format: {model_name}")
                    results[f'finetuning_naming_{tenant_name}'] = False
                
                # Test models directory structure
                if os.path.exists(manager.models_dir):
                    dir_name = os.path.basename(manager.models_dir)
                    if tenant_name.lower() in dir_name.lower():
                        self.print_test(f"Models Directory ({tenant_name})", "PASS", 
                                      f"Tenant-specific: {dir_name}")
                        results[f'models_dir_{tenant_name}'] = True
                    else:
                        self.print_test(f"Models Directory ({tenant_name})", "WARN", 
                                      f"Not tenant-specific: {dir_name}")
                        results[f'models_dir_{tenant_name}'] = False
                else:
                    self.print_test(f"Models Directory ({tenant_name})", "FAIL", 
                                  f"Directory not created")
                    results[f'models_dir_{tenant_name}'] = False
                
                # Test model registry file
                registry_file = os.path.basename(manager.model_registry_path)
                if tenant_name.lower() in registry_file.lower():
                    self.print_test(f"Model Registry ({tenant_name})", "PASS", 
                                  f"Tenant-specific: {registry_file}")
                    results[f'model_registry_{tenant_name}'] = True
                else:
                    self.print_test(f"Model Registry ({tenant_name})", "WARN", 
                                  f"Not tenant-specific: {registry_file}")
                    results[f'model_registry_{tenant_name}'] = False
                
            except Exception as e:
                self.print_test(f"Fine-tuning ({tenant_name})", "FAIL", f"Error: {e}")
                results[f'finetuning_{tenant_name}'] = False
                self.results['errors'].append(f"Fine-tuning {tenant_name}: {e}")
        
        self.results['finetuning_models'] = results
        return results
    
    def test_api_endpoints(self) -> Dict[str, bool]:
        """Test 4: API Endpoints"""
        self.print_header("Test API Endpoints", 1)
        results = {}
        
        try:
            import requests
            base_url = 'http://localhost:5000'
            
            # Test tenants endpoint
            try:
                response = requests.get(f'{base_url}/api/tenants', timeout=5)
                if response.status_code == 200:
                    data = response.json()
                    if data.get('success'):
                        tenants = data.get('tenants', [])
                        self.print_test("GET /api/tenants", "PASS", f"Returns {len(tenants)} tenants")
                        results['get_tenants'] = True
                        
                        # Test tenant-specific endpoints for first tenant
                        if tenants:
                            test_tenant = tenants[0]['nome'].lower()
                            
                            # Test review cases endpoint
                            try:
                                response = requests.get(f'{base_url}/api/review/{test_tenant}/cases', timeout=5)
                                status = "PASS" if response.status_code in [200, 404] else "FAIL"
                                self.print_test(f"GET /api/review/{test_tenant}/cases", status, 
                                              f"Status: {response.status_code}")
                                results[f'review_cases_{test_tenant}'] = (status == "PASS")
                            except Exception as e:
                                self.print_test(f"GET /api/review/{test_tenant}/cases", "FAIL", f"Error: {e}")
                                results[f'review_cases_{test_tenant}'] = False
                    else:
                        self.print_test("GET /api/tenants", "FAIL", "success=false")
                        results['get_tenants'] = False
                else:
                    self.print_test("GET /api/tenants", "FAIL", f"Status: {response.status_code}")
                    results['get_tenants'] = False
            except Exception as e:
                self.print_test("API Tenants Endpoint", "FAIL", f"Error: {e}")
                results['get_tenants'] = False
                
        except Exception as e:
            self.print_test("API Testing", "FAIL", f"Error: {e}")
            results['api_testing'] = False
            self.results['errors'].append(f"API Testing: {e}")
        
        self.results['api_endpoints'] = results
        return results
    
    def test_tenant_isolation(self) -> Dict[str, bool]:
        """Test 5: Tenant Isolation"""
        self.print_header("Test Tenant Isolation", 1)
        results = {}
        
        try:
            from mongo_classification_reader import MongoClassificationReader
            reader = MongoClassificationReader()
            tenants = reader.get_available_tenants()
            
            if len(tenants) < 2:
                self.print_test("Tenant Isolation", "SKIP", "Need at least 2 tenants")
                results['isolation_test'] = False
                return results
            
            tenant1 = tenants[0]['nome']
            tenant2 = tenants[1]['nome']
            
            # Test collection names are different
            collection1 = reader.get_collection_name(tenant1)
            collection2 = reader.get_collection_name(tenant2)
            
            if collection1 != collection2:
                self.print_test("Collection Isolation", "PASS", 
                              f"{tenant1}: {collection1} ≠ {tenant2}: {collection2}")
                results['collection_isolation'] = True
            else:
                self.print_test("Collection Isolation", "FAIL", "Same collection names")
                results['collection_isolation'] = False
            
            # Test model names are different
            model1 = reader.generate_model_name(tenant1, 'test')
            model2 = reader.generate_model_name(tenant2, 'test')
            
            if model1 != model2:
                self.print_test("Model Name Isolation", "PASS", 
                              f"Different model names")
                results['model_isolation'] = True
            else:
                self.print_test("Model Name Isolation", "FAIL", "Same model names")
                results['model_isolation'] = False
            
            # Test cache paths are different
            cache1 = reader.generate_semantic_cache_path(tenant1, 'memory')
            cache2 = reader.generate_semantic_cache_path(tenant2, 'memory')
            
            if cache1 != cache2:
                self.print_test("Cache Path Isolation", "PASS", 
                              f"Different cache paths")
                results['cache_isolation'] = True
            else:
                self.print_test("Cache Path Isolation", "FAIL", "Same cache paths")
                results['cache_isolation'] = False
                
        except Exception as e:
            self.print_test("Tenant Isolation", "FAIL", f"Error: {e}")
            results['isolation_test'] = False
            self.results['errors'].append(f"Tenant Isolation: {e}")
        
        self.results['tenant_isolation'] = results
        return results
    
    def run_all_tests(self) -> Dict[str, Any]:
        """Run all tests and return results"""
        self.print_header("Sistema Multi-Tenant - Test Completo", 0)
        
        # Run all test suites
        backend_results = self.test_backend_components()
        pipeline_results = self.test_pipeline_integration()
        finetuning_results = self.test_finetuning_models()
        api_results = self.test_api_endpoints()
        isolation_results = self.test_tenant_isolation()
        
        # Calculate overall success
        all_results = [backend_results, pipeline_results, finetuning_results, api_results, isolation_results]
        total_tests = sum(len(r) for r in all_results)
        passed_tests = sum(sum(1 for v in r.values() if v) for r in all_results)
        
        success_rate = (passed_tests / total_tests * 100) if total_tests > 0 else 0
        self.results['overall_success'] = success_rate >= 75  # 75% success rate
        self.results['test_summary'] = {
            'total_tests': total_tests,
            'passed_tests': passed_tests,
            'success_rate': success_rate
        }
        self.results['end_time'] = datetime.now().isoformat()
        
        # Print summary
        self.print_header("Riassunto Test", 1)
        self.print_test("Test Totali", "INFO", f"{total_tests}")
        self.print_test("Test Superati", "INFO", f"{passed_tests}")
        self.print_test("Tasso di Successo", "PASS" if success_rate >= 75 else "WARN", f"{success_rate:.1f}%")
        
        if self.results['errors']:
            self.print_test("Errori Trovati", "WARN", f"{len(self.results['errors'])}")
            for error in self.results['errors']:
                print(f"      • {error}")
        
        overall_status = "✅ SISTEMA FUNZIONANTE" if self.results['overall_success'] else "⚠️ NECESSARI MIGLIORAMENTI"
        print(f"\n🎯 {overall_status}")
        
        return self.results
    
    def save_results(self, output_file: str = None):
        """Save test results to JSON file"""
        if output_file is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            output_file = f"test_results_multitenant_{timestamp}.json"
        
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(self.results, f, indent=2, ensure_ascii=False, default=str)
            self.print_test("Save Results", "PASS", f"Saved to {output_file}")
        except Exception as e:
            self.print_test("Save Results", "FAIL", f"Error: {e}")


def main():
    """Main test function"""
    print("🚀 Avvio Test Sistema Multi-Tenant")
    
    tester = MultiTenantSystemTester()
    
    try:
        results = tester.run_all_tests()
        tester.save_results()
        
        # Exit with appropriate code
        exit_code = 0 if results['overall_success'] else 1
        return exit_code
        
    except KeyboardInterrupt:
        print("\n❌ Test interrotto dall'utente")
        return 130
    except Exception as e:
        print(f"\n❌ Errore grave durante i test: {e}")
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
