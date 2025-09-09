#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Test diretto del tracing batch senza import pesanti
"""

import asyncio
import sys
import os
from datetime import datetime

# Aggiungi il percorso del progetto
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def trace_all(function_name: str, action: str = "ENTER", called_from: str = None, **kwargs):
    """Funzione di tracing diretta per test"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
    
    message = f"[{timestamp}] {action} {function_name}"
    if called_from:
        message += f" (from {called_from})"
        
    if kwargs:
        params = ", ".join([f"{k}={v}" for k, v in kwargs.items()])
        message += f" - {params}"
        
    print(f"🔍 TRACE: {message}")


async def test_direct_batch():
    """Test diretto per verificare il tracing del batch processing"""
    print("🧪 Test Diretto Batch Processing")
    
    # Simula una chiamata batch con tracing
    requests = [{"id": 1}, {"id": 2}, {"id": 3}]
    batch_size = 3
    
    trace_all("direct_batch_test", "ENTER", 
             requests_count=len(requests), batch_size=batch_size)
    
    # Simula elaborazione
    await asyncio.sleep(0.1)  # Simula tempo di elaborazione
    
    trace_all("direct_batch_test", "INFO", 
             message="processing_parallel_requests")
    
    # Simula completamento
    success_count = len(requests)
    error_count = 0
    execution_time = 0.15
    
    trace_all("direct_batch_test", "EXIT",
             success=success_count, errors=error_count, 
             execution_time=f"{execution_time:.2f}s")
    
    print(f"✅ Test completato - {success_count} successi")


if __name__ == "__main__":
    print("🚀 Test Tracing Diretto")
    asyncio.run(test_direct_batch())
    print("✅ Test completato")
