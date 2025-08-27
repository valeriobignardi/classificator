#!/usr/bin/env python3
"""
File: tokenization_utils.py
Autore: GitHub Copilot
Data creazione: 2025-08-26
Descrizione: Utilità per tokenizzazione con tiktoken per gestione conversazioni lunghe

Storia aggiornamenti:
2025-08-26 - Creazione iniziale con supporto tiktoken per OpenAI
"""

import os
import yaml
import tiktoken
from typing import Union, List, Tuple, Dict, Any
import logging


class TokenizationManager:
    """
    Scopo: Gestisce la tokenizzazione delle conversazioni per prevenire errori di token limit
    
    Parametri input:
    - config_path: Percorso del file di configurazione YAML
    - model_name: Nome del modello tiktoken (default: cl100k_base per GPT-3.5/4)
    - max_tokens: Limite massimo di token (da config.yaml)
    
    Valori di ritorno:
    - Testi tokenizzati e troncati se necessario
    - Debug dettagliato dei conteggi token
    
    Data ultima modifica: 2025-08-26
    """
    
    def __init__(self, config_path: str = None):
        """
        Inizializza il TokenizationManager
        
        Args:
            config_path: Percorso del file di configurazione
        """
        self.config_path = config_path or os.path.join(
            os.path.dirname(os.path.dirname(__file__)), 'config.yaml'
        )
        
        # Carica configurazione
        self._load_config()
        
        # Inizializza tokenizer
        self.encoding = tiktoken.get_encoding(self.model_name)
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        
        print(f"🔧 TokenizationManager inizializzato:")
        print(f"   📊 Modello tokenizer: {self.model_name}")
        print(f"   🔢 Limite massimo token: {self.max_tokens}")
        print(f"   ✂️  Strategia troncamento: {self.truncation_strategy}")
    
    def _load_config(self):
        """
        Carica la configurazione da config.yaml
        
        Ultima modifica: 2025-08-26
        """
        try:
            with open(self.config_path, 'r', encoding='utf-8') as file:
                config = yaml.safe_load(file)
            
            tokenization_config = config.get('tokenization', {})
            
            self.max_tokens = tokenization_config.get('max_tokens', 8000)
            self.model_name = tokenization_config.get('model_name', 'cl100k_base')
            self.truncation_strategy = tokenization_config.get('truncation_strategy', 'start')
            
        except Exception as e:
            print(f"⚠️  Errore caricamento config: {e}")
            # Valori di default
            self.max_tokens = 8000
            self.model_name = 'cl100k_base'
            self.truncation_strategy = 'start'
    
    def count_tokens(self, text: str) -> int:
        """
        Conta il numero di token in un testo
        
        Args:
            text: Testo da tokenizzare
            
        Returns:
            Numero di token
            
        Ultima modifica: 2025-08-26
        """
        if not text or not isinstance(text, str):
            return 0
        
        tokens = self.encoding.encode(text)
        return len(tokens)
    
    def truncate_text(self, text: str, max_tokens: int = None) -> Tuple[str, int, int]:
        """
        Tronca un testo per rispettare il limite di token
        
        Args:
            text: Testo da troncare
            max_tokens: Limite massimo di token (se None, usa self.max_tokens)
            
        Returns:
            Tupla (testo_troncato, token_originali, token_finali)
            
        Ultima modifica: 2025-08-26
        """
        if not text or not isinstance(text, str):
            return "", 0, 0
        
        max_tokens = max_tokens or self.max_tokens
        
        # Conta token originali
        original_tokens = self.count_tokens(text)
        
        # Se già sotto il limite, restituisci il testo originale
        if original_tokens <= max_tokens:
            return text, original_tokens, original_tokens
        
        print(f"✂️  TRONCAMENTO NECESSARIO:")
        print(f"   📊 Token originali: {original_tokens}")
        print(f"   🎯 Target token: {max_tokens}")
        print(f"   ⚡ Strategia: {self.truncation_strategy}")
        
        # Tokenizza il testo
        tokens = self.encoding.encode(text)
        
        # Tronca secondo la strategia
        if self.truncation_strategy == 'start':
            # Mantieni l'inizio della conversazione
            truncated_tokens = tokens[:max_tokens]
        elif self.truncation_strategy == 'end':
            # Mantieni la fine della conversazione
            truncated_tokens = tokens[-max_tokens:]
        else:
            # Default: mantieni l'inizio
            truncated_tokens = tokens[:max_tokens]
        
        # Decodifica i token troncati
        truncated_text = self.encoding.decode(truncated_tokens)
        final_tokens = len(truncated_tokens)
        
        print(f"   ✅ Token finali: {final_tokens}")
        print(f"   📏 Caratteri rimossi: {len(text) - len(truncated_text)}")
        
        return truncated_text, original_tokens, final_tokens
    
    def process_conversations_for_clustering(self, texts: List[str], 
                                           session_ids: List[str] = None) -> Tuple[List[str], Dict[str, Any]]:
        """
        Elabora una lista di conversazioni per il clustering, troncando quelle troppo lunghe
        
        Args:
            texts: Lista di testi delle conversazioni
            session_ids: Lista opzionale di session_id corrispondenti
            
        Returns:
            Tupla (testi_processati, statistiche_debug)
            
        Ultima modifica: 2025-08-26
        """
        print(f"🔍 ELABORAZIONE CONVERSAZIONI PER CLUSTERING")
        print(f"   📊 Numero conversazioni: {len(texts)}")
        print(f"   🆔 Session IDs forniti: {'Sì' if session_ids else 'No'}")
        
        processed_texts = []
        stats = {
            'total_conversations': len(texts),
            'processed_count': len(texts),  # 🆕 Aggiunto per compatibilità BGE-M3
            'truncated_count': 0,
            'total_tokens_original': 0,
            'total_tokens_final': 0,
            'max_tokens_found': 0,
            'min_tokens_found': float('inf'),
            'max_tokens': self.max_tokens,  # 🆕 Aggiunto per compatibilità BGE-M3
            'truncated_sessions': []
        }
        
        for i, text in enumerate(texts):
            session_id = session_ids[i] if session_ids and i < len(session_ids) else f"conv_{i+1}"
            
            # Elabora la conversazione
            truncated_text, original_tokens, final_tokens = self.truncate_text(text)
            
            processed_texts.append(truncated_text)
            
            # Aggiorna statistiche
            stats['total_tokens_original'] += original_tokens
            stats['total_tokens_final'] += final_tokens
            stats['max_tokens_found'] = max(stats['max_tokens_found'], original_tokens)
            stats['min_tokens_found'] = min(stats['min_tokens_found'], original_tokens)
            
            # Se è stato troncato
            if original_tokens > final_tokens:
                stats['truncated_count'] += 1
                stats['truncated_sessions'].append({
                    'session_id': session_id,
                    'original_tokens': original_tokens,
                    'final_tokens': final_tokens,
                    'reduction_percent': round((1 - final_tokens/original_tokens) * 100, 1)
                })
                
                print(f"   ✂️  Conversazione {session_id}: {original_tokens} → {final_tokens} token")
        
        # Stampa statistiche finali
        if stats['min_tokens_found'] == float('inf'):
            stats['min_tokens_found'] = 0
            
        print(f"\n📊 STATISTICHE ELABORAZIONE:")
        print(f"   🔢 Conversazioni totali: {stats['total_conversations']}")
        print(f"   ✂️  Conversazioni troncate: {stats['truncated_count']}")
        print(f"   📏 Token originali totali: {stats['total_tokens_original']:,}")
        print(f"   📐 Token finali totali: {stats['total_tokens_final']:,}")
        print(f"   📊 Range token: {stats['min_tokens_found']} - {stats['max_tokens_found']}")
        
        if stats['truncated_count'] > 0:
            reduction_percent = round((1 - stats['total_tokens_final']/stats['total_tokens_original']) * 100, 1)
            print(f"   💾 Riduzione totale: {reduction_percent}%")
        
        return processed_texts, stats
    
    def process_conversation_for_llm(self, conversation_text: str, 
                                   prompt_text: str, 
                                   session_id: str = None) -> Tuple[str, Dict[str, Any]]:
        """
        Elabora una conversazione per l'etichettatura LLM, considerando sia prompt che conversazione
        
        Args:
            conversation_text: Testo della conversazione
            prompt_text: Prompt da inviare all'LLM
            session_id: ID della sessione (opzionale, per debug)
            
        Returns:
            Tupla (conversazione_troncata, statistiche_debug)
            
        Ultima modifica: 2025-08-26
        """
        session_id = session_id or "llm_conversation"
        
        print(f"🤖 ELABORAZIONE CONVERSAZIONE PER LLM")
        print(f"   🆔 Session ID: {session_id}")
        
        # Conta token del prompt
        prompt_tokens = self.count_tokens(prompt_text)
        conversation_tokens = self.count_tokens(conversation_text)
        total_tokens = prompt_tokens + conversation_tokens
        
        print(f"   📝 Token prompt: {prompt_tokens}")
        print(f"   💬 Token conversazione: {conversation_tokens}")
        print(f"   📊 Token totali: {total_tokens}")
        print(f"   🎯 Limite massimo: {self.max_tokens}")
        
        stats = {
            'session_id': session_id,
            'prompt_tokens': prompt_tokens,
            'conversation_tokens_original': conversation_tokens,
            'total_tokens_original': total_tokens,
            'max_tokens_limit': self.max_tokens,
            'truncated': False,
            'conversation_tokens_final': conversation_tokens,
            'total_tokens_final': total_tokens
        }
        
        # Se supera il limite, tronca la conversazione
        if total_tokens > self.max_tokens:
            print(f"⚠️  SUPERAMENTO LIMITE TOKEN!")
            
            # Calcola quanti token può avere la conversazione
            max_conversation_tokens = self.max_tokens - prompt_tokens - 100  # Buffer sicurezza
            
            if max_conversation_tokens <= 0:
                print(f"❌ ERRORE: Prompt troppo lungo ({prompt_tokens} token)!")
                max_conversation_tokens = 100  # Minimo di emergenza
            
            print(f"   ✂️  Token disponibili per conversazione: {max_conversation_tokens}")
            
            # Tronca la conversazione
            truncated_conversation, _, final_tokens = self.truncate_text(
                conversation_text, max_conversation_tokens
            )
            
            stats.update({
                'truncated': True,
                'conversation_tokens_final': final_tokens,
                'total_tokens_final': prompt_tokens + final_tokens,
                'reduction_percent': round((1 - final_tokens/conversation_tokens) * 100, 1)
            })
            
            print(f"   ✅ Conversazione troncata: {conversation_tokens} → {final_tokens} token")
            print(f"   📊 Totale finale: {stats['total_tokens_final']} token")
            print(f"   💾 Riduzione conversazione: {stats['reduction_percent']}%")
            
            return truncated_conversation, stats
        
        else:
            print(f"   ✅ Conversazione OK: {total_tokens} ≤ {self.max_tokens}")
            return conversation_text, stats
    
    def get_token_info(self) -> Dict[str, Any]:
        """
        Restituisce informazioni sulla configurazione del tokenizer
        
        Returns:
            Dizionario con informazioni di configurazione
            
        Ultima modifica: 2025-08-26
        """
        return {
            'model_name': self.model_name,
            'max_tokens': self.max_tokens,
            'truncation_strategy': self.truncation_strategy,
            'encoding_name': self.encoding.name
        }
