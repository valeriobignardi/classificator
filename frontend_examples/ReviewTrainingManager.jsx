/**
 * React Component per gestione esempi LLM e riaddestramento modello
 * 
 * Autore: Valerio Bignardi
 * Data: 2025-09-07
 * Descrizione: Componente React per interfaccia review con funzionalità
 *              di salvataggio casi come esempi LLM e riaddestramento modello
 */

import React, { useState, useEffect } from 'react';
import axios from 'axios';

const API_BASE_URL = 'http://localhost:5001/api';

/**
 * Componente principale per gestione review e training
 * 
 * Scopo: Interfaccia per review dei casi e gestione del training
 * Props: tenantId (ID del tenant), sessionData (dati della sessione)
 * Stato: gestisce stato loading, errori, successi
 * Data ultima modifica: 2025-09-07
 */
const ReviewTrainingManager = ({ tenantId, sessionData }) => {
  const [isLoading, setIsLoading] = useState(false);
  const [trainingStatus, setTrainingStatus] = useState(null);
  const [message, setMessage] = useState(null);
  const [messageType, setMessageType] = useState('info'); // 'success', 'error', 'info'
  const [userNotes, setUserNotes] = useState(''); // Note dell'utente

  /**
   * Carica stato del training all'avvio del componente
   * 
   * Scopo: Recuperare informazioni attuali sul modello e training
   * Input: tenantId
   * Output: Aggiorna stato trainingStatus
   * Data ultima modifica: 2025-09-07
   */
  useEffect(() => {
    const loadTrainingStatus = async () => {
      try {
        const response = await axios.get(`${API_BASE_URL}/training/status/${tenantId}`);
        setTrainingStatus(response.data);
      } catch (error) {
        console.error('Errore caricamento status training:', error);
        setMessage('Errore nel caricamento dello stato del training');
        setMessageType('error');
      }
    };

    if (tenantId) {
      loadTrainingStatus();
    }
  }, [tenantId]);

  /**
   * Aggiunge un caso di review come esempio LLM
   * 
   * Scopo: Salvare conversazione corretta dall'umano come esempio di training
   * Input: sessionId, conversationText, correctLabel, category, userNotes
   * Output: Conferma di successo o errore
   * Data ultima modifica: 2025-09-07
   */
  const handleAddLLMExample = async (sessionId, conversationText, correctLabel, category = null, userNotes = null) => {
    setIsLoading(true);
    setMessage(null);

    try {
      console.log('📚 Aggiunta caso come esempio LLM...');
      
      const requestData = {
        session_id: sessionId,
        conversation_text: conversationText,
        etichetta_corretta: correctLabel,
        categoria: category,
        note_utente: userNotes,
        tenant_id: tenantId
      };

      const response = await axios.post(`${API_BASE_URL}/examples/add-review-case`, requestData);

      if (response.data.success) {
        const exampleId = response.data.esempio_id;
        setMessage(`✅ Caso salvato come esempio LLM (ID: ${exampleId})`);
        setMessageType('success');
        
        console.log('✅ Esempio LLM creato:', response.data);
        
        // Aggiorna stato training
        const statusResponse = await axios.get(`${API_BASE_URL}/training/status/${tenantId}`);
        setTrainingStatus(statusResponse.data);
        
      } else {
        setMessage(`❌ Errore: ${response.data.message}`);
        setMessageType('error');
      }

    } catch (error) {
      console.error('Errore aggiunta esempio LLM:', error);
      
      const errorMessage = error.response?.data?.error || error.message || 'Errore sconosciuto';
      setMessage(`❌ Errore nell'aggiunta dell'esempio: ${errorMessage}`);
      setMessageType('error');
      
    } finally {
      setIsLoading(false);
    }
  };

  /**
   * Riaddestra manualmente il modello ML
   * 
   * Scopo: Avviare riaddestramento del modello con dati corretti dalla review
   * Input: force (opzionale, ignora controlli di sicurezza)
   * Output: Conferma di successo con nuove metriche o errore
   * Data ultima modifica: 2025-09-07
   */
  const handleManualRetrain = async (force = false) => {
    setIsLoading(true);
    setMessage(null);

    try {
      console.log('🔄 Avvio riaddestramento manuale...');
      
      const requestData = {
        tenant_id: tenantId,
        force: force
      };

      const response = await axios.post(`${API_BASE_URL}/training/manual-retrain`, requestData);

      if (response.data.success) {
        const accuracy = response.data.accuracy;
        const trainingStats = response.data.training_stats;
        
        setMessage(`✅ Riaddestramento completato! Accuracy: ${(accuracy * 100).toFixed(1)}%`);
        setMessageType('success');
        
        console.log('✅ Riaddestramento completato:', response.data);
        
        // Aggiorna stato training
        const statusResponse = await axios.get(`${API_BASE_URL}/training/status/${tenantId}`);
        setTrainingStatus(statusResponse.data);
        
      } else {
        setMessage(`❌ Riaddestramento fallito: ${response.data.message}`);
        setMessageType('error');
      }

    } catch (error) {
      console.error('Errore riaddestramento:', error);
      
      const errorMessage = error.response?.data?.error || error.message || 'Errore sconosciuto';
      setMessage(`❌ Errore nel riaddestramento: ${errorMessage}`);
      setMessageType('error');
      
    } finally {
      setIsLoading(false);
    }
  };

  /**
   * Componente per visualizzazione messaggio di stato
   */
  const StatusMessage = () => {
    if (!message) return null;

    const messageClass = {
      success: 'bg-green-100 border-green-500 text-green-700',
      error: 'bg-red-100 border-red-500 text-red-700',
      info: 'bg-blue-100 border-blue-500 text-blue-700'
    }[messageType];

    return (
      <div className={`border-l-4 p-4 mb-4 ${messageClass}`}>
        {message}
      </div>
    );
  };

  /**
   * Componente per informazioni stato training
   */
  const TrainingStatusDisplay = () => {
    if (!trainingStatus) return null;

    return (
      <div className="bg-gray-100 p-4 rounded-lg mb-4">
        <h3 className="text-lg font-semibold mb-2">📊 Stato Training</h3>
        <div className="grid grid-cols-2 gap-4 text-sm">
          <div>
            <span className="font-medium">Tenant:</span> {trainingStatus.tenant_name}
          </div>
          <div>
            <span className="font-medium">Modello caricato:</span> 
            {trainingStatus.model_loaded ? '✅ Sì' : '❌ No'}
          </div>
          <div>
            <span className="font-medium">Esempi training:</span> {trainingStatus.training_samples}
          </div>
          <div>
            <span className="font-medium">Accuracy:</span> {(trainingStatus.accuracy * 100).toFixed(1)}%
          </div>
          <div>
            <span className="font-medium">Review pending:</span> {trainingStatus.pending_reviews}
          </div>
          <div>
            <span className="font-medium">Ultimo training:</span> 
            {new Date(trainingStatus.last_training).toLocaleString()}
          </div>
        </div>
      </div>
    );
  };

  return (
    <div className="review-training-manager p-6">
      <h2 className="text-2xl font-bold mb-6">🎓 Gestione Review e Training</h2>
      
      <StatusMessage />
      <TrainingStatusDisplay />
      
      {/* Sezione Aggiungi Esempio LLM */}
      <div className="mb-6 p-4 border rounded-lg">
        <h3 className="text-lg font-semibold mb-4">📚 Aggiungi Caso come Esempio LLM</h3>
        
        {sessionData && (
          <div className="space-y-4">
            <div className="bg-gray-50 p-3 rounded">
              <strong>Session ID:</strong> {sessionData.sessionId}
            </div>
            <div className="bg-gray-50 p-3 rounded">
              <strong>Conversazione:</strong>
              <div className="mt-2 text-sm">{sessionData.conversationText}</div>
            </div>
            <div className="bg-gray-50 p-3 rounded">
              <strong>Etichetta corretta:</strong> {sessionData.correctLabel}
            </div>
            
            <div className="space-y-2">
              <label htmlFor="userNotes" className="block text-sm font-medium text-gray-700">
                📋 Note aggiuntive (opzionale):
              </label>
              <textarea
                id="userNotes"
                value={userNotes}
                onChange={(e) => setUserNotes(e.target.value)}
                placeholder="Aggiungi note per spiegare perché questa classificazione è corretta..."
                className="w-full p-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                rows={3}
                maxLength={500}
              />
              <div className="text-xs text-gray-500">
                {userNotes.length}/500 caratteri
              </div>
            </div>
            
            <button
              onClick={() => handleAddLLMExample(
                sessionData.sessionId,
                sessionData.conversationText,
                sessionData.correctLabel,
                sessionData.category,
                userNotes.trim() || null
              )}
              disabled={isLoading}
              className="bg-blue-500 hover:bg-blue-700 text-white font-bold py-2 px-4 rounded disabled:opacity-50"
            >
              {isLoading ? '⏳ Salvando...' : '📚 AGGIUNGI COME ESEMPIO LLM'}
            </button>
          </div>
        )}
      </div>

      {/* Sezione Riaddestramento Modello */}
      <div className="p-4 border rounded-lg">
        <h3 className="text-lg font-semibold mb-4">🔄 Riaddestramento Modello</h3>
        
        <div className="space-y-4">
          <p className="text-sm text-gray-600">
            Riaddestra il modello ML utilizzando tutti i casi corretti dalla review umana.
            Questo migliorerà la precisione delle classificazioni future.
          </p>
          
          <div className="flex space-x-4">
            <button
              onClick={() => handleManualRetrain(false)}
              disabled={isLoading}
              className="bg-green-500 hover:bg-green-700 text-white font-bold py-2 px-4 rounded disabled:opacity-50"
            >
              {isLoading ? '⏳ Riaddestramento...' : '🔄 RIADDESTRA MODELLO'}
            </button>
            
            <button
              onClick={() => handleManualRetrain(true)}
              disabled={isLoading}
              className="bg-orange-500 hover:bg-orange-700 text-white font-bold py-2 px-4 rounded disabled:opacity-50"
            >
              {isLoading ? '⏳ Riaddestramento...' : '⚡ RIADDESTRA (FORCE)'}
            </button>
          </div>
          
          <p className="text-xs text-gray-500">
            Il pulsante "FORCE" ignora i controlli di sicurezza e forza il riaddestramento.
          </p>
        </div>
      </div>
    </div>
  );
};

/**
 * Hook personalizzato per gestione delle operazioni sui modelli
 */
export const useModelTraining = (tenantId) => {
  const [isTraining, setIsTraining] = useState(false);
  const [trainingHistory, setTrainingHistory] = useState([]);

  const addExample = async (sessionData, userNotes = null) => {
    // Implementazione per aggiunta esempio
    const response = await axios.post(`${API_BASE_URL}/examples/add-review-case`, {
      session_id: sessionData.sessionId,
      conversation_text: sessionData.conversationText,
      etichetta_corretta: sessionData.correctLabel,
      categoria: sessionData.category,
      note_utente: userNotes,
      tenant_id: tenantId
    });
    
    return response.data;
  };

  const retrainModel = async (force = false) => {
    setIsTraining(true);
    
    try {
      const response = await axios.post(`${API_BASE_URL}/training/manual-retrain`, {
        tenant_id: tenantId,
        force: force
      });
      
      // Aggiungi alla cronologia
      setTrainingHistory(prev => [...prev, {
        timestamp: new Date(),
        success: response.data.success,
        accuracy: response.data.accuracy,
        message: response.data.message
      }]);
      
      return response.data;
      
    } finally {
      setIsTraining(false);
    }
  };

  return {
    isTraining,
    trainingHistory,
    addExample,
    retrainModel
  };
};

export default ReviewTrainingManager;
