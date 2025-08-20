import os
import sys

# Assicura che la root del progetto sia nel PYTHONPATH
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from Pipeline.end_to_end_pipeline import EndToEndPipeline
import json


def main():
    print('>>> Avvio pipeline reale...')
    pipe = EndToEndPipeline(tenant_slug='humanitas', auto_mode=False)
    print('>>> Pipeline inizializzata, eseguo...')
    try:
        res = pipe.esegui_pipeline_completa(
            giorni_indietro=7,
            limit=50,
            batch_size=16,
            interactive_mode=False,
            use_ensemble=True,
        )
        print('>>> Risultato:')
        print(json.dumps(res, ensure_ascii=False, indent=2))
    except Exception as e:
        print(f'❌ Errore esecuzione pipeline: {e}')
        raise


if __name__ == '__main__':
    main()
