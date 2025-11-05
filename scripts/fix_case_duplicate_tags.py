#!/usr/bin/env python3
"""
Rimuove duplicati di tag che differiscono solo per MAIUSCOLO/minuscolo.

Strategia:
- Per ogni tenant, individua gruppi con stesso LOWER(tag_name)
- Sceglie forma canonica UPPER(lower_name)
- Aggiorna sia `tags.tag_name` sia `session_classifications.tag_name`
- Elimina i duplicati lasciando un solo record per (tenant_id, UPPER(name))

Prerequisiti: MySQL 8.x, config tag_database in config.yaml
"""

import os
import yaml
import mysql.connector
from mysql.connector import Error

ROOT = os.path.dirname(os.path.dirname(__file__))


def load_db_config():
    with open(os.path.join(ROOT, 'config.yaml'), 'r') as f:
        cfg = yaml.safe_load(f)
    return cfg['tag_database']


def main():
    cfg = load_db_config()
    conn = mysql.connector.connect(
        host=cfg['host'], port=cfg['port'], user=cfg['user'],
        password=cfg['password'], database=cfg['database'], autocommit=False
    )
    cur = conn.cursor(dictionary=True)

    try:
        # Trova gruppi duplicati per tenant e lower(tag_name)
        cur.execute(
            """
            SELECT tenant_id, LOWER(tag_name) AS key_name,
                   GROUP_CONCAT(id ORDER BY id) AS ids,
                   GROUP_CONCAT(tag_name ORDER BY id) AS names,
                   COUNT(*) AS cnt
            FROM tags
            GROUP BY tenant_id, key_name
            HAVING cnt > 1
            """
        )
        dup_groups = cur.fetchall()

        if not dup_groups:
            print("✅ Nessun duplicato case-insensitive trovato.")
            return

        print(f"🔧 Gruppi duplicati trovati: {len(dup_groups)}")

        for g in dup_groups:
            tenant_id = g['tenant_id']
            key_name = g['key_name']
            canonical = key_name.upper()
            ids = [int(x) for x in (g['ids'] or '').split(',') if x]
            keep_id = ids[0]

            print(f"  - Tenant {tenant_id} → '{key_name}' → canonical '{canonical}' (ids: {ids})")

            # Aggiorna tag_name alla forma canonica per tutto il gruppo
            cur.execute(
                """
                UPDATE tags
                SET tag_name = %s
                WHERE tenant_id = %s AND LOWER(tag_name) = %s
                """,
                (canonical, tenant_id, key_name)
            )

            # Aggiorna eventuali riferimenti in session_classifications
            try:
                cur.execute(
                    """
                    UPDATE session_classifications
                    SET tag_name = %s
                    WHERE tenant_id = %s AND LOWER(tag_name) = %s
                    """,
                    (canonical, tenant_id, key_name)
                )
            except Error as e:
                # Tabella potrebbe non esistere in alcuni ambienti
                print(f"    ℹ️ Skip update session_classifications: {e}")

            # Elimina duplicati residui lasciando una sola riga
            cur.execute(
                """
                DELETE t1 FROM tags t1
                JOIN tags t2
                  ON t1.tenant_id = t2.tenant_id
                 AND t1.tag_name = t2.tag_name
                 AND t1.id > t2.id
                WHERE t1.tenant_id = %s AND t1.tag_name = %s
                """,
                (tenant_id, canonical)
            )

        conn.commit()
        print("✅ Duplicati rimossi e nomi normalizzati in MAIUSCOLO.")

        # Prova a creare indice unico case-insensitive (MySQL 8 funzional index)
        try:
            cur.execute(
                """
                CREATE UNIQUE INDEX IF NOT EXISTS unique_tag_per_tenant_ci
                ON tags (tenant_id, (LOWER(tag_name)))
                """
            )
            print("🔒 Indice univoco case-insensitive creato (tenant_id, LOWER(tag_name)).")
        except Error as e:
            print(f"⚠️ Impossibile creare indice univoco CI (ok se non supportato): {e}")

    finally:
        cur.close()
        conn.close()


if __name__ == '__main__':
    main()

