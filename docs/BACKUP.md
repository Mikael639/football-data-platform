# Guide Backup (Production)

Ce guide decrit la sauvegarde et la restauration chiffrees PostgreSQL.

Scripts disponibles :

- `scripts/backup_postgres_encrypted.sh`
- `scripts/restore_postgres_encrypted.sh`

## Sauvegarde

Linux/macOS (bash/zsh) :

```bash
export BACKUP_PASSPHRASE='change-me'
./scripts/backup_postgres_encrypted.sh
```

## Restauration

Linux/macOS (bash/zsh) :

```bash
export BACKUP_PASSPHRASE='change-me'
./scripts/restore_postgres_encrypted.sh <backup_file>
```

Remplacez `<backup_file>` par le chemin du fichier de sauvegarde chiffre.
