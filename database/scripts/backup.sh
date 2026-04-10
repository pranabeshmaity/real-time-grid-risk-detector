#!/bin/bash
BACKUP_DIR="./database/backups"
DATE=$(date +%Y%m%d_%H%M%S)
mkdir -p $BACKUP_DIR
pg_dump grid_db > "$BACKUP_DIR/backup_$DATE.sql"
echo "Backup completed at $DATE" >> "$BACKUP_DIR/backup.log"
