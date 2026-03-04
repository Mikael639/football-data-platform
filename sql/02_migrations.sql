ALTER TABLE pipeline_run_log
ADD COLUMN IF NOT EXISTS metrics_jsonb JSONB;

ALTER TABLE pipeline_run_log
ADD COLUMN IF NOT EXISTS volumes_jsonb JSONB;

ALTER TABLE data_quality_check
ADD COLUMN IF NOT EXISTS severity VARCHAR(20);

CREATE INDEX IF NOT EXISTS idx_pipeline_run_log_status
    ON pipeline_run_log(status);

CREATE INDEX IF NOT EXISTS idx_data_quality_check_run_id_created_at
    ON data_quality_check(run_id, created_at DESC);
