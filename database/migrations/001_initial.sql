-- Initial database schema for grid oscillation predictions
CREATE TABLE IF NOT EXISTS predictions (
    id SERIAL PRIMARY KEY,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    risk_score FLOAT NOT NULL,
    alert_level VARCHAR(20),
    oscillation_mode INTEGER,
    confidence FLOAT,
    raw_data JSONB
);

CREATE INDEX idx_predictions_timestamp ON predictions(timestamp);
CREATE INDEX idx_predictions_risk ON predictions(risk_score);
