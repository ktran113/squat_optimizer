CREATE TABLE rep_metrics (
    id SERIAL PRIMARY KEY,
    session_id INTEGER REFERENCES sessions(id) on DELETE CASCADE,
    rep_number INTEGER NOT NULL,

    bottom_frame INTEGER,
    start_frame INTEGER,
    end_frame INTEGER,

    knee_angle FLOAT,
    depth_value FLOAT,
    depth_quality TEXT CHECK (depth_quality IN ('below', 'parallel', 'partial')),
    bar_path_deviation FLOAT,
    tempo FLOAT,
    hip_heel_aligned BOOLEAN,

    created_at TIMESTAMP DEFAULT NOW()
);
CREATE INDEX idx_rep_metrics_session_id ON rep_metrics(session_id);
