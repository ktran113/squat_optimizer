CREATE TABLE rep_metrics{
    id SERIAL Primary KEY,
    session_id INTEGER REFERENCES sessions(id) on DELETE CASCADE,
    rep_number INTEGER NOT NULL,

    knee_angle FLOAT,
    depth FLOAT,
    quality TEXT,
    bar_path_dev FLOAT,
    tempo FLOAT,

    created_at TIMESTAMP DEFAULT NOW()
};