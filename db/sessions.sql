CREATE TABLE sessions(
    id SERIAL PRIMARY KEY,
    user_id INTEGER REFERENCES users(id) ON DELETE CASCADE,
    video_path TEXT NOT NULL,
    fps INTEGER DEFAULT 30,
    total_reps INTEGER NOT NULL,

    avg_depth FLOAT,
    min_knee_angle FLOAT,
    tempo FLOAT,
    alignment FLOAT,
    bar_dev FLOAT,

    created_at TIMESTAMP DEFAULT now()
);

CREATE INDEX idx_sessions_user_id ON sessions(user_id);
CREATE INDEX idx_sessions_created_at ON sessions(created_at);