CREATE TABLE sessions{
    id SERIAL PRIMARY KEY,
    user_id INTEGER REFERENCES users(id) ON DELETE CASCADE,
    video_path TEXT,
    fps INTEGER,
    total_reps, INTEGER
    created_at TIMESTAMP DEFAULT now()
};